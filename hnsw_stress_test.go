package hnsw

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"sync"
	"testing"
)

func TestMixedOperationStress(t *testing.T) {
	path := t.TempDir() + "/test_mixed_ops.hnsw"

	config := IndexConfig{
		Dims:     32,
		M:        8,
		MMax0:    16,
		MaxLevel: 8,
	}

	storage, err := NewStorage(path, config, 256)
	if err != nil {
		t.Fatalf("failed to create storage: %v", err)
	}
	defer storage.Close()

	idx := NewIndex(storage, L2)
	idx.SetEfSearch(32)
	idx.SetEfConst(32)

	const (
		poolSize    = 1024
		initialVecs = 128
		ops         = 240
		searchOps   = 300
		searchers   = 4

		shortOps       = 50
		shortSearchOps = 50
	)

	opsCount := ops
	searchOpsCount := searchOps
	if testing.Short() {
		opsCount = shortOps
		searchOpsCount = shortSearchOps
	}

	vecPool := make([][]float32, poolSize)
	for i := range vecPool {
		vec := make([]float32, config.Dims)
		vec[0] = float32(i)
		for j := 1; j < len(vec); j++ {
			vec[j] = float32((i*31+j*17)%997) / 997
		}
		vecPool[i] = vec
	}

	var (
		liveMu      sync.Mutex
		liveIDs     = make([]uint32, 0, poolSize)
		liveVectors = make(map[uint32][]float32, poolSize)
		nextVec     = 0
	)

	recordInsert := func(ids []uint32, vecs [][]float32) {
		liveMu.Lock()
		defer liveMu.Unlock()
		liveIDs = append(liveIDs, ids...)
		for i, id := range ids {
			liveVectors[id] = vecs[i]
		}
	}

	removeIDs := func(ids []uint32) {
		liveMu.Lock()
		defer liveMu.Unlock()

		if len(ids) == 0 {
			return
		}

		drop := make(map[uint32]struct{}, len(ids))
		for _, id := range ids {
			drop[id] = struct{}{}
			delete(liveVectors, id)
		}

		filtered := liveIDs[:0]
		for _, id := range liveIDs {
			if _, ok := drop[id]; !ok {
				filtered = append(filtered, id)
			}
		}
		liveIDs = filtered
	}

	insertBatch := func(vecs [][]float32) error {
		startID := uint32(idx.Len())
		if err := idx.BatchInsert(vecs, nil); err != nil {
			return err
		}

		ids := make([]uint32, len(vecs))
		for i := range vecs {
			ids[i] = startID + uint32(i)
		}
		recordInsert(ids, vecs)
		return nil
	}

	if err := insertBatch(vecPool[:initialVecs]); err != nil {
		t.Fatalf("initial batch insert failed: %v", err)
	}
	nextVec = initialVecs

	var (
		wg       sync.WaitGroup
		errMu    sync.Mutex
		firstErr error
	)

	recordErr := func(err error) {
		if err == nil {
			return
		}
		errMu.Lock()
		if firstErr == nil {
			firstErr = err
		}
		errMu.Unlock()
	}

	stopSearch := make(chan struct{})
	writerRNG := rand.New(rand.NewPCG(1, 2))

	for g := 0; g < searchers; g++ {
		wg.Add(1)
		go func(seed uint64) {
			defer wg.Done()
			rng := rand.New(rand.NewPCG(seed, seed+1))
			for i := 0; i < searchOpsCount; i++ {
				select {
				case <-stopSearch:
					return
				default:
				}

				query := vecPool[int(rng.Uint32N(poolSize))]
				if _, err := idx.Search(query, 10); err != nil {
					recordErr(fmt.Errorf("search failed: %w", err))
					return
				}
			}
		}(uint64(g + 10))
	}

	for step := 0; step < opsCount; step++ {
		switch op := writerRNG.Uint32N(100); {
		case op < 45 && nextVec < poolSize:
			batchSize := int(writerRNG.Uint32N(3)) + 1
			if nextVec+batchSize > poolSize {
				batchSize = poolSize - nextVec
			}
			if batchSize == 0 {
				continue
			}
			if err := insertBatch(vecPool[nextVec : nextVec+batchSize]); err != nil {
				recordErr(fmt.Errorf("batch insert failed: %w", err))
				break
			}
			nextVec += batchSize

		case op < 75:
			liveMu.Lock()
			if len(liveIDs) == 0 {
				liveMu.Unlock()
				continue
			}
			id := liveIDs[int(writerRNG.Uint32N(uint32(len(liveIDs))))]
			liveMu.Unlock()

			if n := idx.Delete(id); n == 0 {
				recordErr(fmt.Errorf("delete returned 0 for live id %d", id))
				break
			}
			removeIDs([]uint32{id})

		case op < 90:
			liveMu.Lock()
			if len(liveIDs) == 0 {
				liveMu.Unlock()
				continue
			}

			count := int(writerRNG.Uint32N(4)) + 1
			if count > len(liveIDs) {
				count = len(liveIDs)
			}

			ids := make([]uint32, 0, count)
			seen := make(map[uint32]struct{}, count)
			for len(ids) < count {
				id := liveIDs[int(writerRNG.Uint32N(uint32(len(liveIDs))))]
				if _, ok := seen[id]; ok {
					continue
				}
				seen[id] = struct{}{}
				ids = append(ids, id)
			}
			liveMu.Unlock()

			deleted, err := idx.BulkDelete(ids)
			if err != nil {
				recordErr(fmt.Errorf("bulk delete failed: %w", err))
				break
			}
			if deleted != len(ids) {
				recordErr(fmt.Errorf("bulk delete deleted %d ids, want %d", deleted, len(ids)))
				break
			}
			removeIDs(ids)

		default:
			query := vecPool[int(writerRNG.Uint32N(poolSize))]
			if _, err := idx.Search(query, 5); err != nil {
				recordErr(fmt.Errorf("writer search failed: %w", err))
				break
			}
		}

		errMu.Lock()
		if firstErr != nil {
			errMu.Unlock()
			break
		}
		errMu.Unlock()
	}

	close(stopSearch)
	wg.Wait()

	errMu.Lock()
	stressErr := firstErr
	errMu.Unlock()
	if stressErr != nil {
		t.Fatal(stressErr)
	}

	liveMu.Lock()
	liveSnapshot := slices.Clone(liveIDs)
	liveVecSnapshot := make(map[uint32][]float32, len(liveVectors))
	for id, vec := range liveVectors {
		liveVecSnapshot[id] = vec
	}
	liveMu.Unlock()

	if len(liveSnapshot) == 0 {
		t.Fatal("expected at least one live vector after mixed ops")
	}

	if err := idx.Vacuum(); err != nil {
		t.Fatalf("Vacuum failed after mixed ops: %v", err)
	}

	if idx.Len() != len(liveSnapshot) {
		t.Fatalf("expected len %d after vacuum, got %d", len(liveSnapshot), idx.Len())
	}

	checkCount := 5
	if len(liveSnapshot) < checkCount {
		checkCount = len(liveSnapshot)
	}
	for i := 0; i < checkCount; i++ {
		vec := liveVecSnapshot[liveSnapshot[i]]
		results, err := idx.Search(vec, 1)
		if err != nil {
			t.Fatalf("search after vacuum failed: %v", err)
		}
		if len(results) == 0 {
			t.Fatalf("expected search results after vacuum for live vector %d", liveSnapshot[i])
		}
		if results[0].Distance > 1e-4 {
			t.Fatalf(
				"expected live vector %d to match itself after vacuum, got distance %f",
				liveSnapshot[i],
				results[0].Distance,
			)
		}
	}
}
