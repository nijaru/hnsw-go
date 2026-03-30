package hnsw

import (
	"math/rand/v2"
	"os"
	"sync"
	"sync/atomic"
	"testing"
)

func TestHNSW(t *testing.T) {
	path := "test.hnsw"
	defer os.Remove(path)

	config := IndexConfig{
		Dims:     128,
		M:        16,
		MMax0:    32,
		MaxLevel: 16,
	}

	storage, err := NewStorage(path, config, 1000)
	if err != nil {
		t.Fatalf("failed to create storage: %v", err)
	}
	defer storage.Close()

	idx := NewIndex(storage, L2, 100, 100)

	vectors := make([][]float32, 100)
	for i := range vectors {
		vectors[i] = make([]float32, 128)
		for j := range vectors[i] {
			vectors[i][j] = rand.Float32()
		}
		if err := idx.Insert(vectors[i]); err != nil {
			t.Fatalf("failed to insert vector %d: %v", i, err)
		}
	}

	results, err := idx.Search(vectors[50], 5)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results) == 0 {
		t.Fatalf("expected results, got none")
	}

	if results[0].ID != 50 {
		t.Errorf(
			"expected closest result to be ID 50, got %d (dist=%f)",
			results[0].ID,
			results[0].Distance,
		)
	}
	if results[0].Distance != 0 {
		t.Errorf("expected distance 0, got %f", results[0].Distance)
	}
}

func TestStorageGrow(t *testing.T) {
	path := "test_grow.hnsw"
	defer os.Remove(path)

	config := IndexConfig{
		Dims:     4,
		M:        4,
		MMax0:    8,
		MaxLevel: 4,
	}

	storage, err := NewStorage(path, config, 10)
	if err != nil {
		t.Fatalf("failed to create storage: %v", err)
	}
	defer storage.Close()

	// Initial node count should be 0, Allocated should be 10
	if storage.readUint32(32) != 10 {
		t.Errorf("expected 10 allocated nodes, got %d", storage.readUint32(32))
	}
}

func TestConcurrentInsertSearch(t *testing.T) {
	path := "test_concurrent.hnsw"
	defer os.Remove(path)

	config := IndexConfig{
		Dims:     128,
		M:        16,
		MMax0:    32,
		MaxLevel: 16,
	}

	storage, err := NewStorage(path, config, 5000)
	if err != nil {
		t.Fatalf("create storage: %v", err)
	}
	defer storage.Close()

	idx := NewIndex(storage, L2, 50, 50)

	// Insert 100 vectors sequentially first (seed)
	for i := 0; i < 100; i++ {
		vec := make([]float32, 128)
		for j := range vec {
			vec[j] = rand.Float32()
		}
		if err := idx.Insert(vec); err != nil {
			t.Fatalf("seed insert %d: %v", i, err)
		}
	}

	// Concurrent: 8 inserters, 16 searchers, 500 ops each
	const inserters = 8
	const searchers = 16
	const opsPerGoroutine = 500

	var wg sync.WaitGroup
	var searchErrs, insertErrs atomic.Int32

	for s := 0; s < searchers; s++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			query := make([]float32, 128)
			for i := 0; i < opsPerGoroutine; i++ {
				for j := range query {
					query[j] = rand.Float32()
				}
				results, err := idx.Search(query, 10)
				if err != nil {
					searchErrs.Add(1)
					return
				}
				if results == nil {
					searchErrs.Add(1)
					return
				}
			}
		}()
	}

	for g := 0; g < inserters; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < opsPerGoroutine; i++ {
				vec := make([]float32, 128)
				for j := range vec {
					vec[j] = rand.Float32()
				}
				if err := idx.Insert(vec); err != nil {
					insertErrs.Add(1)
					return
				}
			}
		}()
	}

	wg.Wait()

	if se := searchErrs.Load(); se > 0 {
		t.Errorf("%d search errors during concurrent ops", se)
	}
	if ie := insertErrs.Load(); ie > 0 {
		t.Errorf("%d insert errors during concurrent ops", ie)
	}

	stats := idx.Stats()
	expected := uint32(100 + inserters*opsPerGoroutine)
	if stats.NodeCount != expected {
		t.Errorf("expected %d nodes, got %d", expected, stats.NodeCount)
	}

	// Final sanity search
	query := make([]float32, 128)
	for j := range query {
		query[j] = rand.Float32()
	}
	results, err := idx.Search(query, 10)
	if err != nil {
		t.Fatalf("post-concurrent search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("no results after concurrent ops")
	}
	t.Logf("Concurrent test passed: %d nodes, final search OK", stats.NodeCount)
}
