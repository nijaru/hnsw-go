package hnsw

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"slices"
	"testing"
)

func loadSIFT10kBinary(t testing.TB) (vectors, queries [][]float32, groundTruth [][]int32) {
	t.Helper()
	path := "sift10k_test.bin"
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skip("sift10k_test.bin not found — run conversion first")
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}

	r := bytesReader{data: data}
	numVec := r.uint32()
	numQuery := r.uint32()
	dims := r.uint32()
	topK := r.uint32()

	t.Logf(
		"Loading SIFT 10k: %d vectors, %d queries, %d dims, topK=%d",
		numVec,
		numQuery,
		dims,
		topK,
	)

	vectors = make([][]float32, numVec)
	for i := range vectors {
		vectors[i] = r.float32s(dims)
	}

	queries = make([][]float32, numQuery)
	for i := range queries {
		queries[i] = r.float32s(dims)
	}

	groundTruth = make([][]int32, numQuery)
	for i := range groundTruth {
		groundTruth[i] = r.int32s(topK)
	}

	return
}

type bytesReader struct {
	data   []byte
	offset int
}

func (r *bytesReader) uint32() uint32 {
	v := binary.LittleEndian.Uint32(r.data[r.offset:])
	r.offset += 4
	return v
}

func (r *bytesReader) float32s(n uint32) []float32 {
	out := make([]float32, n)
	for i := range out {
		bits := binary.LittleEndian.Uint32(r.data[r.offset:])
		out[i] = math.Float32frombits(bits)
		r.offset += 4
	}
	return out
}

func (r *bytesReader) int32s(n uint32) []int32 {
	out := make([]int32, n)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint32(r.data[r.offset:]))
		r.offset += 4
	}
	return out
}

func recallAtK(results []Node, groundTruth []int32, k int) float64 {
	gtSet := make(map[uint32]struct{}, k)
	for i := 0; i < k && i < len(groundTruth); i++ {
		gtSet[uint32(groundTruth[i])] = struct{}{}
	}

	hits := 0
	for i := 0; i < k && i < len(results); i++ {
		if _, ok := gtSet[results[i].ID]; ok {
			hits++
		}
	}
	return float64(hits) / float64(k)
}

func TestSIFT10kRecall(t *testing.T) {
	vectors, queries, groundTruth := loadSIFT10kBinary(t)

	path := fmt.Sprintf("sift10k_recall_%d.hnsw", rand.Int64())
	defer removeTestFiles(path)

	config := IndexConfig{
		Dims:     128,
		M:        16,
		MMax0:    32,
		MaxLevel: 16,
	}

	storage, err := NewStorage(path, config, uint32(len(vectors)))
	if err != nil {
		t.Fatalf("create storage: %v", err)
	}
	defer storage.Close()

	idx := NewIndex(storage, L2, 200, 200)

	t.Logf("Inserting %d vectors...", len(vectors))
	for i, vec := range vectors {
		if err := idx.Insert(vec, nil); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}

	stats := idx.Stats()
	t.Logf("Built: %d nodes, maxLevel=%d", stats.NodeCount, stats.MaxLevel)

	k := 10
	totalRecall := 0.0
	for i, query := range queries {
		results, err := idx.Search(query, k)
		if err != nil {
			t.Fatalf("search %d: %v", i, err)
		}
		totalRecall += recallAtK(results, groundTruth[i], k)
	}

	avgRecall := totalRecall / float64(len(queries))
	t.Logf("Recall@%d: %.4f (%d queries)", k, avgRecall, len(queries))

	if avgRecall < 0.80 {
		t.Errorf("recall %.4f below threshold 0.80", avgRecall)
	}
}

func TestSIFT10kGroundTruthVerify(t *testing.T) {
	vectors, queries, groundTruth := loadSIFT10kBinary(t)

	// Verify ground truth for ALL queries by brute-force
	// using our exact L2 function (squared Euclidean).
	k := 10
	mismatches := 0
	for qi := 0; qi < len(queries); qi++ {
		query := queries[qi]

		type idDist struct {
			id   uint32
			dist float32
		}
		all := make([]idDist, len(vectors))
		for i, vec := range vectors {
			all[i] = idDist{id: uint32(i), dist: L2(query, vec)}
		}
		slices.SortFunc(all, func(a, b idDist) int {
			if a.dist < b.dist {
				return -1
			}
			if a.dist > b.dist {
				return 1
			}
			return 0
		})

		for j := 0; j < k; j++ {
			gtID := uint32(groundTruth[qi][j])
			bfID := all[j].id
			if gtID != bfID {
				gtDist := L2(query, vectors[gtID])
				bfDist := all[j].dist
				if gtDist != bfDist {
					mismatches++
					t.Errorf(
						"q=%d rank=%d: ground truth says id=%d (dist=%.6f) but brute force says id=%d (dist=%.6f)",
						qi,
						j,
						gtID,
						gtDist,
						bfID,
						bfDist,
					)
					if mismatches >= 10 {
						t.Fatalf("too many mismatches, stopping")
					}
				}
			}
		}
	}
	if mismatches == 0 {
		t.Logf(
			"Ground truth verified against brute force: all %d queries, top-%d each",
			len(queries),
			k,
		)
	}
}

func TestSIFT10kRecallDistribution(t *testing.T) {
	vectors, queries, groundTruth := loadSIFT10kBinary(t)

	path := fmt.Sprintf("sift10k_dist_%d.hnsw", rand.Int64())
	defer removeTestFiles(path)

	config := IndexConfig{
		Dims:     128,
		M:        16,
		MMax0:    32,
		MaxLevel: 16,
	}

	storage, err := NewStorage(path, config, uint32(len(vectors)))
	if err != nil {
		t.Fatalf("create storage: %v", err)
	}
	defer storage.Close()

	idx := NewIndex(storage, L2, 200, 200)
	for i, vec := range vectors {
		if err := idx.Insert(vec, nil); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}

	k := 10
	perfectCount := 0
	missCount := 0
	missedQueries := []string{}

	for qi, query := range queries {
		results, err := idx.Search(query, k)
		if err != nil {
			t.Fatalf("search %d: %v", qi, err)
		}

		recall := recallAtK(results, groundTruth[qi], k)
		if recall == 1.0 {
			perfectCount++
		} else {
			missCount++
			hits := int(recall * float64(k))
			missedQueries = append(missedQueries, fmt.Sprintf("  q=%d: recall=%.2f (%d/%d)", qi, recall, hits, k))
		}
	}

	t.Logf("Recall distribution: %d perfect (%.1f%%), %d imperfect (%.1f%%)",
		perfectCount, float64(perfectCount)/float64(len(queries))*100,
		missCount, float64(missCount)/float64(len(queries))*100)

	for _, m := range missedQueries {
		t.Log(m)
	}
}

func TestSIFT10kSearchVsBruteForce(t *testing.T) {
	vectors, queries, _ := loadSIFT10kBinary(t)

	path := fmt.Sprintf("sift10k_bf_%d.hnsw", rand.Int64())
	defer removeTestFiles(path)

	config := IndexConfig{
		Dims:     128,
		M:        16,
		MMax0:    32,
		MaxLevel: 16,
	}

	storage, err := NewStorage(path, config, uint32(len(vectors)))
	if err != nil {
		t.Fatalf("create storage: %v", err)
	}
	defer storage.Close()

	idx := NewIndex(storage, L2, 200, 200)
	for i, vec := range vectors {
		if err := idx.Insert(vec, nil); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}

	k := 10
	for qi := 0; qi < 50; qi++ {
		query := queries[qi]

		results, err := idx.Search(query, k)
		if err != nil {
			t.Fatalf("search %d: %v", qi, err)
		}

		// Brute force
		type idDist struct {
			id   uint32
			dist float32
		}
		all := make([]idDist, len(vectors))
		for i, vec := range vectors {
			all[i] = idDist{id: uint32(i), dist: L2(query, vec)}
		}
		slices.SortFunc(all, func(a, b idDist) int {
			if a.dist < b.dist {
				return -1
			}
			if a.dist > b.dist {
				return 1
			}
			return 0
		})

		for j := 0; j < k; j++ {
			hnswID := results[j].ID
			bfID := all[j].id
			if hnswID != bfID {
				hnswDist := L2(query, vectors[hnswID])
				bfDist := all[j].dist
				if hnswDist != bfDist {
					t.Errorf(
						"q=%d rank=%d: HNSW says id=%d (dist=%.6f) but brute force says id=%d (dist=%.6f)",
						qi,
						j,
						hnswID,
						hnswDist,
						bfID,
						bfDist,
					)
				}
			}
		}
	}
	t.Log("HNSW search matches brute force for first 50 queries")
}

func TestSIFT10kPersistence(t *testing.T) {
	vectors, _, _ := loadSIFT10kBinary(t)

	path := fmt.Sprintf("sift10k_persist_%d.hnsw", rand.Int64())
	defer removeTestFiles(path)

	config := IndexConfig{
		Dims:     128,
		M:        16,
		MMax0:    32,
		MaxLevel: 16,
	}

	func() {
		storage, err := NewStorage(path, config, uint32(len(vectors)))
		if err != nil {
			t.Fatalf("create storage: %v", err)
		}
		defer storage.Close()

		idx := NewIndex(storage, L2, 100, 100)
		for i, vec := range vectors {
			if err := idx.Insert(vec, nil); err != nil {
				t.Fatalf("insert %d: %v", i, err)
			}
		}
	}()

	storage2, err := NewStorage(path, config, 0)
	if err != nil {
		t.Fatalf("reopen storage: %v", err)
	}
	defer storage2.Close()

	idx2 := NewIndex(storage2, L2, 200, 200)
	stats := idx2.Stats()
	if stats.NodeCount != uint32(len(vectors)) {
		t.Errorf("expected %d nodes after reopen, got %d", len(vectors), stats.NodeCount)
	}

	query := vectors[0]
	results, err := idx2.Search(query, 5)
	if err != nil {
		t.Fatalf("search after reopen: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("no results after reopen")
	}
	t.Logf(
		"Persistence OK: %d nodes, top result ID=%d dist=%f",
		stats.NodeCount,
		results[0].ID,
		results[0].Distance,
	)
}
