package hnsw

import (
	"runtime/debug"
	"testing"
)

func TestSearchIntoZeroAlloc(t *testing.T) {
	path := "test_search_into.hnsw"
	defer removeTestFiles(path)

	config := IndexConfig{
		Dims:     32,
		M:        8,
		MMax0:    16,
		MaxLevel: 8,
	}

	storage, err := NewStorage(path, config, 32)
	if err != nil {
		t.Fatalf("failed to create storage: %v", err)
	}
	defer storage.Close()

	idx := NewIndex(storage, L2, 32, 32)

	vecs := make([][]float32, 32)
	for i := range vecs {
		vecs[i] = make([]float32, 32)
		vecs[i][0] = float32(i)
		if err := idx.Insert(vecs[i], nil); err != nil {
			t.Fatalf("Insert %d failed: %v", i, err)
		}
	}

	query := vecs[16]
	dst := make([]Node, 0, 8)

	// Keep the pool warm and avoid GC churn while measuring steady-state reuse.
	allocs := func() float64 {
		oldGC := debug.SetGCPercent(-1)
		defer debug.SetGCPercent(oldGC)

		var err error
		dst, err = idx.SearchInto(dst[:0], query, 8)
		if err != nil {
			t.Fatalf("SearchInto warmup failed: %v", err)
		}

		return testing.AllocsPerRun(100, func() {
			var runErr error
			dst, runErr = idx.SearchInto(dst[:0], query, 8)
			if runErr != nil {
				t.Fatalf("SearchInto failed: %v", runErr)
			}
		})
	}()

	if allocs != 0 {
		t.Fatalf("expected zero allocations, got %f", allocs)
	}

	if len(dst) == 0 {
		t.Fatal("expected at least one result")
	}
}
