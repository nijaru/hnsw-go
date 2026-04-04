package hnsw

import (
	"runtime/debug"
	"testing"
)

func TestAllowListContains(t *testing.T) {
	ids := NewAllowIDsSorted([]uint32{1, 3, 5, 7})
	if !ids.Contains(3) {
		t.Fatal("expected sorted-ID allow-list to contain 3")
	}
	if ids.Contains(2) {
		t.Fatal("expected sorted-ID allow-list to reject 2")
	}

	bits := NewAllowBitset([]uint64{0b10101010})
	if !bits.Contains(1) {
		t.Fatal("expected bitset allow-list to contain 1")
	}
	if bits.Contains(2) {
		t.Fatal("expected bitset allow-list to reject 2")
	}
}

func TestSearchAllowedZeroAlloc(t *testing.T) {
	path := "test_search_allowed.hnsw"
	defer removeTestFiles(path)

	config := IndexConfig{
		Dims:     8,
		M:        4,
		MMax0:    8,
		MaxLevel: 4,
	}

	storage, err := NewStorage(path, config, 16)
	if err != nil {
		t.Fatalf("failed to create storage: %v", err)
	}
	defer storage.Close()

	idx := NewIndex(storage, L2, 16, 16)

	vecs := make([][]float32, 6)
	for i := range vecs {
		vecs[i] = make([]float32, 8)
		vecs[i][0] = float32(i)
		if err := idx.Insert(vecs[i], nil); err != nil {
			t.Fatalf("Insert %d failed: %v", i, err)
		}
	}

	allow := NewAllowBitset([]uint64{0b010101})
	dst := make([]Node, 0, 3)

	oldGC := debug.SetGCPercent(-1)
	defer debug.SetGCPercent(oldGC)

	var warmErr error
	dst, warmErr = idx.SearchAllowedInto(dst[:0], vecs[0], 3, allow)
	if warmErr != nil {
		t.Fatalf("SearchAllowedInto warmup failed: %v", warmErr)
	}

	allocs := testing.AllocsPerRun(100, func() {
		var runErr error
		dst, runErr = idx.SearchAllowedInto(dst[:0], vecs[0], 3, allow)
		if runErr != nil {
			t.Fatalf("SearchAllowedInto failed: %v", runErr)
		}
	})
	if allocs != 0 {
		t.Fatalf("expected zero allocations, got %f", allocs)
	}

	if len(dst) == 0 {
		t.Fatal("expected at least one allowed result")
	}

	for _, n := range dst {
		if n.ID%2 != 0 {
			t.Fatalf("unexpected disallowed result %d", n.ID)
		}
	}
}
