package hnsw

import (
	"runtime/debug"
	"testing"
)

func TestChooseSearchPlan(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		allowLen int
		k        int
		nodes    uint32
		efSearch int
		want     searchPlan
	}{
		{
			name:     "exact scan for tiny allow-list",
			allowLen: 8,
			k:        4,
			nodes:    1_000,
			efSearch: 32,
			want:     searchPlanExact,
		},
		{
			name:     "standard traversal for dense allow-list",
			allowLen: 800,
			k:        8,
			nodes:    1_000,
			efSearch: 32,
			want:     searchPlanStandard,
		},
		{
			name:     "filtered traversal for middle ground",
			allowLen: 128,
			k:        8,
			nodes:    1_000,
			efSearch: 32,
			want:     searchPlanFiltered,
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			got := chooseSearchPlan(tc.allowLen, tc.k, tc.nodes, tc.efSearch)
			if got != tc.want {
				t.Fatalf("chooseSearchPlan(%d, %d, %d, %d) = %v, want %v",
					tc.allowLen, tc.k, tc.nodes, tc.efSearch, got, tc.want,
				)
			}
		})
	}
}

func TestSearchPlannedZeroAlloc(t *testing.T) {
	path := t.TempDir() + "/test_search_planned.hnsw"

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

	idx := NewIndex(storage, L2)
	idx.SetEfSearch(16)
	idx.SetEfConst(16)

	vecs := make([][]float32, 6)
	for i := range vecs {
		vecs[i] = make([]float32, 8)
		vecs[i][0] = float32(i)
		if err := idx.Insert(vecs[i], nil); err != nil {
			t.Fatalf("Insert %d failed: %v", i, err)
		}
	}

	allow := NewAllowIDsSorted([]uint32{0, 2, 4})
	dst := make([]Node, 0, 3)

	oldGC := debug.SetGCPercent(-1)
	defer debug.SetGCPercent(oldGC)

	var warmErr error
	dst, warmErr = idx.SearchPlannedInto(dst[:0], vecs[0], 3, allow)
	if warmErr != nil {
		t.Fatalf("SearchPlannedInto warmup failed: %v", warmErr)
	}

	allocs := testing.AllocsPerRun(100, func() {
		var runErr error
		dst, runErr = idx.SearchPlannedInto(dst[:0], vecs[0], 3, allow)
		if runErr != nil {
			t.Fatalf("SearchPlannedInto failed: %v", runErr)
		}
	})
	if allocs != 0 {
		t.Fatalf("expected zero allocations, got %f", allocs)
	}

	if len(dst) == 0 {
		t.Fatal("expected at least one planned result")
	}

	for _, n := range dst {
		if n.ID%2 != 0 {
			t.Fatalf("unexpected disallowed result %d", n.ID)
		}
	}
}
