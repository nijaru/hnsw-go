package hnsw

import (
	"fmt"
	"runtime/debug"
	"slices"
	"testing"
)

func TestSegmentedIndexSearchMergesTopK(t *testing.T) {
	head, headPath := mustSegmentIndex(t, t.TempDir(), "head", []float32{0, 2})
	frozen, frozenPath := mustSegmentIndex(t, t.TempDir(), "frozen", []float32{1, 3})
	defer func() {
		if err := head.Close(); err != nil {
			t.Fatalf("close head: %v", err)
		}
		removeTestFiles(headPath)
		if err := frozen.Close(); err != nil {
			t.Fatalf("close frozen: %v", err)
		}
		removeTestFiles(frozenPath)
	}()

	seg, err := NewSegmentedIndexFrom(head, frozen)
	if err != nil {
		t.Fatalf("publish failed: %v", err)
	}

	query := []float32{1.4}
	results, err := seg.SearchInto(make([]Node, 0, 2), query, 2)
	if err != nil {
		t.Fatalf("SearchInto failed: %v", err)
	}

	want := []uint32{2, 1}
	if len(results) != len(want) {
		t.Fatalf("got %d results, want %d", len(results), len(want))
	}
	for i, n := range results {
		if n.ID != want[i] {
			t.Fatalf("result %d id=%d, want %d", i, n.ID, want[i])
		}
	}
}

func TestSegmentedIndexSearchRemapsFrozenIDs(t *testing.T) {
	head, headPath := mustSegmentIndex(t, t.TempDir(), "head", []float32{0, 10})
	frozen, frozenPath := mustSegmentIndex(t, t.TempDir(), "frozen", []float32{1, 11})
	defer func() {
		if err := head.Close(); err != nil {
			t.Fatalf("close head: %v", err)
		}
		removeTestFiles(headPath)
		if err := frozen.Close(); err != nil {
			t.Fatalf("close frozen: %v", err)
		}
		removeTestFiles(frozenPath)
	}()

	seg, err := NewSegmentedIndexFrom(head, frozen)
	if err != nil {
		t.Fatalf("publish failed: %v", err)
	}

	results, err := seg.SearchInto(make([]Node, 0, 1), []float32{11}, 1)
	if err != nil {
		t.Fatalf("SearchInto failed: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("got %d results, want 1", len(results))
	}
	if results[0].ID != 3 {
		t.Fatalf("frozen local id remapped to %d, want 3", results[0].ID)
	}
}

func BenchmarkSegmentedSearchMerge(b *testing.B) {
	head, frozen, query := benchSegmentedIndices(b)
	defer func() {
		if err := head.Close(); err != nil {
			b.Fatal(err)
		}
		if err := frozen.Close(); err != nil {
			b.Fatal(err)
		}
	}()

	seg, err := NewSegmentedIndexFrom(head, frozen)
	if err != nil {
		b.Fatal(err)
	}

	results := make([]Node, 0, benchSearchK)
	if _, err := seg.SearchInto(results[:0], query, benchSearchK); err != nil {
		b.Fatal(err)
	}

	oldGC := debug.SetGCPercent(-1)
	defer debug.SetGCPercent(oldGC)

	b.StartTimer()
	for b.Loop() {
		var runErr error
		results, runErr = seg.SearchInto(results[:0], query, benchSearchK)
		if runErr != nil {
			b.Fatal(runErr)
		}
	}
	b.StopTimer()
}

func BenchmarkSegmentedPublish(b *testing.B) {
	head, frozen, _ := benchSegmentedIndices(b)
	defer func() {
		if err := head.Close(); err != nil {
			b.Fatal(err)
		}
		if err := frozen.Close(); err != nil {
			b.Fatal(err)
		}
	}()

	seg := NewSegmentedIndex()
	oldGC := debug.SetGCPercent(-1)
	defer debug.SetGCPercent(oldGC)

	b.StartTimer()
	for b.Loop() {
		if err := seg.Publish(head, frozen); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

func mustSegmentIndex(tb testing.TB, root, name string, values []float32) (*Index, string) {
	tb.Helper()

	path := fmt.Sprintf("%s/%s.hnsw", root, name)
	storage, err := NewStorage(path, IndexConfig{
		Dims:     1,
		M:        4,
		MMax0:    8,
		MaxLevel: 4,
	}, uint32(len(values)))
	if err != nil {
		tb.Fatalf("NewStorage(%s): %v", name, err)
	}

	idx := NewIndex(storage, L2)
	idx.SetEfSearch(16)
	idx.SetEfConst(16)
	for _, v := range values {
		if err := idx.Insert([]float32{v}, nil); err != nil {
			tb.Fatalf("Insert(%s): %v", name, err)
		}
	}
	return idx, path
}

func benchSegmentedIndices(tb testing.TB) (*Index, *Index, []float32) {
	tb.Helper()

	vectors := benchVectors(benchNodes, 1, 0x71, 0x72)
	half := len(vectors) / 2

	root := tb.TempDir()
	head := benchOpenSegmentIndex(tb, fmt.Sprintf("%s/head.hnsw", root), vectors[:half])
	frozen := benchOpenSegmentIndex(tb, fmt.Sprintf("%s/frozen.hnsw", root), vectors[half:])
	query := vectors[len(vectors)/2]
	return head, frozen, query
}

func benchOpenSegmentIndex(tb testing.TB, path string, vectors [][]float32) *Index {
	tb.Helper()

	storage, err := NewStorage(path, IndexConfig{
		Dims:     uint32(len(vectors[0])),
		M:        benchM,
		MMax0:    benchMMax0,
		MaxLevel: benchMaxLevel,
	}, uint32(len(vectors)))
	if err != nil {
		tb.Fatal(err)
	}

	idx := NewIndex(storage, L2)
	idx.SetEfSearch(benchEfSearch)
	idx.SetEfConst(benchEfConst)
	if err := idx.BatchInsert(vectors, nil); err != nil {
		tb.Fatal(err)
	}
	return idx
}

func exactTopK(vecs [][]float32, query []float32, baseID uint32, k int) []uint32 {
	type scored struct {
		id   uint32
		dist float32
	}

	scoredIDs := make([]scored, 0, len(vecs))
	for i, vec := range vecs {
		scoredIDs = append(scoredIDs, scored{
			id:   baseID + uint32(i),
			dist: L2(query, vec),
		})
	}

	slices.SortFunc(scoredIDs, func(a, b scored) int {
		switch {
		case a.dist < b.dist:
			return -1
		case a.dist > b.dist:
			return 1
		default:
			return 0
		}
	})

	if k > len(scoredIDs) {
		k = len(scoredIDs)
	}

	ids := make([]uint32, 0, k)
	for i := 0; i < k; i++ {
		ids = append(ids, scoredIDs[i].id)
	}
	return ids
}

func TestSegmentedIndexSearchMatchesExactMerge(t *testing.T) {
	head, headPath := mustSegmentIndex(t, t.TempDir(), "head", []float32{0, 2, 4})
	frozen, frozenPath := mustSegmentIndex(t, t.TempDir(), "frozen", []float32{1, 3, 5})
	defer func() {
		if err := head.Close(); err != nil {
			t.Fatalf("close head: %v", err)
		}
		removeTestFiles(headPath)
		if err := frozen.Close(); err != nil {
			t.Fatalf("close frozen: %v", err)
		}
		removeTestFiles(frozenPath)
	}()

	seg, err := NewSegmentedIndexFrom(head, frozen)
	if err != nil {
		t.Fatalf("publish failed: %v", err)
	}

	query := []float32{2.6}
	got, err := seg.SearchInto(make([]Node, 0, 3), query, 3)
	if err != nil {
		t.Fatalf("SearchInto failed: %v", err)
	}

	want := exactTopK([][]float32{{0}, {2}, {4}, {1}, {3}, {5}}, query, 0, 3)
	if len(got) != len(want) {
		t.Fatalf("got %d results, want %d", len(got), len(want))
	}
	for i, n := range got {
		if n.ID != want[i] {
			t.Fatalf("result %d id=%d, want %d", i, n.ID, want[i])
		}
	}
}

func TestSegmentedIndexPublishRejectsEmpty(t *testing.T) {
	seg := NewSegmentedIndex()
	if err := seg.Publish(nil); err == nil {
		t.Fatal("expected empty publish to fail")
	}
}

func TestSegmentedIndexSearchAllowed(t *testing.T) {
	head, headPath := mustSegmentIndex(t, t.TempDir(), "head", []float32{0, 2, 4})
	frozen, frozenPath := mustSegmentIndex(t, t.TempDir(), "frozen", []float32{1, 3, 5})
	defer func() {
		head.Close()
		removeTestFiles(headPath)
		frozen.Close()
		removeTestFiles(frozenPath)
	}()

	seg, err := NewSegmentedIndexFrom(head, frozen)
	if err != nil {
		t.Fatalf("publish failed: %v", err)
	}

	// global IDs: head (0, 1, 2) values (0, 2, 4)
	// global IDs: frozen (3, 4, 5) values (1, 3, 5)

	// Allow only global IDs 1 (val 2) and 4 (val 3)
	allow := NewAllowIDsSorted([]uint32{1, 4})

	query := []float32{2.5}
	got, err := seg.SearchAllowedInto(make([]Node, 0, 3), query, 3, allow)
	if err != nil {
		t.Fatalf("SearchAllowedInto failed: %v", err)
	}

	if len(got) != 2 {
		t.Fatalf("got %d results, want 2", len(got))
	}

	has1 := false
	has4 := false
	for _, n := range got {
		if n.ID == 1 {
			has1 = true
		}
		if n.ID == 4 {
			has4 = true
		}
	}
	if !has1 || !has4 {
		t.Errorf("expected IDs 1 and 4, got %v", got)
	}
}
