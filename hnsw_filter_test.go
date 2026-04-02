package hnsw

import (
	"bytes"
	"testing"
)

func TestSearchFiltered(t *testing.T) {
	path := "test_search_filtered.hnsw"
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
	}

	metas := [][]byte{
		[]byte("group:a"),
		[]byte("group:b"),
		[]byte("group:a"),
		[]byte("group:b"),
		[]byte("group:a"),
		[]byte("group:b"),
	}

	for i := range vecs {
		if err := idx.Insert(vecs[i], metas[i]); err != nil {
			t.Fatalf("Insert %d failed: %v", i, err)
		}
	}

	dst := make([]Node, 0, 3)
	prefix := []byte("group:b")
	filter := func(n Node) bool {
		return bytes.HasPrefix(n.Metadata, prefix)
	}
	allocs := testing.AllocsPerRun(100, func() {
		var err error
		dst, err = idx.SearchFilteredInto(dst[:0], vecs[0], 3, filter)
		if err != nil {
			t.Fatalf("SearchFilteredInto failed: %v", err)
		}
	})
	if allocs != 0 {
		t.Fatalf("expected zero allocations, got %f", allocs)
	}

	if len(dst) == 0 {
		t.Fatal("expected filtered results")
	}

	for _, n := range dst {
		if !bytes.HasPrefix(n.Metadata, prefix) {
			t.Fatalf("unexpected metadata %q in filtered results", string(n.Metadata))
		}
	}
}

func TestCopyMetadata(t *testing.T) {
	path := "test_copy_metadata.hnsw"
	defer removeTestFiles(path)

	config := IndexConfig{
		Dims:     4,
		M:        4,
		MMax0:    8,
		MaxLevel: 4,
	}

	storage, err := NewStorage(path, config, 4)
	if err != nil {
		t.Fatalf("failed to create storage: %v", err)
	}
	defer storage.Close()

	idx := NewIndex(storage, L2, 4, 4)

	if err := idx.Insert([]float32{1, 0, 0, 0}, []byte("payload")); err != nil {
		t.Fatalf("Insert failed: %v", err)
	}

	copied := idx.CopyMetadata(0)
	if string(copied) != "payload" {
		t.Fatalf("expected payload, got %q", string(copied))
	}

	copied[0] = 'P'

	copiedAgain := idx.CopyMetadata(0)
	if string(copiedAgain) != "payload" {
		t.Fatalf("expected payload after copy mutation, got %q", string(copiedAgain))
	}
}
