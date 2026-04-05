package hnsw

import (
	"testing"
)

func TestMetadata(t *testing.T) {
	path := "test_meta.hnsw"
	defer removeTestFiles(path)

	config := IndexConfig{
		Dims:     4,
		M:        4,
		MaxLevel: 4,
	}

	storage, err := NewStorage(path, config, 10)
	if err != nil {
		t.Fatalf("failed to create storage: %v", err)
	}
	defer storage.Close()

	idx := NewIndex(storage, L2)
	idx.SetEfSearch(10)
	idx.SetEfConst(10)

	// 1. Insert with metadata
	vec1 := []float32{1, 0, 0, 0}
	meta1 := []byte("payload 1")
	if err := idx.Insert(vec1, meta1); err != nil {
		t.Fatalf("Insert 1 failed: %v", err)
	}

	vec2 := []float32{0, 1, 0, 0}
	meta2 := []byte("payload 2 - slightly longer")
	if err := idx.Insert(vec2, meta2); err != nil {
		t.Fatalf("Insert 2 failed: %v", err)
	}

	// 2. Search and verify metadata
	results, err := idx.Search(vec1, 5)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) < 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}

	found1 := false
	found2 := false
	for _, res := range results {
		if res.ID == 0 {
			if string(res.Metadata) != string(meta1) {
				t.Errorf("expected meta1 %q, got %q", string(meta1), string(res.Metadata))
			}
			found1 = true
		} else if res.ID == 1 {
			if string(res.Metadata) != string(meta2) {
				t.Errorf("expected meta2 %q, got %q", string(meta2), string(res.Metadata))
			}
			found2 = true
		}
	}

	if !found1 || !found2 {
		t.Errorf("did not find both nodes in results: found1=%v, found2=%v", found1, found2)
	}

	// 3. BatchInsert with metadata
	vecs := [][]float32{
		{0, 0, 1, 0},
		{0, 0, 0, 1},
	}
	metas := [][]byte{
		[]byte("meta 3"),
		[]byte("meta 4"),
	}
	if err := idx.BatchInsert(vecs, metas); err != nil {
		t.Fatalf("BatchInsert failed: %v", err)
	}

	results, err = idx.Search(vecs[0], 1)
	if err != nil {
		t.Fatalf("Search 3 failed: %v", err)
	}
	if string(results[0].Metadata) != "meta 3" {
		t.Errorf("expected meta 3, got %q", string(results[0].Metadata))
	}

	// 4. Persistence check
	storage.Close()

	storage2, err := NewStorage(path, config, 0)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer storage2.Close()

	idx2 := NewIndex(storage2, L2)
	idx2.SetEfSearch(10)
	idx2.SetEfConst(10)
	results, err = idx2.Search(vec2, 1)
	if err != nil {
		t.Fatalf("search after reopen: %v", err)
	}
	if string(results[0].Metadata) != string(meta2) {
		t.Errorf(
			"persistence: expected meta2 %q, got %q",
			string(meta2),
			string(results[0].Metadata),
		)
	}
}

func TestVacuumWithMetadata(t *testing.T) {
	path := "test_vacuum_meta.hnsw"
	defer removeTestFiles(path)

	config := IndexConfig{
		Dims:     4,
		M:        4,
		MaxLevel: 4,
	}

	storage, err := NewStorage(path, config, 10)
	if err != nil {
		t.Fatalf("failed to create storage: %v", err)
	}
	defer storage.Close()

	idx := NewIndex(storage, L2)
	idx.SetEfSearch(10)
	idx.SetEfConst(10)

	// Insert 3 nodes, delete one
	idx.Insert([]float32{1, 0, 0, 0}, []byte("meta 0"))
	idx.Insert([]float32{0, 1, 0, 0}, []byte("meta 1"))
	idx.Insert([]float32{0, 0, 1, 0}, []byte("meta 2"))

	if n := idx.Delete(1); n == 0 {
		t.Fatalf("Delete returned 0")
	}

	if err := idx.Vacuum(); err != nil {
		t.Fatalf("Vacuum failed: %v", err)
	}

	if idx.Len() != 2 {
		t.Errorf("expected len 2, got %d", idx.Len())
	}

	// Verify meta 0 and meta 2 are still there
	res, _ := idx.Search([]float32{1, 0, 0, 0}, 1)
	if string(res[0].Metadata) != "meta 0" {
		t.Errorf("expected meta 0, got %q", string(res[0].Metadata))
	}

	res, _ = idx.Search([]float32{0, 0, 1, 0}, 1)
	if string(res[0].Metadata) != "meta 2" {
		t.Errorf("expected meta 2, got %q", string(res[0].Metadata))
	}
}
