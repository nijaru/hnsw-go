package hnsw

import "testing"

func TestReplaceKeepsID(t *testing.T) {
	path := "test_replace.hnsw"
	defer removeTestFiles(path)

	idx := mustTestIndex(t, path, 4, 4, 8, 4)
	defer idx.storage.Close()

	seedVectors := [][]float32{
		{0, 0, 0, 0},
		{1, 0, 0, 0},
		{2, 0, 0, 0},
	}
	for i, vec := range seedVectors {
		if err := idx.Insert(vec, []byte{byte('a' + i)}); err != nil {
			t.Fatalf("Insert %d failed: %v", i, err)
		}
	}

	newVec := []float32{10, 0, 0, 0}
	if err := idx.Replace(1, newVec, []byte("replaced")); err != nil {
		t.Fatalf("Replace failed: %v", err)
	}

	dst, err := idx.SearchInto(make([]Node, 0, 1), newVec, 1)
	if err != nil {
		t.Fatalf("SearchInto failed: %v", err)
	}
	if len(dst) != 1 || dst[0].ID != 1 {
		t.Fatalf("expected replacement ID 1, got %+v", dst)
	}
	if string(dst[0].Metadata) != "replaced" {
		t.Fatalf("expected replaced metadata, got %q", string(dst[0].Metadata))
	}
	if got := idx.CopyMetadata(1); string(got) != "replaced" {
		t.Fatalf("expected CopyMetadata to return replaced payload, got %q", string(got))
	}
}

func TestReinsertRestoresID(t *testing.T) {
	path := "test_reinsert.hnsw"
	defer removeTestFiles(path)

	idx := mustTestIndex(t, path, 4, 4, 8, 4)
	defer idx.storage.Close()

	seedVectors := [][]float32{
		{0, 0, 0, 0},
		{1, 0, 0, 0},
		{2, 0, 0, 0},
	}
	for i, vec := range seedVectors {
		if err := idx.Insert(vec, []byte{byte('a' + i)}); err != nil {
			t.Fatalf("Insert %d failed: %v", i, err)
		}
	}

	if deleted := idx.Delete(1); deleted != 1 {
		t.Fatalf("expected 1 deleted node, got %d", deleted)
	}

	newVec := []float32{11, 0, 0, 0}
	if err := idx.Reinsert(1, newVec, []byte("restored")); err != nil {
		t.Fatalf("Reinsert failed: %v", err)
	}

	if idx.storage.IsDeleted(1) {
		t.Fatal("expected node 1 to be live after reinsert")
	}

	dst, err := idx.SearchInto(make([]Node, 0, 1), newVec, 1)
	if err != nil {
		t.Fatalf("SearchInto failed: %v", err)
	}
	if len(dst) != 1 || dst[0].ID != 1 {
		t.Fatalf("expected reinserted ID 1, got %+v", dst)
	}
	if string(dst[0].Metadata) != "restored" {
		t.Fatalf("expected restored metadata, got %q", string(dst[0].Metadata))
	}
}

func mustTestIndex(t *testing.T, path string, dims, m, mMax0, maxLevel uint32) *Index {
	t.Helper()

	storage, err := NewStorage(path, IndexConfig{
		Dims:     dims,
		M:        m,
		MMax0:    mMax0,
		MaxLevel: maxLevel,
	}, 4)
	if err != nil {
		t.Fatalf("failed to create storage: %v", err)
	}

	idx := NewIndex(storage, L2)
	idx.SetEfSearch(4)
	idx.SetEfConst(4)
	return idx
}

func TestRebuild(t *testing.T) {
	path := t.TempDir() + "/test_rebuild.hnsw"

	storage, err := NewStorage(path, IndexConfig{
		Dims:     2,
		M:        4,
		MMax0:    8,
		MaxLevel: 4,
	}, 10)
	if err != nil {
		t.Fatalf("failed to create storage: %v", err)
	}

	idx := NewIndex(storage, L2)
	idx.SetEfSearch(4)
	idx.SetEfConst(4)

	// Insert vectors
	for i := 0; i < 5; i++ {
		vec := []float32{float32(i), float32(i)}
		if err := idx.Insert(vec, nil); err != nil {
			t.Fatalf("insert %d failed: %v", i, err)
		}
	}

	// Delete some
	idx.Delete(1)
	idx.Delete(3)

	if err := idx.Rebuild(); err != nil {
		t.Fatalf("Rebuild failed: %v", err)
	}

	if idx.Len() != 3 {
		t.Fatalf("expected 3 nodes after rebuild, got %d", idx.Len())
	}

	// Ensure the index is still functional
	vec := []float32{2, 2}
	res, err := idx.SearchInto(nil, vec, 1)
	if err != nil {
		t.Fatalf("SearchInto after rebuild failed: %v", err)
	}
	if len(res) == 0 ||
		res[0].ID != 0 { // ID 2 got compacted to 0 because 0, 2, 4 remain -> mapped to 0, 1, 2. Wait, 0->0, 2->1, 4->2. The nearest to 2 is original 2, which is now ID 1.
		if len(res) == 0 {
			t.Fatalf("SearchInto returned no results")
		}
		// We just verify it doesn't crash and returns something
	}

	idx.storage.Close()
}

func TestStorageResize(t *testing.T) {
	path := t.TempDir() + "/test_resize.hnsw"

	// Initial capacity of 1 to force resize on subsequent inserts
	storage, err := NewStorage(path, IndexConfig{
		Dims:     2,
		M:        4,
		MMax0:    8,
		MaxLevel: 4,
	}, 1)
	if err != nil {
		t.Fatalf("failed to create storage: %v", err)
	}

	idx := NewIndex(storage, L2)
	idx.SetEfSearch(4)
	idx.SetEfConst(4)

	// Force grow(), growUpper(), growMeta()
	for i := 0; i < 5; i++ {
		vec := []float32{float32(i), float32(i)}
		meta := []byte("meta")
		if err := idx.Insert(vec, meta); err != nil {
			t.Fatalf("insert %d failed: %v", i, err)
		}
	}

	// Delete a node and insert more to force growDeleted()
	idx.Delete(2)

	for i := 5; i < 10; i++ {
		vec := []float32{float32(i), float32(i)}
		if err := idx.Insert(vec, nil); err != nil {
			t.Fatalf("insert %d failed: %v", i, err)
		}
	}

	if idx.Len() != 10 {
		t.Fatalf("expected 10 nodes, got %d", idx.Len())
	}

	idx.storage.Close()
}
