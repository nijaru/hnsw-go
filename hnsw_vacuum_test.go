package hnsw

import (
	"os"
	"testing"
)

func TestVacuum(t *testing.T) {
	path := "test_vacuum.hnsw"
	defer removeTestFiles(path)

	config := IndexConfig{
		Dims:     128,
		M:        16,
		MMax0:    32,
		MaxLevel: 16,
	}

	storage, err := NewStorage(path, config, 100)
	if err != nil {
		t.Fatalf("failed to create storage: %v", err)
	}
	defer storage.Close()

	idx := NewIndex(storage, L2, 50, 50)

	// 1. Insert 100 vectors
	vectors := make([][]float32, 100)
	for i := range vectors {
		vectors[i] = make([]float32, 128)
		vectors[i][0] = float32(i)
	}
	if err := idx.BatchInsert(vectors, nil); err != nil {
		t.Fatalf("BatchInsert failed: %v", err)
	}

	// 2. Delete 50 vectors
	for i := 0; i < 50; i++ {
		if n := idx.Delete(uint32(i)); n == 0 {
			t.Fatalf("Delete %d returned 0", i)
		}
	}

	if idx.Len() != 100 {
		t.Errorf("expected len 100 before vacuum, got %d", idx.Len())
	}

	// Record file sizes
	info, _ := os.Stat(path)
	oldSize := info.Size()

	// 3. Vacuum
	if err := idx.Vacuum(); err != nil {
		t.Fatalf("Vacuum failed: %v", err)
	}

	// 4. Verify
	info2, _ := os.Stat(path)
	newSize := info2.Size()

	if newSize >= oldSize {
		t.Errorf("expected size reduction after vacuum, got %d -> %d", oldSize, newSize)
	}

	if idx.Len() != 50 {
		t.Errorf("expected len 50 after vacuum, got %d", idx.Len())
	}

	stats := idx.Stats()
	if stats.NodeCount != 50 {
		t.Errorf("expected stats.NodeCount 50, got %d", stats.NodeCount)
	}

	// Verify we can still search and find the remaining vectors
	results, err := idx.Search(vectors[75], 5)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("no results for vector 75")
	}
	// Note: IDs have changed after Vacuum! ID 75 became something else (likely 25).
	// We check the distance instead.
	if results[0].Distance > 0.0001 {
		t.Errorf("expected distance ~0 for identical vector, got %f", results[0].Distance)
	}

	// Verify that deleted vectors are truly gone and IDs 0..49 don't return them
	// Actually, the new IDs 0..49 now point to old IDs 50..99.

	// Check that we don't find old ID 0 (which was vectors[0])
	results, err = idx.Search(vectors[0], 1)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(results) > 0 && results[0].Distance < 0.1 {
		t.Errorf("found deleted vector 0 in search results after vacuum")
	}

	t.Logf("Vacuum successful: 100 -> 50 nodes. Old size: %d, New size: %d", oldSize, newSize)
}
