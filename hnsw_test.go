package hnsw

import (
	"os"
	"testing"
)

func TestHNSW(t *testing.T) {
	path := "test.hnsw"
	defer os.Remove(path)

	config := IndexConfig{
		Dims:     128,
		M:        16,
		M_max0:   32,
		MaxLevel: 16,
	}

	storage, err := NewStorage(path, config, 1000)
	if err != nil {
		t.Fatalf("failed to create storage: %v", err)
	}
	defer storage.Close()

	idx := NewIndex(storage, L2, 16, 100, 100)

	// Insert 100 random vectors
	vectors := make([][]float32, 100)
	for i := range vectors {
		vectors[i] = make([]float32, 128)
		for j := range vectors[i] {
			vectors[i][j] = float32(i + j)
		}
		if err := idx.Insert(vectors[i]); err != nil {
			t.Fatalf("failed to insert vector %d: %v", i, err)
		}
	}

	// Search for one of the vectors
	query := vectors[50]
	results := idx.Search(query, 5)

	if len(results) == 0 {
		t.Fatalf("expected results, got none")
	}

	// Closest result should be the vector itself (dist 0)
	if results[0].ID != 50 {
		t.Errorf("expected closest result to be ID 50, got %d", results[0].ID)
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
		M_max0:   8,
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
