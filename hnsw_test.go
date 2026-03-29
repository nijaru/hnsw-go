package hnsw

import (
	"math/rand/v2"
	"os"
	"testing"
)

func TestHNSW(t *testing.T) {
	path := "test.hnsw"
	defer os.Remove(path)

	config := IndexConfig{
		Dims:     128,
		M:        16,
		MMax0:    32,
		MaxLevel: 16,
	}

	storage, err := NewStorage(path, config, 1000)
	if err != nil {
		t.Fatalf("failed to create storage: %v", err)
	}
	defer storage.Close()

	idx := NewIndex(storage, L2, 100, 100)

	vectors := make([][]float32, 100)
	for i := range vectors {
		vectors[i] = make([]float32, 128)
		for j := range vectors[i] {
			vectors[i][j] = rand.Float32()
		}
		if err := idx.Insert(vectors[i]); err != nil {
			t.Fatalf("failed to insert vector %d: %v", i, err)
		}
	}

	results, err := idx.Search(vectors[50], 5)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results) == 0 {
		t.Fatalf("expected results, got none")
	}

	if results[0].ID != 50 {
		t.Errorf(
			"expected closest result to be ID 50, got %d (dist=%f)",
			results[0].ID,
			results[0].Distance,
		)
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
		MMax0:    8,
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
