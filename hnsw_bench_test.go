package hnsw

import (
	"math/rand/v2"
	"os"
	"testing"
)

func BenchmarkHNSWBuild(b *testing.B) {
	path := "bench_build.hnsw"
	defer os.Remove(path)

	config := IndexConfig{
		Dims:     128,
		M:        16,
		M_max0:   32,
		MaxLevel: 16,
	}

	storage, err := NewStorage(path, config, uint32(b.N))
	if err != nil {
		b.Fatalf("failed to create storage: %v", err)
	}
	defer storage.Close()

	idx := NewIndex(storage, L2, 16, 100, 100)

	vec := make([]float32, 128)
	for i := range vec {
		vec[i] = rand.Float32()
	}

	for b.Loop() {
		if err := idx.Insert(vec); err != nil {
			b.Fatalf("failed to insert: %v", err)
		}
	}
}

func BenchmarkHNSWSearch(b *testing.B) {
	path := "bench_search.hnsw"
	defer os.Remove(path)

	config := IndexConfig{
		Dims:     128,
		M:        16,
		M_max0:   32,
		MaxLevel: 16,
	}

	numNodes := 10000
	storage, err := NewStorage(path, config, uint32(numNodes))
	if err != nil {
		b.Fatalf("failed to create storage: %v", err)
	}
	defer storage.Close()

	idx := NewIndex(storage, L2, 16, 100, 100)

	vec := make([]float32, 128)
	for i := 0; i < numNodes; i++ {
		for j := range vec {
			vec[j] = rand.Float32()
		}
		idx.Insert(vec)
	}

	query := make([]float32, 128)
	for i := range query {
		query[i] = rand.Float32()
	}

	for b.Loop() {
		idx.Search(query, 10)
	}
}
