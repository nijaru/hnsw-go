package hnsw

import (
	"fmt"
	"math/rand/v2"
	"path/filepath"
	"testing"
)

const (
	benchDims      = 128
	benchNodes     = 10_000
	benchM         = 16
	benchMMax0     = 32
	benchMaxLevel  = 16
	benchEfSearch  = 200
	benchEfConst   = 200
	benchSearchK   = 10
	benchDeleteCnt = benchNodes / 4
)

func BenchmarkHNSWSearch(b *testing.B) {
	vectors := benchVectors(benchNodes, benchDims, 0x01, 0x02)
	path := filepath.Join(b.TempDir(), "search.hnsw")

	b.StopTimer()
	idx := benchOpenIndex(b, path, len(vectors))
	defer func() {
		if err := idx.Close(); err != nil {
			b.Fatal(err)
		}
	}()

	if err := idx.BatchInsert(vectors, nil); err != nil {
		b.Fatal(err)
	}

	query := vectors[len(vectors)/2]
	if _, err := idx.Search(query, benchSearchK); err != nil {
		b.Fatal(err)
	}

	b.StartTimer()
	for b.Loop() {
		if _, err := idx.Search(query, benchSearchK); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

func BenchmarkHNSWBatchInsert(b *testing.B) {
	vectors := benchVectors(benchNodes, benchDims, 0x11, 0x22)
	root := b.TempDir()

	b.StopTimer()
	for i := 0; i < b.N; i++ {
		path := filepath.Join(root, fmt.Sprintf("batch_insert_%d.hnsw", i))
		func() {
			idx := benchOpenIndex(b, path, len(vectors))
			defer removeTestFiles(path)
			defer func() {
				if err := idx.Close(); err != nil {
					b.Fatal(err)
				}
			}()

			b.StartTimer()
			if err := idx.BatchInsert(vectors, nil); err != nil {
				b.Fatal(err)
			}
			b.StopTimer()
		}()
	}
}

func BenchmarkHNSWDelete(b *testing.B) {
	vectors := benchVectors(benchNodes, benchDims, 0x33, 0x44)
	root := b.TempDir()
	deleteID := uint32(len(vectors) / 2)

	b.StopTimer()
	for i := 0; i < b.N; i++ {
		path := filepath.Join(root, fmt.Sprintf("delete_%d.hnsw", i))
		func() {
			idx := benchOpenIndex(b, path, len(vectors))
			defer removeTestFiles(path)
			defer func() {
				if err := idx.Close(); err != nil {
					b.Fatal(err)
				}
			}()

			if err := idx.BatchInsert(vectors, nil); err != nil {
				b.Fatal(err)
			}

			b.StartTimer()
			if n := idx.Delete(deleteID); n != 1 {
				b.Fatalf("expected 1 deleted node, got %d", n)
			}
			b.StopTimer()
		}()
	}
}

func BenchmarkHNSWVacuum(b *testing.B) {
	vectors := benchVectors(benchNodes, benchDims, 0x55, 0x66)
	root := b.TempDir()
	deleteIDs := benchDeleteIDs(len(vectors), benchDeleteCnt)

	b.StopTimer()
	for i := 0; i < b.N; i++ {
		path := filepath.Join(root, fmt.Sprintf("vacuum_%d.hnsw", i))
		func() {
			idx := benchOpenIndex(b, path, len(vectors))
			defer removeTestFiles(path)
			defer func() {
				if err := idx.Close(); err != nil {
					b.Fatal(err)
				}
			}()

			if err := idx.BatchInsert(vectors, nil); err != nil {
				b.Fatal(err)
			}
			if deleted, err := idx.BulkDelete(deleteIDs); err != nil {
				b.Fatal(err)
			} else if deleted != len(deleteIDs) {
				b.Fatalf("expected %d deleted nodes, got %d", len(deleteIDs), deleted)
			}

			b.StartTimer()
			if err := idx.Vacuum(); err != nil {
				b.Fatal(err)
			}
			b.StopTimer()
		}()
	}
}

func benchOpenIndex(tb testing.TB, path string, capacity int) *Index {
	tb.Helper()

	storage, err := NewStorage(path, IndexConfig{
		Dims:     benchDims,
		M:        benchM,
		MMax0:    benchMMax0,
		MaxLevel: benchMaxLevel,
	}, uint32(capacity))
	if err != nil {
		tb.Fatal(err)
	}

	return NewIndex(storage, L2, benchEfSearch, benchEfConst)
}

func benchVectors(n, dims int, seed1, seed2 uint64) [][]float32 {
	r := rand.New(rand.NewPCG(seed1, seed2))
	vecs := make([][]float32, n)
	for i := range vecs {
		vecs[i] = make([]float32, dims)
		for j := range vecs[i] {
			vecs[i][j] = r.Float32()
		}
	}
	return vecs
}

func benchDeleteIDs(total, count int) []uint32 {
	if count > total {
		count = total
	}

	ids := make([]uint32, count)
	for i := range ids {
		ids[i] = uint32(i)
	}
	return ids
}
