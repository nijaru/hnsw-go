package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sort"
	"time"

	"github.com/omendb/hnsw-go"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: bench <sift10k_test.bin>")
		os.Exit(1)
	}
	data, err := os.ReadFile(os.Args[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	vectors, queries, gt := parseSIFT(data)
	fmt.Printf("SIFT 10k: %d vectors, %d queries, %d dims, topK=%d\n\n",
		len(vectors), len(queries), len(vectors[0]), len(gt[0]))

	benchmarkBuild(vectors)
	benchmarkSearch(vectors, queries, gt)
}

func parseSIFT(data []byte) (vectors, queries [][]float32, gt [][]int32) {
	r := &binReader{data: data}
	nVec := r.uint32()
	nQuery := r.uint32()
	dims := r.uint32()
	topK := r.uint32()

	vectors = make([][]float32, nVec)
	for i := range vectors {
		vectors[i] = r.float32s(dims)
	}
	queries = make([][]float32, nQuery)
	for i := range queries {
		queries[i] = r.float32s(dims)
	}
	gt = make([][]int32, nQuery)
	for i := range gt {
		gt[i] = r.int32s(topK)
	}
	return
}

func benchmarkBuild(vectors [][]float32) {
	const (
		M       = 16
		MMax0   = 32
		EfConst = 200
	)

	dims := uint32(len(vectors[0]))

	var best, worst time.Duration
	var total time.Duration
	const runs = 5

	for run := 0; run < runs; run++ {
		path := fmt.Sprintf("%s/bench_build_%d.hnsw", os.TempDir(), os.Getpid())
		cfg := hnsw.IndexConfig{Dims: dims, M: M, MMax0: MMax0, MaxLevel: 16}
		storage, err := hnsw.NewStorage(path, cfg, uint32(len(vectors)))
		if err != nil {
			panic(err)
		}
		idx := hnsw.NewIndex(storage, hnsw.L2, 200, EfConst)

		start := time.Now()
		for _, vec := range vectors {
			if err := idx.Insert(vec, nil); err != nil {
				panic(err)
			}
		}
		elapsed := time.Since(start)

		storage.Close()
		os.Remove(path)
		os.Remove(path + ".vec")

		total += elapsed
		if run == 0 || elapsed < best {
			best = elapsed
		}
		if run == 0 || elapsed > worst {
			worst = elapsed
		}
	}

	avg := total / runs
	fmt.Println("=== Build ===")
	fmt.Printf(
		"  Avg:    %s (%.0f vecs/sec)\n",
		avg.Round(time.Microsecond),
		float64(len(vectors))/avg.Seconds(),
	)
	fmt.Printf("  Best:   %s\n", best.Round(time.Microsecond))
	fmt.Printf("  Worst:  %s\n", worst.Round(time.Microsecond))
	fmt.Println()
}

func benchmarkSearch(vectors, queries [][]float32, gt [][]int32) {
	const (
		M        = 16
		MMax0    = 32
		EfSearch = 200
		K        = 10
	)

	dims := uint32(len(vectors[0]))
	path := fmt.Sprintf("%s/bench_search_%d.hnsw", os.TempDir(), os.Getpid())
	defer os.Remove(path)
	defer os.Remove(path + ".vec")

	cfg := hnsw.IndexConfig{Dims: dims, M: M, MMax0: MMax0, MaxLevel: 16}
	storage, err := hnsw.NewStorage(path, cfg, uint32(len(vectors)))
	if err != nil {
		panic(err)
	}
	defer storage.Close()

	idx := hnsw.NewIndex(storage, hnsw.L2, EfSearch, 200)
	for _, vec := range vectors {
		if err := idx.Insert(vec, nil); err != nil {
			panic(err)
		}
	}

	latencies := make([]time.Duration, len(queries))
	totalRecall := 0.0

	for qi, query := range queries {
		start := time.Now()
		results, err := idx.Search(query, K)
		latencies[qi] = time.Since(start)
		if err != nil {
			panic(err)
		}

		gtSet := make(map[uint32]struct{}, K)
		for i := 0; i < K && i < len(gt[qi]); i++ {
			gtSet[uint32(gt[qi][i])] = struct{}{}
		}
		hits := 0
		for i := 0; i < K && i < len(results); i++ {
			if _, ok := gtSet[results[i].ID]; ok {
				hits++
			}
		}
		totalRecall += float64(hits) / float64(K)
	}

	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
	avgRecall := totalRecall / float64(len(queries))
	var totalTime time.Duration
	for _, l := range latencies {
		totalTime += l
	}
	qps := float64(len(queries)) / totalTime.Seconds()

	p := func(pct float64) time.Duration {
		idx := int(float64(len(latencies)-1) * pct)
		return latencies[idx]
	}

	fmt.Println("=== Search ===")
	fmt.Printf("  QPS:      %.0f\n", qps)
	fmt.Printf("  Recall@%d: %.4f\n", K, avgRecall)
	fmt.Printf(
		"  Mean:     %s\n",
		(totalTime / time.Duration(len(queries))).Round(time.Microsecond),
	)
	fmt.Printf("  p50:      %s\n", p(0.50).Round(time.Microsecond))
	fmt.Printf("  p90:      %s\n", p(0.90).Round(time.Microsecond))
	fmt.Printf("  p99:      %s\n", p(0.99).Round(time.Microsecond))
	fmt.Printf("  p99.9:    %s\n", p(0.999).Round(time.Microsecond))
	fmt.Printf("  Worst:    %s\n", latencies[len(latencies)-1].Round(time.Microsecond))
}

type binReader struct {
	data   []byte
	offset int
}

func (r *binReader) uint32() uint32 {
	v := binary.LittleEndian.Uint32(r.data[r.offset:])
	r.offset += 4
	return v
}

func (r *binReader) float32s(n uint32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(r.data[r.offset:]))
		r.offset += 4
	}
	return out
}

func (r *binReader) int32s(n uint32) []int32 {
	out := make([]int32, n)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint32(r.data[r.offset:]))
		r.offset += 4
	}
	return out
}
