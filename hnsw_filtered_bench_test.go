package hnsw

import (
	"fmt"
	"math/rand/v2"
	"path/filepath"
	"strings"
	"testing"
)

const (
	filteredBenchNodes = 1024
	filteredBenchDims  = benchDims
)

type filteredPlannerCase struct {
	selectivity string
	allowCount  int
	k           int
	correlated  bool
}

func BenchmarkFilteredPlannerMatrix(b *testing.B) {
	vectors := benchFilteredVectors(filteredBenchNodes, filteredBenchDims, 0x71, 0x19)
	cases := []filteredPlannerCase{
		{selectivity: "1pct", allowCount: 16, k: 1, correlated: true},
		{selectivity: "1pct", allowCount: 16, k: 8, correlated: true},
		{selectivity: "1pct", allowCount: 16, k: 32, correlated: true},
		{selectivity: "10pct", allowCount: 128, k: 1, correlated: true},
		{selectivity: "10pct", allowCount: 128, k: 8, correlated: true},
		{selectivity: "10pct", allowCount: 128, k: 32, correlated: true},
		{selectivity: "50pct", allowCount: 512, k: 1, correlated: true},
		{selectivity: "50pct", allowCount: 512, k: 8, correlated: true},
		{selectivity: "50pct", allowCount: 512, k: 32, correlated: true},
		{selectivity: "1pct", allowCount: 16, k: 1, correlated: false},
		{selectivity: "1pct", allowCount: 16, k: 8, correlated: false},
		{selectivity: "1pct", allowCount: 16, k: 32, correlated: false},
		{selectivity: "10pct", allowCount: 128, k: 1, correlated: false},
		{selectivity: "10pct", allowCount: 128, k: 8, correlated: false},
		{selectivity: "10pct", allowCount: 128, k: 32, correlated: false},
		{selectivity: "50pct", allowCount: 512, k: 1, correlated: false},
		{selectivity: "50pct", allowCount: 512, k: 8, correlated: false},
		{selectivity: "50pct", allowCount: 512, k: 32, correlated: false},
	}

	for _, tc := range cases {
		tc := tc
		b.Run(tc.name(), func(b *testing.B) {
			path := filepath.Join(b.TempDir(), fmt.Sprintf("filtered_%s.hnsw", tc.slug()))
			idx := benchOpenIndex(b, path, len(vectors))
			defer func() {
				if err := idx.Close(); err != nil {
					b.Fatal(err)
				}
			}()

			if err := idx.BatchInsert(vectors, nil); err != nil {
				b.Fatal(err)
			}

			query := vectors[filteredBenchNodes/2]
			allow := benchFilteredAllowList(
				len(vectors),
				filteredBenchNodes/2,
				tc.allowCount,
				tc.correlated,
			)
			buf := idx.acquireSearchBuffer()
			exact, err := idx.searchExactAllowedInto(nil, query, tc.k, allow, buf)
			idx.releaseSearchBuffer(buf)
			if err != nil {
				b.Fatal(err)
			}

			exactSet := benchNodeSet(exact)
			plan := chooseSearchPlan(allow.Len(), tc.k, uint32(idx.Len()), idx.Stats().EfSearch)

			b.Run("exact", func(b *testing.B) {
				benchFilteredMode(b, tc.k, exactSet, func(dst []Node) ([]Node, error) {
					buf := idx.acquireSearchBuffer()
					defer idx.releaseSearchBuffer(buf)
					return idx.searchExactAllowedInto(dst, query, tc.k, allow, buf)
				})
			})

			b.Run("filtered", func(b *testing.B) {
				benchFilteredMode(b, tc.k, exactSet, func(dst []Node) ([]Node, error) {
					return idx.SearchAllowedInto(dst, query, tc.k, allow)
				})
			})

			b.Run("planned/"+planName(plan), func(b *testing.B) {
				benchFilteredMode(b, tc.k, exactSet, func(dst []Node) ([]Node, error) {
					return idx.SearchPlannedInto(dst, query, tc.k, allow)
				})
			})
		})
	}
}

func benchFilteredVectors(n, dims int, seed1, seed2 uint64) [][]float32 {
	r := rand.New(rand.NewPCG(seed1, seed2))
	vecs := make([][]float32, n)
	invN := 1 / float32(n)

	for i := range vecs {
		vec := make([]float32, dims)
		vec[0] = float32(i) * invN
		for j := 1; j < dims; j++ {
			vec[j] = r.Float32() * 0.01
		}
		vecs[i] = vec
	}

	return vecs
}

func benchFilteredAllowList(total, center, count int, correlated bool) AllowList {
	if count < 0 {
		count = 0
	}
	if count > total {
		count = total
	}

	bits := make([]uint64, (total+63)/64)
	if count == 0 {
		return NewAllowBitset(bits)
	}

	start := 0
	if correlated {
		start = center - count/2
		if start < 0 {
			start = 0
		}
		if start+count > total {
			start = total - count
		}
	} else if center < total/2 {
		start = total - count
	} else {
		start = 0
	}

	for i := 0; i < count; i++ {
		id := start + i
		word := id / 64
		bits[word] |= uint64(1) << (uint(id) % 64)
	}

	return NewAllowBitset(bits)
}

func benchFilteredMode(
	b *testing.B,
	k int,
	exactSet map[uint32]struct{},
	run func(dst []Node) ([]Node, error),
) {
	b.Helper()

	results := make([]Node, 0, k)
	b.ResetTimer()
	for b.Loop() {
		var err error
		results, err = run(results[:0])
		if err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()

	if len(results) == 0 {
		b.Fatal("expected filtered benchmark results")
	}

	b.ReportMetric(benchRecall(results, exactSet), "recall")
}

func benchNodeSet(nodes []Node) map[uint32]struct{} {
	set := make(map[uint32]struct{}, len(nodes))
	for _, n := range nodes {
		set[n.ID] = struct{}{}
	}
	return set
}

func benchRecall(results []Node, exactSet map[uint32]struct{}) float64 {
	if len(exactSet) == 0 {
		return 1
	}

	hits := 0
	for _, n := range results {
		if _, ok := exactSet[n.ID]; ok {
			hits++
		}
	}
	return float64(hits) / float64(len(exactSet))
}

func (tc filteredPlannerCase) name() string {
	corr := "far"
	if tc.correlated {
		corr = "near"
	}
	return fmt.Sprintf("%s/k=%d/%s", tc.selectivity, tc.k, corr)
}

func (tc filteredPlannerCase) slug() string {
	return strings.NewReplacer("/", "_", "=", "-").Replace(tc.name())
}

func planName(plan searchPlan) string {
	switch plan {
	case searchPlanExact:
		return "exact"
	case searchPlanStandard:
		return "standard"
	default:
		return "filtered"
	}
}
