package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"time"

	hnsw "github.com/omendb/hnsw-go"
)

const (
	profileDims          = 128
	profileNodes         = 100_000
	profileM             = 16
	profileMMax0         = 32
	profileMaxLevel      = 16
	profileEfSearch      = 200
	profileEfConst       = 200
	profileSearchK       = 10
	profileSearchQueries = 128
	profileFilteredNodes = 1_024
	profileFilteredK     = 8
	profileDeleteCount   = profileNodes / 4
)

type workloadDefinition struct {
	name    string
	prepare func(*profileContext) error
	run     func(*profileContext) error
}

type profileContext struct {
	opts    options
	state   any
	cleanup func() error
}

type searchState struct {
	source  string
	idx     *hnsw.Index
	vectors [][]float32
	queries [][]float32
	gt      []map[uint32]struct{}
	results []hnsw.Node
}

type buildState struct {
	idx     *hnsw.Index
	vectors [][]float32
}

type deleteState struct {
	idx       *hnsw.Index
	vectors   [][]float32
	deleteIDs []uint32
}

type vacuumState struct {
	idx       *hnsw.Index
	vectors   [][]float32
	deleteIDs []uint32
}

type filteredState struct {
	idx     *hnsw.Index
	vectors [][]float32
	query   []float32
	allow   hnsw.AllowList
	exact   map[uint32]struct{}
	results []hnsw.Node
}

type plannerCase struct {
	selectivity string
	allowCount  int
	k           int
	correlated  bool
	allow       hnsw.AllowList
	exact       map[uint32]struct{}
	expected    string
	results     []hnsw.Node
}

type plannerState struct {
	idx     *hnsw.Index
	vectors [][]float32
	query   []float32
	cases   []plannerCase
}

var workloadOrder = []string{
	"search",
	"filtered",
	"build",
	"delete",
	"vacuum",
	"planner",
}

var workloadCatalog = map[string]workloadDefinition{
	"search": {
		name:    "search",
		prepare: prepareSearch,
		run:     runSearch,
	},
	"filtered": {
		name:    "filtered",
		prepare: prepareFiltered,
		run:     runFiltered,
	},
	"build": {
		name:    "build",
		prepare: prepareBuild,
		run:     runBuild,
	},
	"delete": {
		name:    "delete",
		prepare: prepareDelete,
		run:     runDelete,
	},
	"vacuum": {
		name:    "vacuum",
		prepare: prepareVacuum,
		run:     runVacuum,
	},
	"planner": {
		name:    "planner",
		prepare: preparePlanner,
		run:     runPlanner,
	},
}

type searchPlanName string

const (
	searchPlanExact    searchPlanName = "exact"
	searchPlanStandard searchPlanName = "standard"
	searchPlanFiltered searchPlanName = "filtered"
)

type plannerSpec struct {
	selectivity string
	allowCount  int
	k           int
	correlated  bool
}

var plannerSpecs = []plannerSpec{
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

func profileIndexConfig(m int) hnsw.IndexConfig {
	return hnsw.IndexConfig{
		Dims:     profileDims,
		M:        uint32(m),
		MMax0:    uint32(m * 2),
		MaxLevel: profileMaxLevel,
	}
}

func newProfileIndex(capacity, efSearch, m, efConst int) (*hnsw.Index, func() error, error) {
	if capacity < 1 {
		capacity = 1
	}

	dir, err := os.MkdirTemp("", "hnsw-profile-*")
	if err != nil {
		return nil, nil, err
	}

	path := filepath.Join(dir, "profile.hnsw")
	storage, err := hnsw.NewStorage(path, profileIndexConfig(m), uint32(capacity))
	if err != nil {
		_ = os.RemoveAll(dir)
		return nil, nil, err
	}

	idx := hnsw.NewIndex(storage, hnsw.L2)
	idx.SetEfSearch(efSearch)
	idx.SetEfConst(efConst)
	cleanup := func() error {
		var cleanupErr error
		if err := idx.Close(); err != nil {
			cleanupErr = err
		}
		if err := os.RemoveAll(dir); err != nil && cleanupErr == nil {
			cleanupErr = err
		}
		return cleanupErr
	}

	return idx, cleanup, nil
}

func prepareSearch(ctx *profileContext) error {
	st := &searchState{}
	if ctx.opts.siftPath != "" {
		data, err := os.ReadFile(ctx.opts.siftPath)
		if err != nil {
			return err
		}
		vectors, queries, gt := parseSIFT(data)
		st.source = "sift"
		st.vectors = vectors
		st.queries = queries
		st.gt = groundTruthSets(gt, profileSearchK)
	} else {
		st.source = "synthetic"
		st.vectors = profileVectors(profileNodes, profileDims, 0x01, 0x02)
		st.queries = profileVectors(profileSearchQueries, profileDims, 0x03, 0x04)
		st.gt = exactSetsForQueries(st.vectors, st.queries, profileSearchK)
	}

	idx, cleanup, err := newProfileIndex(
		len(st.vectors),
		ctx.opts.efSearch,
		ctx.opts.m,
		ctx.opts.efConst,
	)
	if err != nil {
		return err
	}
	st.idx = idx
	ctx.cleanup = cleanup

	if err := st.idx.BatchInsert(st.vectors, nil); err != nil {
		return err
	}
	if len(st.queries) == 0 {
		return fmt.Errorf("search workload has no queries")
	}
	st.results = make([]hnsw.Node, 0, profileSearchK)
	if _, err := st.idx.SearchInto(st.results[:0], st.queries[0], profileSearchK); err != nil {
		return err
	}

	ctx.state = st
	return nil
}

func runSearch(ctx *profileContext) error {
	st := ctx.state.(*searchState)
	repeats := max(ctx.opts.repeats, 1)
	totalQueries := len(st.queries) * repeats
	latencies := make([]time.Duration, 0, totalQueries)
	results := st.results
	totalHits := 0
	totalPossible := 0

	start := time.Now()
	for r := 0; r < repeats; r++ {
		for qi, query := range st.queries {
			iterStart := time.Now()
			var err error
			results, err = st.idx.SearchInto(results[:0], query, profileSearchK)
			if err != nil {
				return err
			}
			latencies = append(latencies, time.Since(iterStart))
			if qi < len(st.gt) {
				totalHits += profileRecallHits(results, st.gt[qi])
				totalPossible += min(profileSearchK, len(st.gt[qi]))
			}
		}
	}
	elapsed := time.Since(start)

	recall := 1.0
	if totalPossible > 0 {
		recall = float64(totalHits) / float64(totalPossible)
	}

	fmt.Printf(
		"source=%s vectors=%d queries=%d repeats=%d\n",
		st.source,
		len(st.vectors),
		len(st.queries),
		repeats,
	)
	fmt.Printf("Recall@%d: %.4f\n", profileSearchK, recall)
	fmt.Printf("QPS: %.0f\n", float64(totalQueries)/elapsed.Seconds())
	printLatencySummary(latencies)
	return nil
}

func prepareFiltered(ctx *profileContext) error {
	st := &filteredState{}
	st.vectors = profileFilteredVectors(profileFilteredNodes, profileDims, 0x71, 0x19)
	st.query = st.vectors[profileFilteredNodes/2]
	st.allow = profileAllowList(len(st.vectors), len(st.vectors)/2, 128, true)

	idx, cleanup, err := newProfileIndex(
		len(st.vectors),
		ctx.opts.efSearch,
		ctx.opts.m,
		ctx.opts.efConst,
	)
	if err != nil {
		return err
	}
	st.idx = idx
	ctx.cleanup = cleanup

	if err := st.idx.BatchInsert(st.vectors, nil); err != nil {
		return err
	}
	st.exact = profileSetFromNodes(
		profileExactTopK(st.vectors, st.query, st.allow, profileFilteredK),
	)
	st.results = make([]hnsw.Node, 0, profileFilteredK)
	if _, err := st.idx.SearchAllowedInto(st.results[:0], st.query, profileFilteredK, st.allow); err != nil {
		return err
	}

	ctx.state = st
	return nil
}

func runFiltered(ctx *profileContext) error {
	st := ctx.state.(*filteredState)
	iterations := max(ctx.opts.repeats, 1) * 256
	latencies := make([]time.Duration, 0, iterations)
	results := st.results

	start := time.Now()
	for i := 0; i < iterations; i++ {
		iterStart := time.Now()
		var err error
		results, err = st.idx.SearchAllowedInto(results[:0], st.query, profileFilteredK, st.allow)
		if err != nil {
			return err
		}
		latencies = append(latencies, time.Since(iterStart))
	}
	elapsed := time.Since(start)

	fmt.Printf("allow=%d k=%d iterations=%d\n", st.allow.Len(), profileFilteredK, iterations)
	fmt.Printf("Recall@%d: %.4f\n", profileFilteredK, profileRecall(results, st.exact))
	fmt.Printf("QPS: %.0f\n", float64(iterations)/elapsed.Seconds())
	printLatencySummary(latencies)
	return nil
}

func prepareBuild(ctx *profileContext) error {
	st := &buildState{
		vectors: profileVectors(profileNodes, profileDims, 0x11, 0x22),
	}

	idx, cleanup, err := newProfileIndex(
		len(st.vectors),
		ctx.opts.efSearch,
		ctx.opts.m,
		ctx.opts.efConst,
	)
	if err != nil {
		return err
	}
	st.idx = idx
	ctx.cleanup = cleanup
	ctx.state = st
	return nil
}

func runBuild(ctx *profileContext) error {
	st := ctx.state.(*buildState)
	start := time.Now()
	if err := st.idx.BatchInsert(st.vectors, nil); err != nil {
		return err
	}
	elapsed := time.Since(start)

	fmt.Printf("vectors=%d dims=%d elapsed=%s throughput=%.0f vecs/sec\n",
		len(st.vectors),
		profileDims,
		elapsed.Round(time.Microsecond),
		float64(len(st.vectors))/elapsed.Seconds(),
	)
	return nil
}

func prepareDelete(ctx *profileContext) error {
	st := &deleteState{
		vectors:   profileVectors(profileNodes, profileDims, 0x33, 0x44),
		deleteIDs: profileDeleteIDs(profileNodes, profileDeleteCount),
	}

	idx, cleanup, err := newProfileIndex(
		len(st.vectors),
		ctx.opts.efSearch,
		ctx.opts.m,
		ctx.opts.efConst,
	)
	if err != nil {
		return err
	}
	st.idx = idx
	ctx.cleanup = cleanup

	if err := st.idx.BatchInsert(st.vectors, nil); err != nil {
		return err
	}

	ctx.state = st
	return nil
}

func runDelete(ctx *profileContext) error {
	st := ctx.state.(*deleteState)
	start := time.Now()
	deleted, err := st.idx.BulkDelete(st.deleteIDs)
	if err != nil {
		return err
	}
	elapsed := time.Since(start)

	fmt.Printf("deleted=%d requested=%d elapsed=%s throughput=%.0f ids/sec\n",
		deleted,
		len(st.deleteIDs),
		elapsed.Round(time.Microsecond),
		float64(deleted)/elapsed.Seconds(),
	)
	return nil
}

func prepareVacuum(ctx *profileContext) error {
	st := &vacuumState{
		vectors:   profileVectors(profileNodes, profileDims, 0x55, 0x66),
		deleteIDs: profileDeleteIDs(profileNodes, profileDeleteCount),
	}

	idx, cleanup, err := newProfileIndex(
		len(st.vectors),
		ctx.opts.efSearch,
		ctx.opts.m,
		ctx.opts.efConst,
	)
	if err != nil {
		return err
	}
	st.idx = idx
	ctx.cleanup = cleanup

	if err := st.idx.BatchInsert(st.vectors, nil); err != nil {
		return err
	}
	if deleted, err := st.idx.BulkDelete(st.deleteIDs); err != nil {
		return err
	} else if deleted != len(st.deleteIDs) {
		return fmt.Errorf("vacuum setup deleted %d of %d ids", deleted, len(st.deleteIDs))
	}

	ctx.state = st
	return nil
}

func runVacuum(ctx *profileContext) error {
	st := ctx.state.(*vacuumState)
	start := time.Now()
	if err := st.idx.Vacuum(); err != nil {
		return err
	}
	elapsed := time.Since(start)

	fmt.Printf("deleted=%d elapsed=%s\n", len(st.deleteIDs), elapsed.Round(time.Microsecond))
	return nil
}

func preparePlanner(ctx *profileContext) error {
	st := &plannerState{
		vectors: profileFilteredVectors(profileFilteredNodes, profileDims, 0x71, 0x19),
	}
	st.query = st.vectors[profileFilteredNodes/2]

	idx, cleanup, err := newProfileIndex(
		len(st.vectors),
		ctx.opts.efSearch,
		ctx.opts.m,
		ctx.opts.efConst,
	)
	if err != nil {
		return err
	}
	st.idx = idx
	ctx.cleanup = cleanup

	if err := st.idx.BatchInsert(st.vectors, nil); err != nil {
		return err
	}

	st.cases = make([]plannerCase, 0, len(plannerSpecs))
	for _, spec := range plannerSpecs {
		allow := profileAllowList(
			len(st.vectors),
			len(st.vectors)/2,
			spec.allowCount,
			spec.correlated,
		)
		exact := profileSetFromNodes(profileExactTopK(st.vectors, st.query, allow, spec.k))
		st.cases = append(st.cases, plannerCase{
			selectivity: spec.selectivity,
			allowCount:  spec.allowCount,
			k:           spec.k,
			correlated:  spec.correlated,
			allow:       allow,
			exact:       exact,
			expected: profilePlanName(
				profileChoosePlan(allow.Len(), spec.k, uint32(len(st.vectors)), profileEfSearch),
			),
			results: make([]hnsw.Node, 0, spec.k),
		})
	}

	ctx.state = st
	return nil
}

func runPlanner(ctx *profileContext) error {
	st := ctx.state.(*plannerState)
	repeats := max(ctx.opts.repeats, 1) * 64

	for _, tc := range st.cases {
		latencies := make([]time.Duration, 0, repeats)
		results := tc.results
		var last []hnsw.Node

		start := time.Now()
		for i := 0; i < repeats; i++ {
			iterStart := time.Now()
			var err error
			results, err = st.idx.SearchPlannedInto(results[:0], st.query, tc.k, tc.allow)
			if err != nil {
				return err
			}
			last = results
			latencies = append(latencies, time.Since(iterStart))
		}
		elapsed := time.Since(start)

		fmt.Printf("%s plan=%s recall=%.4f qps=%.0f mean=%s p50=%s p99=%s\n",
			tc.label(),
			tc.expected,
			profileRecall(last, tc.exact),
			float64(repeats)/elapsed.Seconds(),
			profileMean(latencies).Round(time.Microsecond),
			profilePercentile(latencies, 0.50).Round(time.Microsecond),
			profilePercentile(latencies, 0.99).Round(time.Microsecond),
		)
	}

	return nil
}

func profileVectors(n, dims int, seed1, seed2 uint64) [][]float32 {
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

func profileFilteredVectors(n, dims int, seed1, seed2 uint64) [][]float32 {
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

func profileAllowList(total, center, count int, correlated bool) hnsw.AllowList {
	if count < 0 {
		count = 0
	}
	if count > total {
		count = total
	}

	bits := make([]uint64, (total+63)/64)
	if count == 0 {
		return hnsw.NewAllowBitset(bits)
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
		bits[id/64] |= uint64(1) << (uint(id) % 64)
	}

	return hnsw.NewAllowBitset(bits)
}

func profileDeleteIDs(total, count int) []uint32 {
	if count > total {
		count = total
	}
	ids := make([]uint32, count)
	for i := range ids {
		ids[i] = uint32(i)
	}
	return ids
}

func profileExactTopK(
	vectors [][]float32,
	query []float32,
	allow hnsw.AllowList,
	k int,
) []hnsw.Node {
	if k <= 0 {
		return nil
	}

	allowed := allow.Len() == 0
	nodes := make([]hnsw.Node, 0, len(vectors))
	for id, vec := range vectors {
		if !allowed && !allow.Contains(uint32(id)) {
			continue
		}
		nodes = append(nodes, hnsw.Node{
			ID:       uint32(id),
			Distance: hnsw.L2(query, vec),
		})
	}

	slices.SortFunc(nodes, func(a, b hnsw.Node) int {
		if a.Distance < b.Distance {
			return -1
		}
		if a.Distance > b.Distance {
			return 1
		}
		switch {
		case a.ID < b.ID:
			return -1
		case a.ID > b.ID:
			return 1
		default:
			return 0
		}
	})

	if len(nodes) > k {
		nodes = nodes[:k]
	}
	return nodes
}

func exactSetsForQueries(vectors, queries [][]float32, k int) []map[uint32]struct{} {
	out := make([]map[uint32]struct{}, len(queries))
	for i, query := range queries {
		out[i] = profileSetFromNodes(profileExactTopK(vectors, query, hnsw.AllowList{}, k))
	}
	return out
}

func groundTruthSets(gt [][]int32, limit int) []map[uint32]struct{} {
	out := make([]map[uint32]struct{}, len(gt))
	for i, row := range gt {
		n := limit
		if n > len(row) {
			n = len(row)
		}
		set := make(map[uint32]struct{}, n)
		for _, id := range row[:n] {
			set[uint32(id)] = struct{}{}
		}
		out[i] = set
	}
	return out
}

func profileSetFromNodes(nodes []hnsw.Node) map[uint32]struct{} {
	set := make(map[uint32]struct{}, len(nodes))
	for _, node := range nodes {
		set[node.ID] = struct{}{}
	}
	return set
}

func profileRecall(results []hnsw.Node, exact map[uint32]struct{}) float64 {
	if len(exact) == 0 {
		return 1
	}

	hits := profileRecallHits(results, exact)
	return float64(hits) / float64(len(exact))
}

func profileRecallHits(results []hnsw.Node, exact map[uint32]struct{}) int {
	hits := 0
	for _, node := range results {
		if _, ok := exact[node.ID]; ok {
			hits++
		}
	}
	return hits
}

func profileMean(latencies []time.Duration) time.Duration {
	if len(latencies) == 0 {
		return 0
	}

	var total time.Duration
	for _, latency := range latencies {
		total += latency
	}
	return total / time.Duration(len(latencies))
}

func profilePercentile(latencies []time.Duration, pct float64) time.Duration {
	if len(latencies) == 0 {
		return 0
	}

	sorted := slices.Clone(latencies)
	slices.SortFunc(sorted, func(a, b time.Duration) int {
		switch {
		case a < b:
			return -1
		case a > b:
			return 1
		default:
			return 0
		}
	})

	idx := int(float64(len(sorted)-1) * pct)
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

func profileMeanDuration(latencies []time.Duration) time.Duration {
	return profileMean(latencies)
}

func printLatencySummary(latencies []time.Duration) {
	fmt.Printf("Mean: %s\n", profileMeanDuration(latencies).Round(time.Microsecond))
	fmt.Printf("p50:  %s\n", profilePercentile(latencies, 0.50).Round(time.Microsecond))
	fmt.Printf("p90:  %s\n", profilePercentile(latencies, 0.90).Round(time.Microsecond))
	fmt.Printf("p99:  %s\n", profilePercentile(latencies, 0.99).Round(time.Microsecond))
	fmt.Printf("Worst:%s\n", profilePercentile(latencies, 1.00).Round(time.Microsecond))
}

func profileChoosePlan(allowLen, k int, nodeCount uint32, efSearch int) searchPlanName {
	if allowLen <= 0 || k <= 0 || nodeCount == 0 {
		return searchPlanExact
	}

	exactLimit := max(32, k*4)
	if k <= efSearch && allowLen <= exactLimit {
		return searchPlanExact
	}

	denseLimit := max(64, int(nodeCount)/3)
	if allowLen >= denseLimit {
		return searchPlanStandard
	}

	return searchPlanFiltered
}

func profilePlanName(plan searchPlanName) string {
	return string(plan)
}

func (c plannerCase) label() string {
	corr := "far"
	if c.correlated {
		corr = "near"
	}
	return fmt.Sprintf("%s/k=%d/%s", c.selectivity, c.k, corr)
}

type binReader struct {
	data   []byte
	offset int
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
