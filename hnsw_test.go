package hnsw

import (
	"math/rand/v2"
	"os"
	"sync"
	"sync/atomic"
	"testing"
)

func removeTestFiles(path string) {
	os.Remove(path)
	os.Remove(path + ".vec")
	os.Remove(path + ".upper")
}

func TestHNSW(t *testing.T) {
	path := "test.hnsw"
	defer removeTestFiles(path)

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
	defer removeTestFiles(path)

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

	if storage.readUint32(32) != 10 {
		t.Errorf("expected 10 allocated nodes, got %d", storage.readUint32(32))
	}
}

func TestGrowTriggered(t *testing.T) {
	path := "test_grow_trigger.hnsw"
	defer removeTestFiles(path)

	config := IndexConfig{
		Dims:     4,
		M:        4,
		MMax0:    8,
		MaxLevel: 4,
	}

	storage, err := NewStorage(path, config, 5)
	if err != nil {
		t.Fatalf("create storage: %v", err)
	}
	defer storage.Close()

	idx := NewIndex(storage, L2, 10, 10)

	for i := 0; i < 20; i++ {
		vec := make([]float32, 4)
		for j := range vec {
			vec[j] = rand.Float32()
		}
		if err := idx.Insert(vec); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}

	if idx.Len() != 20 {
		t.Errorf("expected 20 nodes, got %d", idx.Len())
	}

	query := make([]float32, 4)
	for j := range query {
		query[j] = rand.Float32()
	}
	results, err := idx.Search(query, 5)
	if err != nil {
		t.Fatalf("search after grow: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("no results after grow")
	}
}

func TestSearchEmptyIndex(t *testing.T) {
	path := "test_empty.hnsw"
	defer removeTestFiles(path)

	config := IndexConfig{
		Dims:     4,
		M:        4,
		MMax0:    8,
		MaxLevel: 4,
	}

	storage, err := NewStorage(path, config, 10)
	if err != nil {
		t.Fatalf("create storage: %v", err)
	}
	defer storage.Close()

	idx := NewIndex(storage, L2, 10, 10)

	results, err := idx.Search(make([]float32, 4), 5)
	if err != nil {
		t.Fatalf("search empty: %v", err)
	}
	if results != nil {
		t.Errorf("expected nil results from empty index, got %v", results)
	}
}

func TestSearchKZero(t *testing.T) {
	path := "test_k0.hnsw"
	defer removeTestFiles(path)

	config := IndexConfig{
		Dims:     4,
		M:        4,
		MMax0:    8,
		MaxLevel: 4,
	}

	storage, err := NewStorage(path, config, 10)
	if err != nil {
		t.Fatalf("create storage: %v", err)
	}
	defer storage.Close()

	idx := NewIndex(storage, L2, 10, 10)

	vec := make([]float32, 4)
	for j := range vec {
		vec[j] = rand.Float32()
	}
	idx.Insert(vec)

	results, err := idx.Search(make([]float32, 4), 0)
	if err != nil {
		t.Fatalf("search k=0: %v", err)
	}
	if results != nil {
		t.Errorf("expected nil results for k=0, got %v", results)
	}
}

func TestSearchKGreaterThanNodeCount(t *testing.T) {
	path := "test_kbig.hnsw"
	defer removeTestFiles(path)

	config := IndexConfig{
		Dims:     4,
		M:        4,
		MMax0:    8,
		MaxLevel: 4,
	}

	storage, err := NewStorage(path, config, 10)
	if err != nil {
		t.Fatalf("create storage: %v", err)
	}
	defer storage.Close()

	idx := NewIndex(storage, L2, 10, 10)

	for i := 0; i < 3; i++ {
		vec := make([]float32, 4)
		for j := range vec {
			vec[j] = rand.Float32()
		}
		idx.Insert(vec)
	}

	results, err := idx.Search(make([]float32, 4), 100)
	if err != nil {
		t.Fatalf("search k>count: %v", err)
	}
	if len(results) != 3 {
		t.Errorf("expected 3 results, got %d", len(results))
	}
}

func TestWrongDimensions(t *testing.T) {
	path := "test_wrongdims.hnsw"
	defer removeTestFiles(path)

	config := IndexConfig{
		Dims:     4,
		M:        4,
		MMax0:    8,
		MaxLevel: 4,
	}

	storage, err := NewStorage(path, config, 10)
	if err != nil {
		t.Fatalf("create storage: %v", err)
	}
	defer storage.Close()

	idx := NewIndex(storage, L2, 10, 10)

	err = idx.Insert(make([]float32, 8))
	if err == nil {
		t.Fatal("expected error for wrong dims in Insert")
	}

	_, err = idx.Search(make([]float32, 8), 5)
	if err == nil {
		t.Fatal("expected error for wrong dims in Search")
	}
}

func TestInvalidConfig(t *testing.T) {
	path := "test_badconfig.hnsw"
	defer removeTestFiles(path)

	_, err := NewStorage(path, IndexConfig{Dims: 0, M: 16, MaxLevel: 4}, 10)
	if err == nil {
		t.Fatal("expected error for Dims=0")
		os.Remove(path)
	}

	_, err = NewStorage(path, IndexConfig{Dims: 4, M: 0, MaxLevel: 4}, 10)
	if err == nil {
		t.Fatal("expected error for M=0")
		os.Remove(path)
	}

	_, err = NewStorage(path, IndexConfig{Dims: 4, M: 16, MaxLevel: 0}, 10)
	if err == nil {
		t.Fatal("expected error for MaxLevel=0")
		os.Remove(path)
	}
}

func TestCosineDistance(t *testing.T) {
	path := "test_cosine.hnsw"
	defer removeTestFiles(path)

	config := IndexConfig{
		Dims:     4,
		M:        4,
		MMax0:    8,
		MaxLevel: 4,
	}

	storage, err := NewStorage(path, config, 100)
	if err != nil {
		t.Fatalf("create storage: %v", err)
	}
	defer storage.Close()

	idx := NewIndex(storage, Cosine, 10, 10)

	vecs := [][]float32{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{1, 1, 0, 0},
		{0, 0, 1, 0},
	}
	for _, v := range vecs {
		if err := idx.Insert(v); err != nil {
			t.Fatalf("insert: %v", err)
		}
	}

	results, err := idx.Search([]float32{1, 0, 0, 0}, 2)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("no results")
	}
	if results[0].ID != 0 {
		t.Errorf(
			"expected ID 0 (identical vector), got %d dist=%f",
			results[0].ID,
			results[0].Distance,
		)
	}
}

func TestDotDistance(t *testing.T) {
	path := "test_dot.hnsw"
	defer removeTestFiles(path)

	config := IndexConfig{
		Dims:     4,
		M:        4,
		MMax0:    8,
		MaxLevel: 4,
	}

	storage, err := NewStorage(path, config, 100)
	if err != nil {
		t.Fatalf("create storage: %v", err)
	}
	defer storage.Close()

	idx := NewIndex(storage, Dot, 10, 10)

	vecs := [][]float32{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{2, 1, 0, 0},
		{0, 0, 1, 0},
	}
	for _, v := range vecs {
		if err := idx.Insert(v); err != nil {
			t.Fatalf("insert: %v", err)
		}
	}

	results, err := idx.Search([]float32{1, 0, 0, 0}, 2)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("no results")
	}
	if results[0].ID != 2 {
		t.Errorf(
			"expected ID 2 (highest dot product), got %d dist=%f",
			results[0].ID,
			results[0].Distance,
		)
	}
}

func TestConcurrentInsertSearch(t *testing.T) {
	path := "test_concurrent.hnsw"
	defer removeTestFiles(path)

	config := IndexConfig{
		Dims:     128,
		M:        16,
		MMax0:    32,
		MaxLevel: 16,
	}

	storage, err := NewStorage(path, config, 5000)
	if err != nil {
		t.Fatalf("create storage: %v", err)
	}
	defer storage.Close()

	idx := NewIndex(storage, L2, 50, 50)

	for i := 0; i < 100; i++ {
		vec := make([]float32, 128)
		for j := range vec {
			vec[j] = rand.Float32()
		}
		if err := idx.Insert(vec); err != nil {
			t.Fatalf("seed insert %d: %v", i, err)
		}
	}

	const inserters = 8
	const searchers = 16
	const opsPerGoroutine = 500

	var wg sync.WaitGroup
	var searchErrs, insertErrs atomic.Int32

	for s := 0; s < searchers; s++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			query := make([]float32, 128)
			for i := 0; i < opsPerGoroutine; i++ {
				for j := range query {
					query[j] = rand.Float32()
				}
				results, err := idx.Search(query, 10)
				if err != nil {
					searchErrs.Add(1)
					return
				}
				if results == nil {
					searchErrs.Add(1)
					return
				}
			}
		}()
	}

	for g := 0; g < inserters; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < opsPerGoroutine; i++ {
				vec := make([]float32, 128)
				for j := range vec {
					vec[j] = rand.Float32()
				}
				if err := idx.Insert(vec); err != nil {
					insertErrs.Add(1)
					return
				}
			}
		}()
	}

	wg.Wait()

	if se := searchErrs.Load(); se > 0 {
		t.Errorf("%d search errors during concurrent ops", se)
	}
	if ie := insertErrs.Load(); ie > 0 {
		t.Errorf("%d insert errors during concurrent ops", ie)
	}

	stats := idx.Stats()
	expected := uint32(100 + inserters*opsPerGoroutine)
	if stats.NodeCount != expected {
		t.Errorf("expected %d nodes, got %d", expected, stats.NodeCount)
	}

	query := make([]float32, 128)
	for j := range query {
		query[j] = rand.Float32()
	}
	results, err := idx.Search(query, 10)
	if err != nil {
		t.Fatalf("post-concurrent search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("no results after concurrent ops")
	}
	t.Logf("Concurrent test passed: %d nodes, final search OK", stats.NodeCount)
}

func TestBatchInsert(t *testing.T) {
	path := "test_batch.hnsw"
	defer removeTestFiles(path)

	config := IndexConfig{
		Dims:     128,
		M:        16,
		MMax0:    32,
		MaxLevel: 16,
	}

	storage, err := NewStorage(path, config, 10)
	if err != nil {
		t.Fatalf("failed to create storage: %v", err)
	}
	defer storage.Close()

	idx := NewIndex(storage, L2, 50, 50)

	vectors := make([][]float32, 100)
	for i := range vectors {
		vectors[i] = make([]float32, 128)
		for j := range vectors[i] {
			vectors[i][j] = rand.Float32()
		}
	}

	if err := idx.BatchInsert(vectors); err != nil {
		t.Fatalf("BatchInsert failed: %v", err)
	}

	if idx.Len() != 100 {
		t.Errorf("expected 100 nodes, got %d", idx.Len())
	}

	results, err := idx.Search(vectors[0], 1)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}
	if len(results) == 0 || results[0].ID != 0 {
		t.Errorf("expected result ID 0, got %v", results)
	}
}

func TestMultiProbe(t *testing.T) {
	path := "test_multiprobe.hnsw"
	defer removeTestFiles(path)

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

	idx := NewIndex(storage, L2, 50, 50)

	// Insert random vectors
	for i := 0; i < 500; i++ {
		vec := make([]float32, 128)
		for j := range vec {
			vec[j] = rand.Float32()
		}
		idx.Insert(vec)
	}

	query := make([]float32, 128)
	for j := range query {
		query[j] = rand.Float32()
	}

	// Compare k=1 search with different probe counts
	idx.SetEfSearch(1) // Low efSearch to make it harder
	idx.SetProbes(1)
	res1, _ := idx.Search(query, 1)

	idx.SetProbes(4)
	res4, _ := idx.Search(query, 1)

	if len(res1) > 0 && len(res4) > 0 {
		if res4[0].Distance > res1[0].Distance {
			t.Errorf("multi-probe (4) found worse result than single-probe: %f > %f", res4[0].Distance, res1[0].Distance)
		}
		t.Logf("Distances: probe=1: %f, probe=4: %f", res1[0].Distance, res4[0].Distance)
	}
}

func TestPersistence(t *testing.T) {
	path := "test_persist.hnsw"
	defer removeTestFiles(path)

	config := IndexConfig{
		Dims:     4,
		M:        4,
		MMax0:    8,
		MaxLevel: 4,
	}

	vecs := make([][]float32, 50)
	for i := range vecs {
		vecs[i] = make([]float32, 4)
		for j := range vecs[i] {
			vecs[i][j] = rand.Float32()
		}
	}

	func() {
		storage, err := NewStorage(path, config, 100)
		if err != nil {
			t.Fatalf("create: %v", err)
		}
		defer storage.Close()

		idx := NewIndex(storage, L2, 20, 20)
		for _, v := range vecs {
			if err := idx.Insert(v); err != nil {
				t.Fatalf("insert: %v", err)
			}
		}
	}()

	storage2, err := NewStorage(path, config, 0)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer storage2.Close()

	idx2 := NewIndex(storage2, L2, 20, 20)
	stats := idx2.Stats()
	if stats.NodeCount != 50 {
		t.Errorf("expected 50 nodes after reopen, got %d", stats.NodeCount)
	}

	results, err := idx2.Search(vecs[0], 5)
	if err != nil {
		t.Fatalf("search after reopen: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("no results after reopen")
	}
	if results[0].ID != 0 {
		t.Errorf("expected ID 0 (self-search), got %d dist=%f", results[0].ID, results[0].Distance)
	}
}
