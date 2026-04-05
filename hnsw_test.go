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
	os.Remove(path + ".del")
	os.Remove(path + ".meta")
}

func TestBulkDelete(t *testing.T) {
	path := "test_bulk_delete.hnsw"
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

	idx := NewIndex(storage, L2)
	idx.SetEfSearch(50)
	idx.SetEfConst(50)

	// Insert 100 vectors
	vectors := make([][]float32, 100)
	for i := range vectors {
		vectors[i] = make([]float32, 128)
		vectors[i][0] = float32(i)
		idx.Insert(vectors[i], nil)
	}

	// Bulk delete IDs 10-59 (50 items)
	n, err := idx.BulkDelete(func() []uint32 {
		ids := make([]uint32, 50)
		for i := 0; i < 50; i++ {
			ids[i] = uint32(i + 10)
		}
		return ids
	}())
	if err != nil {
		t.Fatalf("BulkDelete failed: %v", err)
	}
	if n != 50 {
		t.Errorf("expected 50 deleted, got %d", n)
	}

	// Deleting already-deleted IDs is idempotent
	n2, err := idx.BulkDelete([]uint32{10, 11, 12})
	if err != nil {
		t.Fatalf("BulkDelete re-delete failed: %v", err)
	}
	if n2 != 0 {
		t.Errorf("expected 0 deleted (already deleted), got %d", n2)
	}

	// No deleted ID should appear in search results
	query := make([]float32, 128)
	query[0] = 30.0
	results, err := idx.Search(query, 20)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	for _, r := range results {
		if r.ID >= 10 && r.ID < 60 {
			t.Errorf("deleted ID %d appeared in results", r.ID)
		}
	}

	// Freelist should have 50 IDs
	idx.mu.RLock()
	fl := len(idx.freelist)
	idx.mu.RUnlock()
	if fl != 50 {
		t.Errorf("expected 50 freelist entries, got %d", fl)
	}
}

func TestFreelistBookkeeping(t *testing.T) {
	path := "test_freelist.hnsw"
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

	idx := NewIndex(storage, L2)
	idx.SetEfSearch(10)
	idx.SetEfConst(10)

	// Insert 10 vectors with distinct values
	vecs := make([][]float32, 10)
	for i := range vecs {
		vecs[i] = []float32{float32(i * 100), 0, 0, 0}
		idx.Insert(vecs[i], nil)
	}

	// Delete IDs 2, 3, 4
	idx.BulkDelete([]uint32{2, 3, 4})

	// The freelist tracks the deletes, but inserts should still allocate fresh IDs.
	newVecs := make([][]float32, 5)
	for i := range newVecs {
		newVecs[i] = []float32{float32(1000 + i), 0, 0, 0}
	}
	if err := idx.BatchInsert(newVecs, nil); err != nil {
		t.Fatalf("BatchInsert: %v", err)
	}

	if idx.Len() != 15 {
		t.Fatalf("expected len 15 after inserts, got %d", idx.Len())
	}

	// All 5 new vectors should be findable at new IDs.
	for i, v := range newVecs {
		results, err := idx.Search(v, 1)
		if err != nil || len(results) == 0 {
			t.Fatalf("search for %+v: %v", v, err)
		}
		wantID := uint32(10 + i)
		if results[0].ID != wantID {
			t.Fatalf("expected fresh ID %d for vector %v, got %d", wantID, v, results[0].ID)
		}
	}

	// The deleted IDs remain tombstones until vacuum.
	for _, id := range []uint32{2, 3, 4} {
		if !idx.storage.IsDeleted(id) {
			t.Fatalf("expected ID %d to remain deleted", id)
		}
	}

	// The freelist is bookkeeping, not a live allocator.
	idx.mu.RLock()
	fl := len(idx.freelist)
	idx.mu.RUnlock()
	if fl != 3 {
		t.Fatalf("expected freelist length 3, got %d", fl)
	}
}

func TestConcurrentBulkDeleteSearch(t *testing.T) {
	path := "test_concurrent_deletes.hnsw"
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

	idx := NewIndex(storage, L2)
	idx.SetEfSearch(50)
	idx.SetEfConst(50)

	// Seed 1000 nodes
	for i := 0; i < 1000; i++ {
		vec := make([]float32, 128)
		for j := range vec {
			vec[j] = rand.Float32()
		}
		idx.Insert(vec, nil)
	}

	var wg sync.WaitGroup
	var searchErrs atomic.Int32

	// 4 goroutines doing bulk deletes
	for g := 0; g < 4; g++ {
		wg.Add(1)
		go func(seed int) {
			defer wg.Done()
			rng := rand.New(rand.NewPCG(uint64(seed), 0))
			for b := 0; b < 10; b++ {
				ids := make([]uint32, 20)
				for i := range ids {
					ids[i] = rng.Uint32N(uint32(idx.Len()))
				}
				idx.BulkDelete(ids)
			}
		}(g)
	}

	// 8 goroutines searching
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			query := make([]float32, 128)
			for i := 0; i < 200; i++ {
				for j := range query {
					query[j] = rand.Float32()
				}
				_, err := idx.Search(query, 10)
				if err != nil {
					searchErrs.Add(1)
					return
				}
			}
		}()
	}

	wg.Wait()

	if se := searchErrs.Load(); se > 0 {
		t.Errorf("%d search errors during concurrent ops", se)
	}
}

func TestDelete(t *testing.T) {
	path := "test_delete.hnsw"
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

	idx := NewIndex(storage, L2)
	idx.SetEfSearch(50)
	idx.SetEfConst(50)

	// Insert 10 vectors
	vectors := make([][]float32, 10)
	for i := range vectors {
		vectors[i] = make([]float32, 128)
		vectors[i][0] = float32(i) // Make them distinct
		idx.Insert(vectors[i], nil)
	}

	// Delete ID 5
	if n := idx.Delete(5); n == 0 {
		t.Fatalf("Delete returned 0")
	}

	// Search for vector that was at ID 5
	query := make([]float32, 128)
	query[0] = 5.0
	results, err := idx.Search(query, 5)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	for _, res := range results {
		if res.ID == 5 {
			t.Errorf("found deleted node ID 5 in search results")
		}
	}

	// Delete ID 0 (entry point)
	if n := idx.Delete(0); n == 0 {
		t.Fatalf("Delete ID 0 returned 0")
	}

	results, err = idx.Search(query, 5)
	if err != nil {
		t.Fatalf("Search after delete ID 0 failed: %v", err)
	}
	for _, res := range results {
		if res.ID == 0 {
			t.Errorf("found deleted node ID 0 in search results")
		}
	}

	// Verify graph healing: neighbors of deleted nodes should no longer point to them
	for i := uint32(0); i < 10; i++ {
		if i == 5 || i == 0 {
			continue // These were deleted
		}
		for l := 0; l <= idx.storage.GetMaxLevel(i); l++ {
			nb := idx.storage.GetNeighbors(i, l)
			for _, nID := range nb {
				if nID == 5 || nID == 0 {
					t.Errorf("node %d still has deleted node %d as neighbor at level %d", i, nID, l)
				}
			}
		}
	}
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

	idx := NewIndex(storage, L2)
	idx.SetEfSearch(100)
	idx.SetEfConst(100)

	vectors := make([][]float32, 100)
	for i := range vectors {
		vectors[i] = make([]float32, 128)
		for j := range vectors[i] {
			vectors[i][j] = rand.Float32()
		}
		if err := idx.Insert(vectors[i], nil); err != nil {
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

	idx := NewIndex(storage, L2)
	idx.SetEfSearch(10)
	idx.SetEfConst(10)

	for i := 0; i < 20; i++ {
		vec := make([]float32, 4)
		for j := range vec {
			vec[j] = rand.Float32()
		}
		if err := idx.Insert(vec, nil); err != nil {
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

	idx := NewIndex(storage, L2)
	idx.SetEfSearch(10)
	idx.SetEfConst(10)

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

	idx := NewIndex(storage, L2)
	idx.SetEfSearch(10)
	idx.SetEfConst(10)

	vec := make([]float32, 4)
	for j := range vec {
		vec[j] = rand.Float32()
	}
	idx.Insert(vec, nil)

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

	idx := NewIndex(storage, L2)
	idx.SetEfSearch(10)
	idx.SetEfConst(10)

	for i := 0; i < 3; i++ {
		vec := make([]float32, 4)
		for j := range vec {
			vec[j] = rand.Float32()
		}
		idx.Insert(vec, nil)
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

	idx := NewIndex(storage, L2)
	idx.SetEfSearch(10)
	idx.SetEfConst(10)

	err = idx.Insert(make([]float32, 8), nil)
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

	idx := NewIndex(storage, Cosine)
	idx.SetEfSearch(10)
	idx.SetEfConst(10)

	vecs := [][]float32{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{1, 1, 0, 0},
		{0, 0, 1, 0},
	}
	for _, v := range vecs {
		if err := idx.Insert(v, nil); err != nil {
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

	idx := NewIndex(storage, Dot)
	idx.SetEfSearch(10)
	idx.SetEfConst(10)

	vecs := [][]float32{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{2, 1, 0, 0},
		{0, 0, 1, 0},
	}
	for _, v := range vecs {
		if err := idx.Insert(v, nil); err != nil {
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

	idx := NewIndex(storage, L2)
	idx.SetEfSearch(50)
	idx.SetEfConst(50)

	for i := 0; i < 100; i++ {
		vec := make([]float32, 128)
		for j := range vec {
			vec[j] = rand.Float32()
		}
		if err := idx.Insert(vec, nil); err != nil {
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
				if err := idx.Insert(vec, nil); err != nil {
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

	idx := NewIndex(storage, L2)
	idx.SetEfSearch(50)
	idx.SetEfConst(50)

	vectors := make([][]float32, 100)
	for i := range vectors {
		vectors[i] = make([]float32, 128)
		for j := range vectors[i] {
			vectors[i][j] = rand.Float32()
		}
	}

	if err := idx.BatchInsert(vectors, nil); err != nil {
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

	idx := NewIndex(storage, L2)
	idx.SetEfSearch(50)
	idx.SetEfConst(50)

	// Insert random vectors
	for i := 0; i < 500; i++ {
		vec := make([]float32, 128)
		for j := range vec {
			vec[j] = rand.Float32()
		}
		idx.Insert(vec, nil)
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
			t.Errorf(
				"multi-probe (4) found worse result than single-probe: %f > %f",
				res4[0].Distance,
				res1[0].Distance,
			)
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

		idx := NewIndex(storage, L2)
		idx.SetEfSearch(20)
		idx.SetEfConst(20)

		for _, v := range vecs {
			if err := idx.Insert(v, nil); err != nil {
				t.Fatalf("insert: %v", err)
			}
		}
	}()

	storage2, err := NewStorage(path, config, 0)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer storage2.Close()

	idx2 := NewIndex(storage2, L2)
	idx2.SetEfSearch(20)
	idx2.SetEfConst(20)

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
