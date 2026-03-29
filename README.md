# hnsw-go

A high-performance, pure Go implementation of Hierarchical Navigable Small World (HNSW) graphs for vector similarity search.

Designed for in-process use by AI agents (e.g., Canto), `hnsw-go` focuses on memory efficiency, cache locality, and zero-allocation hot paths.

## Key Features

- **Pure Go 1.26+:** No CGO or external dependencies.
- **TigerStyle Engineering:** Fixed-size memory layouts, pointer-free graph structures, and zero heap allocations on search/insertion hot paths.
- **Mmap-backed Storage:** Graph nodes and vectors are stored in a contiguous memory-mapped file, bypassing the Go GC and enabling persistence.
- **Cache-Optimized Layout:** Vectors and multi-layered neighbor lists are colocated for minimal cache misses.
- **SOTA Algorithms:** Implements diverse neighbor selection heuristics for better graph connectivity and search recall.

## Usage

```go
package main

import (
	"fmt"
	"github.com/omendb/hnsw-go/src"
)

func main() {
	config := hnsw.IndexConfig{
		Dims:     128,
		M:        16,
		MaxLevel: 16,
	}

	storage, _ := hnsw.NewStorage("data.hnsw", config, 10000)
	defer storage.Close()

	idx := hnsw.NewIndex(storage, hnsw.L2, 16, 100, 100)

	// Insert
	vec := make([]float32, 128)
	idx.Insert(vec)

	// Search
	results := idx.Search(vec, 10)
	for _, res := range results {
		fmt.Printf("ID: %d, Distance: %f\n", res.ID, res.Distance)
	}
}
```

## Performance & Optimization

- **GC Avoidance:** By storing millions of nodes in `mmap` and using `unsafe.Slice` for access, the Go garbage collector never sees the graph metadata or vector data.
- **Atomic Operations:** Node-level spinlocks ensure thread-safe insertions with minimal overhead.
- **Zero-Allocation Search:** Uses a `sync.Pool` for search buffers (visited lists, priority queues) to eliminate per-query allocations.
