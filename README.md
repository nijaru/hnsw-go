# hnsw-go

A high-performance, pure Go implementation of Hierarchical Navigable Small World (HNSW) graphs for approximate nearest neighbor search.

Designed for in-process use by AI agents and applications that need fast vector similarity search with persistence. Zero external dependencies (pure Go, no CGO).

## Design

- **Paper-faithful algorithm** — Level generation uses `floor(-ln(U) / ln(M))` (Malkov & Yashunin, Algorithm 1). Neighbor selection implements the diversity heuristic (Algorithm 4) with pruned-connection backfill.
- **Mmap-backed storage** — Graph and vectors live in a memory-mapped file, bypassing the Go GC entirely. Reopen a file and the index is ready.
- **Zero-allocation search path** — Hot path uses `sync.Pool` for visited sets and heaps. Generation-based `uint8` visited array. Only allocation is the result slice returned to the caller.
- **Cache-optimized node layout** — Each node is a single 64-byte-aligned block containing metadata, vector, and all neighbor lists. Node ID directly resolves to a byte offset.
- **Thread-safe** — `RWMutex` protects mmap remapping during concurrent inserts. Per-node spinlocks for fine-grained neighbor updates.
- **Config validation** — Opening an existing file verifies stored parameters match the provided config, preventing silent corruption.

## Usage

```go
package main

import (
    "fmt"
    "github.com/omendb/hnsw-go"
)

func main() {
    config := hnsw.IndexConfig{
        Dims:     128,
        M:        16,
        MMax0:    32,      // defaults to 2*M if zero
        MaxLevel: 16,
    }

    storage, _ := hnsw.NewStorage("vectors.hnsw", config, 10000)
    defer storage.Close()

    idx := hnsw.NewIndex(storage, hnsw.L2, 200, 200)

    // Insert with optional metadata
    meta := []byte("{\"text\": \"hello world\"}")
    idx.Insert(vec, meta)

    results, _ := idx.Search(query, 10)
    for _, r := range results {
        fmt.Printf("ID=%d Distance=%f Meta=%s\n", r.ID, r.Distance, string(r.Metadata))
    }
}
```

## API

### Config

```go
type IndexConfig struct {
    Dims     uint32  // Vector dimensions (required)
    M        uint32  // Max neighbors per layer (required)
    MMax0    uint32  // Max neighbors at layer 0 (default: 2*M)
    MaxLevel uint32  // Max graph levels (required)
}
```

### Index

| Method | Description |
|--------|-------------|
| `NewIndex(storage, distFunc, efSearch, efConst)` | Create index from storage. efSearch defaults to 16, efConst to 200. |
| `Insert(vec, meta)` error | Insert a vector with optional metadata |
| `BatchInsert(vecs, metas)` error | Insert multiple vectors/metas in parallel |
| `Search(query, k)` ([]Node, error) | Search for k nearest neighbors (returns ID, Distance, and Metadata) |
| `SetEfSearch(ef)` | Adjust search effort at runtime |
| `SetEfConst(ef)` | Adjust construction effort at runtime |
| `Len()` int | Number of indexed vectors |
| `Stats()` Stats | Index statistics |

### Distance Functions

| Function | Metric | Notes |
|----------|--------|-------|
| `L2` | Euclidean distance | 4-accumulator loop for ILP |
| `Cosine` | 1 - cosine similarity | Normalizes on the fly |
| `Dot` | Negative dot product | For max-inner-product search |

## Performance

Apple M3 Max, SIFT 10k (128-dim, 10,000 vectors), M=16, efSearch=200:

| Metric | Value |
|--------|-------|
| Recall@10 | 99.99% |
| QPS | 7,433 |
| p50 latency | 133μs |
| p99 latency | 241μs |

## Requirements

- Go 1.24+
- `golang.org/x/sys/unix` (mmap)
