# hnsw-go

A high-performance, pure Go implementation of Hierarchical Navigable Small World (HNSW) graphs for approximate nearest neighbor search.

Designed for in-process use by AI agents and applications that need fast vector similarity search with persistence. Zero external dependencies (pure Go, no CGO).

## Design

- **Mmap-backed storage** — Graph and vectors live in a memory-mapped file, bypassing the Go GC entirely. Reopen a file and the index is ready.
- **Zero-allocation search** — Hot path uses `sync.Pool` for search buffers. Benchmarked at exactly 1 alloc/op (the result slice).
- **Cache-optimized node layout** — Each node is a single 64-byte-aligned block containing metadata, vector, and all neighbor lists. Node ID directly resolves to a byte offset.
- **Thread-safe** — `RWMutex` protects mmap remapping during concurrent inserts. Per-node spinlocks for fine-grained neighbor updates.
- **Config validation** — Opening an existing file verifies stored parameters match the provided config, preventing silent corruption.

## Usage

```go
package main

import (
    "fmt"
    "github.com/nijaru/hnsw-go"
)

func main() {
    config := hnsw.IndexConfig{
        Dims:     128,
        M:        16,
        MMax0:    32,
        MaxLevel: 16,
    }

    storage, _ := hnsw.NewStorage("vectors.hnsw", config, 10000)
    defer storage.Close()

    idx := hnsw.NewIndex(storage, hnsw.L2, 200, 200)

    idx.Insert(vec)
    results, _ := idx.Search(query, 10)
    for _, r := range results {
        fmt.Printf("ID=%d Distance=%f\n", r.ID, r.Distance)
    }
}
```

## API

### Config

```go
type IndexConfig struct {
    Dims     uint32  // Vector dimensions
    M        uint32  // Max neighbors per layer (layer 1+)
    MMax0    uint32  // Max neighbors at layer 0 (default: M * 2)
    MaxLevel uint32  // Max graph levels
}
```

### Index

| Method | Description |
|--------|-------------|
| `NewIndex(storage, distFunc, efSearch, efConst)` | Create index from storage |
| `Insert(vec)` | Insert a vector |
| `Search(query, k)` | Search for k nearest neighbors |
| `SetEfSearch(ef)` | Adjust search effort at runtime |
| `SetEfConst(ef)` | Adjust construction effort at runtime |
| `Len()` | Number of indexed vectors |
| `Stats()` | Index statistics |

### Distance Functions

- `L2` — Euclidean distance (ILP-optimized with 4 accumulators)
- `Cosine` — Cosine distance (1 - cosine similarity)
- `Dot` — Negative dot product

## Performance

On Apple M3 Max with SIFT 10k (128-dim, 10k vectors):

- **Recall@10:** 99.99%
- **Search latency:** ~128μs/query
- **Search allocations:** 1 alloc/op (result copy only)

## Requirements

- Go 1.26+
- `golang.org/x/sys/unix` (mmap)
