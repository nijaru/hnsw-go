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

    idx := hnsw.NewIndex(storage, hnsw.L2)
    idx.SetEfSearch(200)
    idx.SetEfConst(200)

    // Insert with optional metadata
    meta := []byte("{\"text\": \"hello world\"}")
    idx.Insert([]float32{0.1, 0.2, /*...*/}, meta)

    // Zero-allocation search path
    results := make([]hnsw.Node, 0, 10)
    results, _ = idx.SearchInto(results, query, 10)
    
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

### Index Operations

| Method | Description |
|--------|-------------|
| `NewIndex(storage, distFunc)` | Create index from storage. Defaults: `efSearch=16`, `efConst=200`. Use `SetEfSearch` and `SetEfConst` to override. |
| `Insert(vec, meta)` error | Insert a vector with optional metadata |
| `BatchInsert(vecs, metas)` error | Insert multiple vectors/metas sequentially to avoid per-node lock contention |
| `Replace(id, vec, meta)` error | Refresh a vector in place |
| `Delete(id)` error | Mark a node as logically deleted |
| `Rebuild()` error / `Vacuum()` | Perform a compacting rewrite of the index to reclaim space |

### Search

| Method | Description |
|--------|-------------|
| `Search(query, k)` ([]Node, error) | Search for k nearest neighbors (allocates result slice) |
| `SearchInto(dst, query, k)` ([]Node, error) | Zero-alloc search appending to a caller-provided slice |
| `SearchAllowed(query, k, allow)` ([]Node, error) | Search restricted to IDs in an `AllowList` |
| `SearchPlanned(query, k, allow)` ([]Node, error) | Auto-selects exact vs HNSW search based on allow-list size |

### Filtering

`hnsw-go` includes embedded filter indexes for fast, zero-alloc generation of allow-lists:

- **`TermIndex`**: Inverted index mapping string terms (tenants, tags) to node IDs. Supports `Allow`, `AllowIntersect`, and `AllowUnion`.
- **`RangeIndex`**: Index mapping field names to sorted `float64` values. Supports `AllowRange`.

### Segmented Index (Coordinator)

For larger embedded workloads, `SegmentedIndex` provides an immutable view over a mutable head segment and zero or more frozen segments. It maps stable global IDs to segment-local IDs, merging search results correctly without mutating frozen data.

| Method | Description |
|--------|-------------|
| `NewSegmentedIndexFrom(head, frozen...)` | Create a coordinator mapping global IDs |
| `Publish(head, frozen...)` error | Atomically publish a new layout of segments |
| `SearchInto(dst, query, k)` ([]Node, error) | Zero-alloc merged top-k search across segments |

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

## Development

Run `make hooks` once per clone to enable the repo-local pre-commit hook. It auto-formats staged Go files and re-stages them before commit.

## Profiling

Use `make profile WORKLOAD=search` for the synthetic search workload, or set `PROFILE_ARGS='-sift sift10k_test.bin'` to run the SIFT dataset through the same harness. Other saved workloads are `filtered`, `build`, `delete`, `vacuum`, and `planner`.

Profiles are written as `.profiles/<workload>.cpu.prof` and `.profiles/<workload>.heap.prof`. Open them with `go tool pprof`, for example:

```sh
go tool pprof -http=:0 .profiles/search.cpu.prof
```

## Requirements

- Go 1.24+
- `golang.org/x/sys/unix` (mmap)
