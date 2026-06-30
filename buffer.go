package hnsw

import "sync/atomic"

// searchBuffer holds per-search temporary state: a generation-based
// visited set, candidate heap, result heap, and output slice.
type searchBuffer struct {
	visited    []uint8
	results    NodeMaxHeap
	candidates nodeHeap
	out        []Node
	gen        uint8
}

func newSearchBuffer(visitedCap, heapCap, outCap int) *searchBuffer {
	if visitedCap < 1 {
		visitedCap = 1
	}
	if heapCap < 1 {
		heapCap = 1
	}
	if outCap < 1 {
		outCap = 1
	}

	return &searchBuffer{
		visited:    make([]uint8, visitedCap),
		results:    NodeMaxHeap{Nodes: make([]Node, 0, heapCap)},
		candidates: nodeHeap{Nodes: make([]Node, 0, heapCap)},
		out:        make([]Node, 0, outCap),
		gen:        1,
	}
}

func (b *searchBuffer) isVisited(id uint32) bool {
	return b.visited[id] == b.gen
}

func (b *searchBuffer) visit(id uint32) {
	b.visited[id] = b.gen
}

func (b *searchBuffer) reset(maxNodes uint32) {
	b.gen++
	if b.gen == 0 {
		for i := range b.visited {
			b.visited[i] = 0
		}
		b.gen = 1
	}
	if uint32(len(b.visited)) < maxNodes {
		b.visited = make([]uint8, maxNodes+2048)
		b.gen = 1
	}
	b.results.Nodes = b.results.Nodes[:0]
	b.candidates.Nodes = b.candidates.Nodes[:0]
	b.out = b.out[:0]
}

// searchBufferSlot holds an atomic pointer to a search buffer, used as a
// single-producer-single-consumer fast path to avoid pool contention.
type searchBufferSlot struct {
	buf atomic.Pointer[searchBuffer]
}
