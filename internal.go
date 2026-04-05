package hnsw

import "sync/atomic"

type Node struct {
	ID       uint32
	Distance float32
	Metadata []byte
}

type nodeHeap struct {
	Nodes []Node
}

func (h *nodeHeap) Push(n Node) {
	h.Nodes = append(h.Nodes, n)
	i := len(h.Nodes) - 1
	for i > 0 {
		p := (i - 1) / 2
		if h.Nodes[i].Distance >= h.Nodes[p].Distance {
			break
		}
		h.Nodes[i], h.Nodes[p] = h.Nodes[p], h.Nodes[i]
		i = p
	}
}

func (h *nodeHeap) Pop() Node {
	n := h.Nodes[0]
	last := len(h.Nodes) - 1
	h.Nodes[0] = h.Nodes[last]
	h.Nodes = h.Nodes[:last]
	i := 0
	for {
		l := 2*i + 1
		r := 2*i + 2
		smallest := i
		if l < len(h.Nodes) && h.Nodes[l].Distance < h.Nodes[smallest].Distance {
			smallest = l
		}
		if r < len(h.Nodes) && h.Nodes[r].Distance < h.Nodes[smallest].Distance {
			smallest = r
		}
		if smallest == i {
			break
		}
		h.Nodes[i], h.Nodes[smallest] = h.Nodes[smallest], h.Nodes[i]
		i = smallest
	}
	return n
}

type NodeMaxHeap struct {
	Nodes []Node
}

func (h *NodeMaxHeap) Push(n Node) {
	h.Nodes = append(h.Nodes, n)
	i := len(h.Nodes) - 1
	for i > 0 {
		p := (i - 1) / 2
		if h.Nodes[i].Distance <= h.Nodes[p].Distance {
			break
		}
		h.Nodes[i], h.Nodes[p] = h.Nodes[p], h.Nodes[i]
		i = p
	}
}

func (h *NodeMaxHeap) Pop() Node {
	n := h.Nodes[0]
	last := len(h.Nodes) - 1
	h.Nodes[0] = h.Nodes[last]
	h.Nodes = h.Nodes[:last]
	i := 0
	for {
		l := 2*i + 1
		r := 2*i + 2
		largest := i
		if l < len(h.Nodes) && h.Nodes[l].Distance > h.Nodes[largest].Distance {
			largest = l
		}
		if r < len(h.Nodes) && h.Nodes[r].Distance > h.Nodes[largest].Distance {
			largest = r
		}
		if largest == i {
			break
		}
		h.Nodes[i], h.Nodes[largest] = h.Nodes[largest], h.Nodes[i]
		i = largest
	}
	return n
}

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

type searchBufferSlot struct {
	buf atomic.Pointer[searchBuffer]
}
