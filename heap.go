package hnsw

// Node is a search result or candidate in the HNSW graph.
type Node struct {
	ID       uint32
	Distance float32
	Metadata []byte
}

// nodeHeap is a min-heap of Node ordered by Distance.
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

// NodeMaxHeap is a max-heap of Node ordered by Distance (used for ef-limited result sets).
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
