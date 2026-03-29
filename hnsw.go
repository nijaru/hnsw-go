package hnsw

import (
	"encoding/binary"
	"math/rand/v2"
	"slices"
	"sync"
	"unsafe"
)

/*
Safety & Performance Mandate (TigerStyle):

1. Zero-Allocation (GC Immunity):
   The Search and findNeighborsAtLayer paths must maintain 0-1 heap allocations.
   All search-related structures (visited lists, priority queues) are reused
   from a sync.Pool.

2. Zero-Copy (Mechanical Sympathy):
   All graph traversal occurs via unsafe.Slice views of the mmap'd backing slice.
   This avoids redundant heap allocations and expensive memcpy operations.

3. Pointers & Remapping:
   The Index.mu RWMutex MUST be held for the duration of any traversal to ensure
   active mmap slices (s.data) are not unmapped/remapped by a concurrent Grow().

4. Alignment:
   All node offsets are aligned to 64-byte boundaries (HeaderSize + NodeID * NodeSize).
   This ensures node metadata and the first few vector elements fit in a single cache line.
*/

// Node represents a point in the HNSW graph.
type Node struct {
	ID       uint32
	Distance float32
}

// NodeHeap is a min-heap of nodes (closest first).
type NodeHeap struct {
	Nodes []Node
}

func (h *NodeHeap) Push(n Node) {
	h.Nodes = append(h.Nodes, n)
	// sift up
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

func (h *NodeHeap) Pop() Node {
	n := h.Nodes[0]
	last := len(h.Nodes) - 1
	h.Nodes[0] = h.Nodes[last]
	h.Nodes = h.Nodes[:last]
	// sift down
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

// MaxNodeHeap is a max-heap of nodes (furthest first).
type MaxNodeHeap struct {
	Nodes []Node
}

func (h *MaxNodeHeap) Push(n Node) {
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

func (h *MaxNodeHeap) Pop() Node {
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
	visited    []uint32 // generation-based bitset or dense array
	results    MaxNodeHeap
	candidates NodeHeap
	out        []Node // reusable output buffer
	gen        uint32
}

func (b *searchBuffer) isVisited(id uint32) bool {
	if int(id) >= len(b.visited) {
		return false
	}
	return b.visited[id] == b.gen
}

func (b *searchBuffer) visit(id uint32) {
	if int(id) >= len(b.visited) {
		// grow visited if necessary
		newCap := len(b.visited) * 2
		if newCap <= int(id) {
			newCap = int(id) + 1024
		}
		newVisited := make([]uint32, newCap)
		copy(newVisited, b.visited)
		b.visited = newVisited
	}
	b.visited[id] = b.gen
}

func (b *searchBuffer) reset(maxNodes uint32) {
	b.gen++
	if b.gen == 0 { // overflow
		for i := range b.visited {
			b.visited[i] = 0
		}
		b.gen = 1
	}
	if uint32(len(b.visited)) < maxNodes {
		b.visited = make([]uint32, maxNodes+2048)
		b.gen = 1
	}
	b.results.Nodes = b.results.Nodes[:0]
	b.candidates.Nodes = b.candidates.Nodes[:0]
	b.out = b.out[:0]
}

// Index represents the HNSW index.
type Index struct {
	storage    *Storage
	distFunc   DistanceFunc
	mu         sync.RWMutex
	entryPoint uint32
	maxLevel   int
	nodeCount  uint32

	// Config
	m        int
	mMax0    int
	efSearch int
	efConst  int

	// Pool for zero-allocation search buffers
	pool sync.Pool
}

// NewIndex creates a new HNSW index.
func NewIndex(storage *Storage, distFunc DistanceFunc, m, efSearch, efConst int) *Index {
	mMax0 := m * 2
	idx := &Index{
		storage:    storage,
		distFunc:   distFunc,
		m:          m,
		mMax0:      mMax0,
		efSearch:   efSearch,
		efConst:    efConst,
		maxLevel:   -1,
		entryPoint: 0,
	}

	idx.pool.New = func() any {
		return &searchBuffer{
			visited:    make([]uint32, 16384), // Start with a decent size
			results:    MaxNodeHeap{Nodes: make([]Node, 0, efSearch+mMax0+1)},
			candidates: NodeHeap{Nodes: make([]Node, 0, efSearch+mMax0+1)},
			out:        make([]Node, 0, efSearch+1),
			gen:        1,
		}
	}

	// Load global state from storage header
	idx.maxLevel = int(storage.readUint32(20))
	idx.entryPoint = storage.readUint32(24)
	idx.nodeCount = storage.readUint32(28)

	return idx
}

// SetEfSearch updates the search effort parameter.
func (idx *Index) SetEfSearch(ef int) {
	idx.mu.Lock()
	idx.efSearch = ef
	idx.mu.Unlock()
}

// SetEfConst updates the construction effort parameter.
func (idx *Index) SetEfConst(ef int) {
	idx.mu.Lock()
	idx.efConst = ef
	idx.mu.Unlock()
}

// Search returns the top K nearest neighbors for the query vector.
func (idx *Index) Search(query []float32, k int) []Node {
	idx.mu.RLock()
	currMaxLevel := idx.maxLevel
	currEntryPoint := idx.entryPoint
	nodeCount := idx.nodeCount
	efSearch := idx.efSearch
	idx.mu.RUnlock()

	if nodeCount == 0 {
		return nil
	}

	currNode := currEntryPoint
	currDist := idx.distFunc(query, idx.storage.GetVector(currNode))

	// 1. Greedy search through upper layers
	for level := currMaxLevel; level >= 1; level-- {
		changed := true
		for changed {
			changed = false
			neighbors := idx.storage.GetNeighbors(currNode, level)
			for _, nb := range neighbors {
				d := idx.distFunc(query, idx.storage.GetVector(nb))
				if d < currDist {
					currDist = d
					currNode = nb
					changed = true
				}
			}
		}
	}

	// 2. Search at layer 0 with efSearch
	buf := idx.pool.Get().(*searchBuffer)
	defer idx.pool.Put(buf)
	buf.reset(nodeCount)

	buf.candidates.Push(Node{ID: currNode, Distance: currDist})
	buf.results.Push(Node{ID: currNode, Distance: currDist})
	buf.visit(currNode)

	for len(buf.candidates.Nodes) > 0 {
		c := buf.candidates.Pop()
		f := buf.results.Nodes[0] // furthest in results

		if c.Distance > f.Distance && len(buf.results.Nodes) >= efSearch {
			break
		}

		neighbors := idx.storage.GetNeighbors(c.ID, 0)
		for _, nb := range neighbors {
			if buf.isVisited(nb) {
				continue
			}
			buf.visit(nb)

			d := idx.distFunc(query, idx.storage.GetVector(nb))
			if len(buf.results.Nodes) < efSearch || d < f.Distance {
				buf.candidates.Push(Node{ID: nb, Distance: d})
				buf.results.Push(Node{ID: nb, Distance: d})
				if len(buf.results.Nodes) > efSearch {
					buf.results.Pop()
				}
				f = buf.results.Nodes[0]
			}
		}
	}

	// Return top K
	for len(buf.results.Nodes) > 0 {
		buf.out = append(buf.out, buf.results.Pop())
	}
	// Reverse to get closest first
	slices.Reverse(buf.out)

	actualK := min(k, len(buf.out))
	res := make([]Node, actualK)
	copy(res, buf.out[:actualK])
	return res
}

// Insert adds a new vector to the index.
func (idx *Index) Insert(vec []float32) error {
	level := idx.randomLevel()

	idx.mu.Lock()
	id, err := idx.storage.AddNode()
	if err != nil {
		idx.mu.Unlock()
		return err
	}
	idx.nodeCount = id + 1

	if id == 0 {
		// First node
		idx.entryPoint = id
		idx.maxLevel = level
		idx.storage.writeUint32(24, id)
		idx.storage.writeUint32(20, uint32(level))
		idx.storage.SetLevel(id, level)
		idx.storage.SetVector(id, vec)
		idx.mu.Unlock()
		return nil
	}

	idx.mu.RLock()
	currEntryPoint := idx.entryPoint
	currMaxLevel := idx.maxLevel
	efConst := idx.efConst
	idx.mu.RUnlock()
	idx.mu.RLock() // Protects mmap from Grow() during traversal

	// Initial greedy search to the insertion level
	currNode := currEntryPoint
	currDist := idx.distFunc(vec, idx.storage.GetVector(currNode))

	for l := currMaxLevel; l > level; l-- {
		changed := true
		for changed {
			changed = false
			neighbors := idx.storage.GetNeighbors(currNode, l)
			for _, nb := range neighbors {
				d := idx.distFunc(vec, idx.storage.GetVector(nb))
				if d < currDist {
					currDist = d
					currNode = nb
					changed = true
				}
			}
		}
	}

	idx.storage.SetLevel(id, level)
	idx.storage.SetVector(id, vec)

	for l := min(level, currMaxLevel); l >= 0; l-- {
		candidates := idx.findNeighborsAtLayer(vec, currNode, l, efConst)

		limit := idx.m
		if l == 0 {
			limit = idx.mMax0
		}

		selected := idx.selectNeighbors(candidates, limit)
		idx.storage.SetNeighbors(id, l, selected)

		// Bi-directional link
		for _, nb := range selected {
			idx.storage.LockNode(nb)
			nbList := idx.storage.GetNeighbors(nb, l)

			// Copy nbList and add id
			newNbList := make([]uint32, len(nbList), len(nbList)+1)
			copy(newNbList, nbList)
			newNbList = append(newNbList, id)

			if len(newNbList) > limit {
				newNbList = idx.shrinkNeighbors(newNbList, vec, limit)
			}
			idx.storage.SetNeighbors(nb, l, newNbList)
			idx.storage.UnlockNode(nb)
		}
	}

	// Update global max level if needed
	idx.mu.RUnlock()
	if level > currMaxLevel {
		idx.mu.Lock()
		if level > idx.maxLevel {
			idx.maxLevel = level
			idx.entryPoint = id
			idx.storage.writeUint32(20, uint32(level))
			idx.storage.writeUint32(24, id)
		}
		idx.mu.Unlock()
	}

	return nil
}

func (idx *Index) randomLevel() int {
	l := 0
	for rand.Float64() < 0.5 && l < int(idx.storage.config.MaxLevel) {
		l++
	}
	return l
}

func (idx *Index) findNeighborsAtLayer(vec []float32, entry uint32, layer, ef int) []Node {
	idx.mu.RLock()
	nodeCount := idx.nodeCount
	idx.mu.RUnlock()

	buf := idx.pool.Get().(*searchBuffer)
	defer idx.pool.Put(buf)
	buf.reset(nodeCount)

	dist := idx.distFunc(vec, idx.storage.GetVector(entry))
	buf.candidates.Push(Node{ID: entry, Distance: dist})
	buf.results.Push(Node{ID: entry, Distance: dist})
	buf.visit(entry)

	for len(buf.candidates.Nodes) > 0 {
		c := buf.candidates.Pop()
		f := buf.results.Nodes[0]

		if c.Distance > f.Distance && len(buf.results.Nodes) >= ef {
			break
		}

		neighbors := idx.storage.GetNeighbors(c.ID, layer)
		for _, nb := range neighbors {
			if buf.isVisited(nb) {
				continue
			}
			buf.visit(nb)

			d := idx.distFunc(vec, idx.storage.GetVector(nb))
			if len(buf.results.Nodes) < ef || d < f.Distance {
				buf.candidates.Push(Node{ID: nb, Distance: d})
				buf.results.Push(Node{ID: nb, Distance: d})
				if len(buf.results.Nodes) > ef {
					buf.results.Pop()
				}
				f = buf.results.Nodes[0]
			}
		}
	}

	for len(buf.results.Nodes) > 0 {
		buf.out = append(buf.out, buf.results.Pop())
	}
	// Reverse to get closest first
	slices.Reverse(buf.out)

	res := make([]Node, len(buf.out))
	copy(res, buf.out)
	return res
}

func (idx *Index) selectNeighbors(candidates []Node, m int) []uint32 {
	if len(candidates) <= m {
		res := make([]uint32, len(candidates))
		for i, c := range candidates {
			res[i] = c.ID
		}
		return res
	}

	// Use a small fixed buffer for diversity check to avoid allocations for small M
	var resultBuf [64]Node
	var result []Node
	if m <= len(resultBuf) {
		result = resultBuf[:0]
	} else {
		result = make([]Node, 0, m)
	}

	for _, c := range candidates {
		if len(result) >= m {
			break
		}

		isDiverse := true
		for _, r := range result {
			distToResult := idx.distFunc(idx.storage.GetVector(c.ID), idx.storage.GetVector(r.ID))
			if distToResult < c.Distance {
				isDiverse = false
				break
			}
		}

		if isDiverse {
			result = append(result, c)
		}
	}

	res := make([]uint32, len(result))
	for i, r := range result {
		res[i] = r.ID
	}
	return res
}

func (idx *Index) shrinkNeighbors(neighbors []uint32, vec []float32, limit int) []uint32 {
	if len(neighbors) <= limit {
		return neighbors
	}

	var nodesBuf [128]Node
	var nodes []Node
	if len(neighbors) <= len(nodesBuf) {
		nodes = nodesBuf[:0]
	} else {
		nodes = make([]Node, 0, len(neighbors))
	}

	for _, nb := range neighbors {
		nodes = append(nodes, Node{ID: nb, Distance: idx.distFunc(vec, idx.storage.GetVector(nb))})
	}

	var heapBuf [128]Node
	var heapNodes []Node
	if len(nodes) <= len(heapBuf) {
		heapNodes = heapBuf[:0]
	} else {
		heapNodes = make([]Node, 0, len(nodes))
	}

	h := NodeHeap{Nodes: heapNodes}
	for _, n := range nodes {
		h.Push(n)
	}

	sorted := nodes[:0]
	for len(h.Nodes) > 0 {
		sorted = append(sorted, h.Pop())
	}

	return idx.selectNeighbors(sorted, limit)
}

func (s *Storage) SetLevel(id uint32, level int) {
	data := s.GetNodeData(id)
	binary.LittleEndian.PutUint32(data[s.layout.LevelOffset:s.layout.LevelOffset+4], uint32(level))
}

func (s *Storage) SetVector(id uint32, vec []float32) {
	data := s.GetNodeData(id)
	vecData := data[s.layout.VectorOffset : s.layout.VectorOffset+(s.config.Dims*4)]
	copy(unsafe.Slice((*float32)(unsafe.Pointer(&vecData[0])), s.config.Dims), vec)
}
