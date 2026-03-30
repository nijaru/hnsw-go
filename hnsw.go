package hnsw

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"sync"
)

type Node struct {
	ID       uint32
	Distance float32
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

type maxNodeHeap struct {
	Nodes []Node
}

func (h *maxNodeHeap) Push(n Node) {
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

func (h *maxNodeHeap) Pop() Node {
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
	results    maxNodeHeap
	candidates nodeHeap
	out        []Node
	gen        uint8
}

func (b *searchBuffer) isVisited(id uint32) bool {
	if int(id) >= len(b.visited) {
		return false
	}
	return b.visited[id] == b.gen
}

func (b *searchBuffer) visit(id uint32) {
	if int(id) >= len(b.visited) {
		return
	}
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

type Index struct {
	storage    *Storage
	distFunc   DistanceFunc
	mu         sync.RWMutex
	entryPoint uint32
	maxLevel   int
	nodeCount  uint32
	m          int
	mMax0      int
	efSearch   int
	efConst    int
	pool       sync.Pool
}

func NewIndex(storage *Storage, distFunc DistanceFunc, efSearch, efConst int) *Index {
	m := int(storage.config.M)
	mMax0 := int(storage.config.MMax0)
	if efSearch <= 0 {
		efSearch = 16
	}
	if efConst <= 0 {
		efConst = 200
	}
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
			visited:    make([]uint8, 16384),
			results:    maxNodeHeap{Nodes: make([]Node, 0, efSearch+mMax0+1)},
			candidates: nodeHeap{Nodes: make([]Node, 0, efSearch+mMax0+1)},
			out:        make([]Node, 0, efSearch+1),
			gen:        1,
		}
	}

	idx.maxLevel = int(storage.readUint32(36))
	idx.entryPoint = storage.readUint32(24)
	idx.nodeCount = storage.readUint32(28)

	return idx
}

func (idx *Index) SetEfSearch(ef int) {
	idx.mu.Lock()
	idx.efSearch = ef
	idx.mu.Unlock()
}

func (idx *Index) SetEfConst(ef int) {
	idx.mu.Lock()
	idx.efConst = ef
	idx.mu.Unlock()
}

func (idx *Index) Len() int {
	idx.mu.RLock()
	n := idx.nodeCount
	idx.mu.RUnlock()
	return int(n)
}

type Stats struct {
	NodeCount  uint32
	MaxLevel   int
	EntryPoint uint32
	M          int
	MMax0      int
	EfSearch   int
	EfConst    int
}

func (idx *Index) Stats() Stats {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return Stats{
		NodeCount:  idx.nodeCount,
		MaxLevel:   idx.maxLevel,
		EntryPoint: idx.entryPoint,
		M:          idx.m,
		MMax0:      idx.mMax0,
		EfSearch:   idx.efSearch,
		EfConst:    idx.efConst,
	}
}

func (idx *Index) Search(query []float32, k int) ([]Node, error) {
	if len(query) != int(idx.storage.config.Dims) {
		return nil, fmt.Errorf(
			"hnsw: query dims %d != index dims %d",
			len(query),
			idx.storage.config.Dims,
		)
	}
	if k <= 0 {
		return nil, nil
	}

	idx.mu.RLock()
	defer idx.mu.RUnlock()
	idx.storage.ReadLock()
	defer idx.storage.ReadUnlock()

	if idx.nodeCount == 0 {
		return nil, nil
	}

	currMaxLevel := idx.maxLevel
	currEntryPoint := idx.entryPoint
	nodeCount := idx.nodeCount
	efSearch := idx.efSearch

	currNode := currEntryPoint
	currDist := idx.distFunc(query, idx.storage.GetVector(currNode))

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

	buf := idx.pool.Get().(*searchBuffer)
	defer idx.pool.Put(buf)
	buf.reset(nodeCount)

	buf.candidates.Push(Node{ID: currNode, Distance: currDist})
	buf.results.Push(Node{ID: currNode, Distance: currDist})
	buf.visit(currNode)

	for len(buf.candidates.Nodes) > 0 {
		c := buf.candidates.Pop()
		f := buf.results.Nodes[0]

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

	for len(buf.results.Nodes) > 0 {
		buf.out = append(buf.out, buf.results.Pop())
	}
	slices.Reverse(buf.out)

	actualK := min(k, len(buf.out))
	res := make([]Node, actualK)
	copy(res, buf.out[:actualK])
	return res, nil
}

func (idx *Index) Insert(vec []float32) error {
	if len(vec) != int(idx.storage.config.Dims) {
		return fmt.Errorf(
			"hnsw: vector dims %d != index dims %d",
			len(vec),
			idx.storage.config.Dims,
		)
	}

	level := idx.randomLevel()

	idx.mu.Lock()
	id, err := idx.storage.addNode()
	if err != nil {
		idx.mu.Unlock()
		return err
	}
	idx.nodeCount = id + 1

	if id == 0 {
		idx.entryPoint = id
		idx.maxLevel = level
		idx.storage.writeUint32(24, id)
		idx.storage.writeUint32(36, uint32(level))
		idx.storage.setLevel(id, level)
		idx.storage.setVector(id, vec)
		idx.mu.Unlock()
		return nil
	}

	currEntryPoint := idx.entryPoint
	currMaxLevel := idx.maxLevel
	efConst := idx.efConst
	nodeCount := idx.nodeCount
	idx.storage.ReadLock()
	idx.mu.Unlock()

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

	idx.storage.setLevel(id, level)
	idx.storage.setVector(id, vec)

	insertBuf := idx.pool.Get().(*searchBuffer)
	defer idx.pool.Put(insertBuf)

	for l := min(level, currMaxLevel); l >= 0; l-- {
		candidates := idx.findNeighborsAtLayer(vec, currNode, l, efConst, nodeCount, insertBuf)

		limit := idx.m
		if l == 0 {
			limit = idx.mMax0
		}

		selected := idx.selectNeighbors(candidates, limit)
		idx.storage.SetNeighbors(id, l, selected)

		var linkBuf [64]uint32
		for _, nb := range selected {
			idx.storage.LockNode(nb)
			nbList := idx.storage.GetNeighbors(nb, l)

			n := len(nbList)
			var tmp []uint32
			if n+1 <= len(linkBuf) {
				tmp = linkBuf[:n+1]
			} else {
				tmp = make([]uint32, n+1)
			}
			copy(tmp, nbList)
			tmp[n] = id

			if len(tmp) > limit {
				nbVec := idx.storage.GetVector(nb)
				tmp = idx.shrinkNeighbors(tmp, nbVec, limit)
			}
			idx.storage.SetNeighbors(nb, l, tmp)
			idx.storage.UnlockNode(nb)
		}
	}

	idx.storage.ReadUnlock()

	if level > currMaxLevel {
		idx.mu.Lock()
		if level > idx.maxLevel {
			idx.maxLevel = level
			idx.entryPoint = id
			idx.storage.writeUint32(36, uint32(level))
			idx.storage.writeUint32(24, id)
		}
		idx.mu.Unlock()
	}

	return nil
}

func (idx *Index) randomLevel() int {
	mL := 1.0 / math.Log(float64(idx.m))
	level := int(-math.Log(1.0-rand.Float64()) * mL)
	if level < 0 {
		level = 0
	}
	if level >= int(idx.storage.config.MaxLevel) {
		level = int(idx.storage.config.MaxLevel) - 1
	}
	return level
}

func (idx *Index) findNeighborsAtLayer(
	vec []float32,
	entry uint32,
	layer, ef int,
	nodeCount uint32,
	buf *searchBuffer,
) []Node {
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

	buf.out = buf.out[:0]
	for len(buf.results.Nodes) > 0 {
		buf.out = append(buf.out, buf.results.Pop())
	}
	slices.Reverse(buf.out)
	return buf.out
}

func (idx *Index) selectNeighbors(candidates []Node, m int) []uint32 {
	if len(candidates) <= m {
		var resBuf [64]uint32
		res := resBuf[:len(candidates)]
		for i, c := range candidates {
			res[i] = c.ID
		}
		return res
	}

	var resultBuf [64]Node
	var result []Node
	if m <= len(resultBuf) {
		result = resultBuf[:0]
	} else {
		result = make([]Node, 0, m)
	}

	var discardBuf [64]Node
	discarded := discardBuf[:0]

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
		} else {
			discarded = append(discarded, c)
		}
	}

	for _, d := range discarded {
		if len(result) >= m {
			break
		}
		result = append(result, d)
	}

	var resBuf [64]uint32
	res := resBuf[:len(result)]
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

	h := nodeHeap{Nodes: heapNodes}
	for _, n := range nodes {
		h.Push(n)
	}

	sorted := nodes[:0]
	for len(h.Nodes) > 0 {
		sorted = append(sorted, h.Pop())
	}

	return idx.selectNeighbors(sorted, limit)
}
