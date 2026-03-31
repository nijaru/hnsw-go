package hnsw

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"sync"
)

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
	probes     int
	pool       sync.Pool
}

func NewIndex(storage *Storage, distFunc DistanceFunc, efSearch, efConst int) *Index {
	m := int(storage.config.M)
	mMax0 := int(storage.config.MMax0)
	probes := int(storage.config.Probes)
	if probes <= 0 {
		probes = 1
	}
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
		probes:     probes,
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

func (idx *Index) SetProbes(probes int) {
	idx.mu.Lock()
	if probes <= 0 {
		probes = 1
	}
	idx.probes = probes
	idx.mu.Unlock()
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
	Probes     int
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
		Probes:     idx.probes,
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

	if idx.nodeCount == 0 {
		return nil, nil
	}

	currMaxLevel := idx.maxLevel
	currEntryPoint := idx.entryPoint
	nodeCount := idx.nodeCount
	efSearch := idx.efSearch
	probes := idx.probes

	var probeBuf [64]Node
	probeNodes := probeBuf[:0]
	currDist := idx.distFunc(query, idx.storage.GetVector(currEntryPoint))
	probeNodes = append(probeNodes, Node{ID: currEntryPoint, Distance: currDist})

	for level := currMaxLevel; level >= 1; level-- {
		changed := true
		for changed {
			changed = false
			// Greedy move from the best probe
			best := probeNodes[0]
			neighbors := idx.storage.GetNeighbors(best.ID, level)
			for _, nb := range neighbors {
				d := idx.distFunc(query, idx.storage.GetVector(nb))
				if d < best.Distance {
					best.Distance = d
					best.ID = nb
					changed = true
				}
			}
			if changed {
				probeNodes[0] = best
			}
		}

		if probes > 1 {
			// Expand from the best node to find more entry points for the next layer
			best := probeNodes[0]
			neighbors := idx.storage.GetNeighbors(best.ID, level)
			for _, nb := range neighbors {
				d := idx.distFunc(query, idx.storage.GetVector(nb))
				
				// Check if already in probes
				exists := false
				for _, p := range probeNodes {
					if p.ID == nb {
						exists = true
						break
					}
				}
				if exists {
					continue
				}

				if len(probeNodes) < probes {
					probeNodes = append(probeNodes, Node{ID: nb, Distance: d})
					// Sort by distance (simple insertion sort for small N)
					for i := len(probeNodes) - 1; i > 0 && probeNodes[i].Distance < probeNodes[i-1].Distance; i-- {
						probeNodes[i], probeNodes[i-1] = probeNodes[i-1], probeNodes[i]
					}
				} else if d < probeNodes[len(probeNodes)-1].Distance {
					probeNodes[len(probeNodes)-1] = Node{ID: nb, Distance: d}
					for i := len(probeNodes) - 1; i > 0 && probeNodes[i].Distance < probeNodes[i-1].Distance; i-- {
						probeNodes[i], probeNodes[i-1] = probeNodes[i-1], probeNodes[i]
					}
				}
			}
		}
	}

	buf := idx.pool.Get().(*searchBuffer)
	defer idx.pool.Put(buf)
	buf.reset(nodeCount)

	for _, p := range probeNodes {
		buf.candidates.Push(p)
		buf.results.Push(p)
		if len(buf.results.Nodes) > efSearch {
			buf.results.Pop()
		}
		buf.visit(p.ID)
	}

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
		n := buf.results.Pop()
		if !idx.storage.IsDeleted(n.ID) {
			buf.out = append(buf.out, n)
		}
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
	return idx.BatchInsert([][]float32{vec})
}

func (idx *Index) BatchInsert(vecs [][]float32) error {
	if len(vecs) == 0 {
		return nil
	}

	dims := int(idx.storage.config.Dims)
	for i, vec := range vecs {
		if len(vec) != dims {
			return fmt.Errorf("hnsw: vector %d dims %d != index dims %d", i, len(vec), dims)
		}
	}

	// 1. Pre-allocate all IDs and grow storage once
	idx.mu.Lock()
	startID := idx.nodeCount
	numVecs := uint32(len(vecs))
	
	// Ensure storage has enough capacity
	allocated := idx.storage.readUint32(32)
	if startID+numVecs > allocated {
		newAllocated := max(allocated*2, startID+numVecs+1000)
		if err := idx.storage.grow(newAllocated); err != nil {
			idx.mu.Unlock()
			return err
		}
	}
	
	idx.nodeCount += numVecs
	idx.storage.writeUint32(28, idx.nodeCount)
	idx.mu.Unlock()

	// 2. Perform inserts in parallel
	numWorkers := min(len(vecs), 16) // Heuristic
	jobs := make(chan int, len(vecs))
	errs := make(chan error, numWorkers)
	var wg sync.WaitGroup

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				id := startID + uint32(i)
				if err := idx.insert(id, vecs[i]); err != nil {
					errs <- err
					return
				}
			}
		}()
	}

	for i := range vecs {
		jobs <- i
	}
	close(jobs)
	wg.Wait()

	select {
	case err := <-errs:
		return err
	default:
	}

	return nil
}

func (idx *Index) insert(id uint32, vec []float32) error {
	level := idx.randomLevel()

	if id == 0 {
		idx.mu.Lock()
		idx.entryPoint = id
		idx.maxLevel = level
		idx.storage.writeUint32(24, id)
		idx.storage.writeUint32(36, uint32(level))
		idx.storage.setLevel(id, level)
		if err := idx.storage.allocateUpper(id, level); err != nil {
			idx.mu.Unlock()
			return err
		}
		idx.storage.setVector(id, vec)
		idx.mu.Unlock()
		return nil
	}

	idx.storage.setLevel(id, level)
	if err := idx.storage.allocateUpper(id, level); err != nil {
		return err
	}
	idx.storage.setVector(id, vec)

	idx.mu.RLock()
	currEntryPoint := idx.entryPoint
	currMaxLevel := idx.maxLevel
	efConst := idx.efConst
	nodeCount := idx.nodeCount

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
	idx.mu.RUnlock()

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

func (idx *Index) Delete(id uint32) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	if id >= idx.nodeCount {
		return fmt.Errorf("hnsw: delete id %d out of bounds", id)
	}
	if idx.storage.IsDeleted(id) {
		return nil
	}

	// Heal graph connectivity before marking as deleted.
	// We do this by connecting neighbors of the deleted node to each other.
	idx.repairNeighborConnection(id)

	idx.storage.SetDeleted(id, true)
	return nil
}

func (idx *Index) repairNeighborConnection(id uint32) {
	maxL := idx.storage.GetMaxLevel(id)
	for level := 0; level <= maxL; level++ {
		neighbors := idx.storage.GetNeighbors(id, level)
		if len(neighbors) == 0 {
			continue
		}

		// N2 is the set of points that have 'id' as a neighbor
		var n2 []uint32
		for _, v := range neighbors {
			vNb := idx.storage.GetNeighbors(v, level)
			hasID := false
			for _, nbID := range vNb {
				if nbID == id {
					hasID = true
					break
				}
			}
			if hasID {
				n2 = append(n2, v)
			}
		}

		// For each u in N2, re-select neighbors from {u's neighbors} + {id's neighbors} - {id}
		for _, u := range n2 {
			uNb := idx.storage.GetNeighbors(u, level)
			
			// Build candidate set for u
			candidates := make([]Node, 0, len(uNb)+len(neighbors))
			uVec := idx.storage.GetVector(u)
			
			seen := make(map[uint32]bool)
			seen[id] = true // Exclude the deleted node
			seen[u] = true  // Exclude self

			// Add u's current neighbors (except id)
			for _, nbID := range uNb {
				if !seen[nbID] {
					d := idx.distFunc(uVec, idx.storage.GetVector(nbID))
					candidates = append(candidates, Node{ID: nbID, Distance: d})
					seen[nbID] = true
				}
			}
			// Add id's neighbors
			for _, nbID := range neighbors {
				if !seen[nbID] {
					d := idx.distFunc(uVec, idx.storage.GetVector(nbID))
					candidates = append(candidates, Node{ID: nbID, Distance: d})
					seen[nbID] = true
				}
			}

			// Prune using the standard heuristic
			limit := idx.storage.config.M
			if level == 0 {
				limit = idx.storage.config.MMax0
			}
			
			// selectNeighbors expects candidates sorted by distance
			slices.SortFunc(candidates, func(a, b Node) int {
				if a.Distance < b.Distance {
					return -1
				}
				if a.Distance > b.Distance {
					return 1
				}
				return 0
			})

			newNb := idx.selectNeighbors(candidates, int(limit))
			idx.storage.SetNeighbors(u, level, newNb)
		}
	}
}

func (idx *Index) Sync() error {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.storage.Sync()
}

func (idx *Index) Close() error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	return idx.storage.Close()
}
