package segment

import (
	"fmt"
	"slices"
	"sync"
	"sync/atomic"

	"github.com/omendb/hnsw-go"
)

type segmentMergeBuffer struct {
	seen     []uint8
	best     []hnsw.Node
	results  hnsw.NodeMaxHeap
	out      []hnsw.Node
	local    []hnsw.Node
	localIDs []uint32
	gen      uint8
}

func newSegmentMergeBuffer(globalCap, localCap, outCap int) *segmentMergeBuffer {
	if globalCap < 1 {
		globalCap = 1
	}
	if localCap < 1 {
		localCap = 1
	}
	if outCap < 1 {
		outCap = 1
	}

	return &segmentMergeBuffer{
		seen:     make([]uint8, globalCap),
		best:     make([]hnsw.Node, globalCap),
		results:  hnsw.NodeMaxHeap{Nodes: make([]hnsw.Node, 0, outCap)},
		out:      make([]hnsw.Node, 0, outCap),
		local:    make([]hnsw.Node, 0, localCap),
		localIDs: make([]uint32, 0, localCap),
		gen:      1,
	}
}

func (b *segmentMergeBuffer) reset(globalCap, localCap, outCap int) {
	b.gen++
	if b.gen == 0 {
		for i := range b.seen {
			b.seen[i] = 0
		}
		b.gen = 1
	}
	if globalCap > len(b.seen) {
		b.seen = make([]uint8, globalCap+2048)
		b.best = make([]hnsw.Node, len(b.seen))
		b.gen = 1
	}
	if localCap < 1 {
		localCap = 1
	}
	if cap(b.local) < localCap {
		b.local = make([]hnsw.Node, 0, localCap)
	} else {
		b.local = b.local[:0]
	}
	if cap(b.localIDs) < localCap {
		b.localIDs = make([]uint32, 0, localCap)
	} else {
		b.localIDs = b.localIDs[:0]
	}
	if cap(b.results.Nodes) < outCap {
		b.results.Nodes = make([]hnsw.Node, 0, outCap)
	} else {
		b.results.Nodes = b.results.Nodes[:0]
	}
	if cap(b.out) < outCap {
		b.out = make([]hnsw.Node, 0, outCap)
	} else {
		b.out = b.out[:0]
	}
}

func (b *segmentMergeBuffer) clearBest() {
	for i := range b.seen {
		b.seen[i] = 0
	}
}

type segmentMergeBufferSlot struct {
	buf atomic.Pointer[segmentMergeBuffer]
}

type SegmentedIndex struct {
	catalog *segmentCatalog
	pool    sync.Pool
	scratch segmentMergeBufferSlot
}

func NewSegmentedIndex() *SegmentedIndex {
	idx := &SegmentedIndex{
		catalog: newSegmentCatalog(),
	}
	idx.pool.New = func() any {
		return newSegmentMergeBuffer(1, 1, 1)
	}
	idx.scratch.buf.Store(newSegmentMergeBuffer(1, 1, 1))
	return idx
}

func NewSegmentedIndexFrom(head *hnsw.Index, frozen ...*hnsw.Index) (*SegmentedIndex, error) {
	idx := NewSegmentedIndex()
	if err := idx.Publish(head, frozen...); err != nil {
		return nil, err
	}
	return idx, nil
}

func (idx *SegmentedIndex) Publish(head *hnsw.Index, frozen ...*hnsw.Index) error {
	headBinding, frozenBindings, err := buildSegmentBindings(head, frozen...)
	if err != nil {
		return err
	}

	_, err = idx.catalog.publish(headBinding, frozenBindings...)
	return err
}

func (idx *SegmentedIndex) Search(query []float32, k int) ([]hnsw.Node, error) {
	return idx.searchInto(nil, query, k, nil, false)
}

func (idx *SegmentedIndex) SearchInto(
	dst []hnsw.Node,
	query []float32,
	k int,
) ([]hnsw.Node, error) {
	return idx.searchInto(dst, query, k, nil, false)
}

func (idx *SegmentedIndex) SearchAllowed(
	query []float32,
	k int,
	allow hnsw.AllowList,
) ([]hnsw.Node, error) {
	return idx.searchInto(nil, query, k, &allow, false)
}

func (idx *SegmentedIndex) SearchAllowedInto(
	dst []hnsw.Node,
	query []float32,
	k int,
	allow hnsw.AllowList,
) ([]hnsw.Node, error) {
	return idx.searchInto(dst, query, k, &allow, false)
}

func (idx *SegmentedIndex) SearchPlanned(
	query []float32,
	k int,
	allow hnsw.AllowList,
) ([]hnsw.Node, error) {
	return idx.searchInto(nil, query, k, &allow, true)
}

func (idx *SegmentedIndex) SearchPlannedInto(
	dst []hnsw.Node,
	query []float32,
	k int,
	allow hnsw.AllowList,
) ([]hnsw.Node, error) {
	return idx.searchInto(dst, query, k, &allow, true)
}

func (idx *SegmentedIndex) searchInto(
	dst []hnsw.Node,
	query []float32,
	k int,
	allow *hnsw.AllowList,
	planned bool,
) ([]hnsw.Node, error) {
	if k <= 0 {
		return nil, nil
	}

	view := idx.catalog.load()
	if view == nil || view.segmentCount() == 0 {
		return nil, nil
	}

	buf := idx.acquireMergeBuffer()
	defer idx.releaseMergeBuffer(buf)
	buf.reset(len(view.locations), k, k)

	for i := 0; i < view.segmentCount(); i++ {
		seg := view.segmentAt(i)
		if seg == nil || seg.index == nil {
			continue
		}

		var local []hnsw.Node
		var err error

		if allow != nil {
			buf.localIDs = buf.localIDs[:0]
			if allow.Len() < int(seg.localCount()) {
				allow.ForEach(func(global uint32) bool {
					canonSeg, canonLocal, ok := view.resolve(global)
					if ok && canonSeg == uint32(i) {
						buf.localIDs = append(buf.localIDs, canonLocal)
					}
					return true
				})
				slices.Sort(buf.localIDs)
			} else {
				for localIdx, global := range seg.localToGlobal {
					if global != invalidGlobalID && allow.Contains(global) {
						canonSeg, canonLocal, ok := view.resolve(global)
						if ok && canonSeg == uint32(i) && canonLocal == uint32(localIdx) {
							buf.localIDs = append(buf.localIDs, uint32(localIdx))
						}
					}
				}
			}

			localAllow := hnsw.NewAllowIDsSorted(buf.localIDs)
			if planned {
				local, err = seg.index.SearchPlannedInto(buf.local[:0], query, k, localAllow)
			} else {
				local, err = seg.index.SearchAllowedInto(buf.local[:0], query, k, localAllow)
			}
		} else {
			local, err = seg.index.SearchInto(buf.local[:0], query, k)
		}

		if err != nil {
			return nil, err
		}
		buf.local = local

		for _, n := range local {
			global, ok := seg.globalID(n.ID)
			if !ok || int(global) >= len(buf.best) {
				continue
			}

			canonSeg, canonLocal, ok := view.resolve(global)
			if !ok || canonSeg != uint32(i) || canonLocal != n.ID {
				continue
			}

			if buf.seen[global] != buf.gen || n.Distance < buf.best[global].Distance {
				n.ID = global
				buf.best[global] = n
				buf.seen[global] = buf.gen
			}
		}
	}

	for global, n := range buf.best {
		if buf.seen[global] != buf.gen {
			continue
		}
		buf.results.Push(n)
		if len(buf.results.Nodes) > k {
			buf.results.Pop()
		}
	}

	for len(buf.results.Nodes) > 0 {
		buf.out = append(buf.out, buf.results.Pop())
	}
	slices.Reverse(buf.out)

	return append(dst[:0], buf.out[:min(k, len(buf.out))]...), nil
}

func (idx *SegmentedIndex) acquireMergeBuffer() *segmentMergeBuffer {
	if buf := idx.scratch.buf.Swap(nil); buf != nil {
		return buf
	}
	return idx.pool.Get().(*segmentMergeBuffer)
}

func (idx *SegmentedIndex) releaseMergeBuffer(buf *segmentMergeBuffer) {
	if buf == nil {
		return
	}
	if idx.scratch.buf.CompareAndSwap(nil, buf) {
		return
	}
	idx.pool.Put(buf)
}

func buildSegmentBindings(
	head *hnsw.Index,
	frozen ...*hnsw.Index,
) (*segmentBinding, []*segmentBinding, error) {
	var headBinding *segmentBinding
	var headCount uint32
	if head != nil {
		headCount = uint32(head.Len())
		headBinding = newContiguousSegmentBinding(head, 0, headCount)
	}

	frozenBindings := make([]*segmentBinding, 0, len(frozen))
	baseGlobal := headCount
	for _, seg := range frozen {
		if seg == nil {
			continue
		}
		localCount := uint32(seg.Len())
		binding := newContiguousSegmentBinding(seg, baseGlobal, localCount)
		frozenBindings = append(frozenBindings, binding)
		baseGlobal += localCount
	}

	if headBinding == nil && len(frozenBindings) == 0 {
		return nil, nil, fmt.Errorf("hnsw: segmented index requires at least one segment")
	}

	return headBinding, frozenBindings, nil
}
