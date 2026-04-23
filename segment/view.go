package segment

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"sync/atomic"

	"github.com/omendb/hnsw-go"
)

const invalidGlobalID = math.MaxUint32

type segmentLocation struct {
	segment uint32
	local   uint32
}

func (loc segmentLocation) valid() bool {
	return loc.segment != invalidGlobalID
}

type segmentBinding struct {
	index         *hnsw.Index
	localToGlobal []uint32
}

func newSegmentBinding(index *hnsw.Index, localToGlobal []uint32) *segmentBinding {
	return &segmentBinding{
		index:         index,
		localToGlobal: slices.Clone(localToGlobal),
	}
}

func newContiguousSegmentBinding(index *hnsw.Index, baseGlobal, localCount uint32) *segmentBinding {
	localToGlobal := make([]uint32, localCount)
	for i := range localToGlobal {
		localToGlobal[i] = baseGlobal + uint32(i)
	}
	return &segmentBinding{
		index:         index,
		localToGlobal: localToGlobal,
	}
}

func (b *segmentBinding) localCount() uint32 {
	return uint32(len(b.localToGlobal))
}

func (b *segmentBinding) globalID(local uint32) (uint32, bool) {
	if int(local) >= len(b.localToGlobal) {
		return 0, false
	}

	global := b.localToGlobal[local]
	if global == invalidGlobalID {
		return 0, false
	}

	return global, true
}

type segmentView struct {
	version   uint64
	head      *segmentBinding
	frozen    []*segmentBinding
	locations []segmentLocation
}

func newSegmentView(
	version uint64,
	head *segmentBinding,
	frozen ...*segmentBinding,
) (*segmentView, error) {
	segments := make([]*segmentBinding, 0, 1+len(frozen))
	frozenCopy := make([]*segmentBinding, 0, len(frozen))
	if head != nil {
		segments = append(segments, head)
	}
	for _, seg := range frozen {
		if seg == nil {
			continue
		}
		segments = append(segments, seg)
		frozenCopy = append(frozenCopy, seg)
	}

	if len(segments) == 0 {
		return &segmentView{version: version}, nil
	}

	maxGlobal := uint32(0)
	seen := make(map[uint32]struct{}, len(segments))
	for _, seg := range segments {
		for _, global := range seg.localToGlobal {
			if global == invalidGlobalID {
				return nil, fmt.Errorf("hnsw: segment view contains invalid global id")
			}
			if _, ok := seen[global]; ok {
				return nil, fmt.Errorf("hnsw: duplicate global id %d", global)
			}
			seen[global] = struct{}{}
			if global > maxGlobal {
				maxGlobal = global
			}
		}
	}

	if maxGlobal == invalidGlobalID {
		return nil, fmt.Errorf("hnsw: global id space exhausted")
	}

	locations := make([]segmentLocation, maxGlobal+1)
	for i := range locations {
		locations[i] = segmentLocation{segment: invalidGlobalID, local: invalidGlobalID}
	}

	for segIdx, seg := range segments {
		for local, global := range seg.localToGlobal {
			locations[global] = segmentLocation{
				segment: uint32(segIdx),
				local:   uint32(local),
			}
		}
	}

	view := &segmentView{
		version:   version,
		head:      head,
		frozen:    frozenCopy,
		locations: locations,
	}
	return view, nil
}

func (v *segmentView) segmentCount() int {
	if v == nil {
		return 0
	}
	if v.head == nil {
		return len(v.frozen)
	}
	return 1 + len(v.frozen)
}

func (v *segmentView) segmentAt(i int) *segmentBinding {
	if v == nil || i < 0 {
		return nil
	}
	if v.head != nil {
		if i == 0 {
			return v.head
		}
		i--
	}
	if i >= len(v.frozen) {
		return nil
	}
	return v.frozen[i]
}

func (v *segmentView) resolve(global uint32) (segment uint32, local uint32, ok bool) {
	if v == nil || int(global) >= len(v.locations) {
		return 0, 0, false
	}

	loc := v.locations[global]
	if !loc.valid() {
		return 0, 0, false
	}

	return loc.segment, loc.local, true
}

func (v *segmentView) globalID(segment int, local uint32) (uint32, bool) {
	seg := v.segmentAt(segment)
	if seg == nil {
		return 0, false
	}
	return seg.globalID(local)
}

type segmentCatalog struct {
	mu      sync.Mutex
	version uint64
	dir     string
	current atomic.Pointer[segmentView]
}

func newSegmentCatalog(dir string) *segmentCatalog {
	return &segmentCatalog{dir: dir}
}

type Manifest struct {
	Version uint64   `json:"version"`
	Head    string   `json:"head,omitempty"`
	Frozen  []string `json:"frozen,omitempty"`
}

func (c *segmentCatalog) load() *segmentView {
	return c.current.Load()
}

func (c *segmentCatalog) publish(
	head *segmentBinding,
	frozen ...*segmentBinding,
) (*segmentView, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.version++
	view, err := newSegmentView(c.version, head, frozen...)
	if err != nil {
		return nil, err
	}

	if c.dir != "" {
		if err := c.writeManifest(head, frozen); err != nil {
			return nil, err
		}
	}

	c.current.Store(view)
	return view, nil
}

func (c *segmentCatalog) writeManifest(head *segmentBinding, frozen []*segmentBinding) error {
	m := Manifest{
		Version: c.version,
	}
	if head != nil {
		m.Head = head.index.Path()
	}
	for _, f := range frozen {
		if f != nil {
			m.Frozen = append(m.Frozen, f.index.Path())
		}
	}

	b, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}

	manifestPath := filepath.Join(c.dir, "manifest.json")
	tmpPath := manifestPath + ".tmp"
	if err := os.WriteFile(tmpPath, b, 0o644); err != nil {
		return err
	}

	return os.Rename(tmpPath, manifestPath)
}

// ReadManifest reads a segment manifest from the specified directory.
func ReadManifest(dir string) (Manifest, error) {
	var m Manifest
	b, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return m, err
	}
	err = json.Unmarshal(b, &m)
	return m, err
}
