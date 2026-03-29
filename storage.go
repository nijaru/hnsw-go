package hnsw

import (
	"encoding/binary"
	"fmt"
	"os"
	"sync/atomic"
	"unsafe"

	"golang.org/x/sys/unix"
)

// Close unmaps the data and closes the file.
func (s *Storage) Close() error {
	if err := unix.Munmap(s.data); err != nil {
		s.file.Close()
		return err
	}
	return s.file.Close()
}

const (
	// HeaderSize is the size of the global metadata at the start of the file.
	// Padded to 64 bytes.
	HeaderSize = 64

	Magic = "HNSW"
)

// IndexConfig holds the static configuration for the HNSW index.
type IndexConfig struct {
	Dims     uint32
	M        uint32
	M_max0   uint32
	MaxLevel uint32
}

// NodeLayout defines the byte offsets for fields within a single node block.
type NodeLayout struct {
	NodeSize uint32

	LockOffset           uint32
	LevelOffset          uint32
	L0CountOffset        uint32
	UpperCountsOffset    uint32
	VectorOffset         uint32
	L0NeighborsOffset    uint32
	UpperNeighborsOffset uint32
}

func NewNodeLayout(config IndexConfig) NodeLayout {
	mMax0 := config.M_max0
	if mMax0 == 0 {
		mMax0 = config.M * 2
	}

	layout := NodeLayout{}
	offset := uint32(0)

	layout.LockOffset = offset
	offset += 4 // uint32

	layout.LevelOffset = offset
	offset += 4 // int32

	layout.L0CountOffset = offset
	offset += 4 // int32

	layout.UpperCountsOffset = offset
	offset += config.MaxLevel * 4 // [MaxLevel]int32

	layout.VectorOffset = offset
	offset += config.Dims * 4 // [Dims]float32

	layout.L0NeighborsOffset = offset
	offset += mMax0 * 4 // [M_max0]uint32

	layout.UpperNeighborsOffset = offset
	offset += config.MaxLevel * config.M * 4 // [MaxLevel][M]uint32

	// Align to 64 bytes
	layout.NodeSize = (offset + 63) &^ 63

	return layout
}

// Storage represents the mmap'd file storage for the HNSW index.
type Storage struct {
	file   *os.File
	data   []byte
	config IndexConfig
	layout NodeLayout
}

// NewStorage opens or creates an HNSW storage file.
func NewStorage(path string, config IndexConfig, initialNodes uint32) (*Storage, error) {
	layout := NewNodeLayout(config)
	totalSize := HeaderSize + (initialNodes * layout.NodeSize)

	file, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o666)
	if err != nil {
		return nil, err
	}

	info, err := file.Stat()
	if err != nil {
		file.Close()
		return nil, err
	}

	if info.Size() < int64(totalSize) {
		if err := file.Truncate(int64(totalSize)); err != nil {
			file.Close()
			return nil, err
		}
	} else {
		totalSize = uint32(info.Size())
	}

	data, err := unix.Mmap(
		int(file.Fd()),
		0,
		int(totalSize),
		unix.PROT_READ|unix.PROT_WRITE,
		unix.MAP_SHARED,
	)
	if err != nil {
		file.Close()
		return nil, err
	}

	s := &Storage{
		file:   file,
		data:   data,
		config: config,
		layout: layout,
	}

	// Initialize header if new file
	if info.Size() == 0 || s.readUint32(0) == 0 {
		s.writeHeader()
	}

	return s, nil
}

func (s *Storage) writeHeader() {
	copy(s.data[0:4], Magic)
	s.writeUint32(4, 1) // Version
	s.writeUint32(8, s.config.Dims)
	s.writeUint32(12, s.config.M)
	s.writeUint32(16, s.config.M_max0)
	s.writeUint32(20, s.config.MaxLevel)
	// EntryPoint (24), NodeCount (28), AllocatedNodes (32) are 0 by default
	s.writeUint32(32, (uint32(len(s.data))-HeaderSize)/s.layout.NodeSize)
}

func (s *Storage) readUint32(offset uint32) uint32 {
	return binary.LittleEndian.Uint32(s.data[offset : offset+4])
}

func (s *Storage) writeUint32(offset uint32, val uint32) {
	binary.LittleEndian.PutUint32(s.data[offset:offset+4], val)
}

// GetNodeData returns the raw byte slice for a node.
func (s *Storage) GetNodeData(id uint32) []byte {
	offset := HeaderSize + (id * s.layout.NodeSize)
	return s.data[offset : offset+s.layout.NodeSize]
}

// GetVector returns a float32 slice pointing to the node's vector.
func (s *Storage) GetVector(id uint32) []float32 {
	data := s.GetNodeData(id)
	vecData := data[s.layout.VectorOffset : s.layout.VectorOffset+(s.config.Dims*4)]
	return unsafe.Slice((*float32)(unsafe.Pointer(&vecData[0])), s.config.Dims)
}

// GetNeighbors returns a uint32 slice pointing to the node's neighbors at a layer.
func (s *Storage) GetNeighbors(id uint32, layer int) []uint32 {
	data := s.GetNodeData(id)
	var neighborsData []byte
	var count int

	if layer == 0 {
		count = int(
			binary.LittleEndian.Uint32(data[s.layout.L0CountOffset : s.layout.L0CountOffset+4]),
		)
		neighborsData = data[s.layout.L0NeighborsOffset : s.layout.L0NeighborsOffset+(s.config.M_max0*4)]
	} else {
		countsOffset := s.layout.UpperCountsOffset + uint32(layer-1)*4
		count = int(binary.LittleEndian.Uint32(data[countsOffset : countsOffset+4]))
		layerOffset := s.layout.UpperNeighborsOffset + uint32(layer-1)*s.config.M*4
		neighborsData = data[layerOffset : layerOffset+(s.config.M*4)]
	}

	return unsafe.Slice((*uint32)(unsafe.Pointer(&neighborsData[0])), count)
}

// SetNeighbors updates the neighbor list for a node at a given layer.
func (s *Storage) SetNeighbors(id uint32, layer int, neighbors []uint32) {
	data := s.GetNodeData(id)
	if layer == 0 {
		binary.LittleEndian.PutUint32(
			data[s.layout.L0CountOffset:s.layout.L0CountOffset+4],
			uint32(len(neighbors)),
		)
		neighborsData := data[s.layout.L0NeighborsOffset : s.layout.L0NeighborsOffset+(s.config.M_max0*4)]
		copy(unsafe.Slice((*uint32)(unsafe.Pointer(&neighborsData[0])), s.config.M_max0), neighbors)
	} else {
		countsOffset := s.layout.UpperCountsOffset + uint32(layer-1)*4
		binary.LittleEndian.PutUint32(data[countsOffset:countsOffset+4], uint32(len(neighbors)))
		layerOffset := s.layout.UpperNeighborsOffset + uint32(layer-1)*s.config.M*4
		neighborsData := data[layerOffset : layerOffset+(s.config.M*4)]
		copy(unsafe.Slice((*uint32)(unsafe.Pointer(&neighborsData[0])), s.config.M), neighbors)
	}
}

// LockNode acquires a spinlock on the node.
// Note: Offset is guaranteed to be 4-byte aligned due to 64-byte node padding.
func (s *Storage) LockNode(id uint32) {
	data := s.GetNodeData(id)
	lockPtr := (*uint32)(unsafe.Pointer(&data[s.layout.LockOffset]))
	for !atomic.CompareAndSwapUint32(lockPtr, 0, 1) {
		// spin
	}
}

// UnlockNode releases the spinlock on the node.
func (s *Storage) UnlockNode(id uint32) {
	data := s.GetNodeData(id)
	lockPtr := (*uint32)(unsafe.Pointer(&data[s.layout.LockOffset]))
	atomic.StoreUint32(lockPtr, 0)
}

// AddNode allocates a new node ID and grows the storage if necessary.
func (s *Storage) AddNode() (uint32, error) {
	nodeCount := s.readUint32(28)
	allocated := s.readUint32(32)

	if nodeCount >= allocated {
		// Grow by 2x or at least 1000 nodes
		newAllocated := allocated * 2
		if newAllocated == 0 {
			newAllocated = 1000
		}
		if err := s.Grow(newAllocated); err != nil {
			return 0, err
		}
	}

	id := nodeCount
	s.writeUint32(28, id+1)
	return id, nil
}

// Grow increases the capacity of the mmap file to accommodate more nodes.
func (s *Storage) Grow(newAllocated uint32) error {
	newSize := HeaderSize + (newAllocated * s.layout.NodeSize)

	// Unmap
	if err := unix.Munmap(s.data); err != nil {
		return fmt.Errorf("munmap: %w", err)
	}

	// Truncate
	if err := s.file.Truncate(int64(newSize)); err != nil {
		return fmt.Errorf("truncate: %w", err)
	}

	// Remap
	data, err := unix.Mmap(
		int(s.file.Fd()),
		0,
		int(newSize),
		unix.PROT_READ|unix.PROT_WRITE,
		unix.MAP_SHARED,
	)
	if err != nil {
		return fmt.Errorf("mmap: %w", err)
	}

	s.data = data
	s.writeUint32(32, newAllocated)
	return nil
}
