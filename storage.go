package hnsw

import (
	"encoding/binary"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"unsafe"

	"golang.org/x/sys/unix"
)

func (s *Storage) Close() error {
	if err := unix.Msync(s.data, unix.MS_SYNC); err != nil {
		s.file.Close()
		return err
	}
	if err := unix.Munmap(s.data); err != nil {
		s.file.Close()
		return err
	}
	return s.file.Close()
}

const (
	HeaderSize = 64
	Magic      = "HNSW"
)

type IndexConfig struct {
	Dims     uint32
	M        uint32
	MMax0    uint32
	MaxLevel uint32
}

type NodeLayout struct {
	NodeSize             uint32
	LockOffset           uint32
	LevelOffset          uint32
	L0CountOffset        uint32
	UpperCountsOffset    uint32
	VectorOffset         uint32
	L0NeighborsOffset    uint32
	UpperNeighborsOffset uint32
}

func NewNodeLayout(config IndexConfig) NodeLayout {
	mMax0 := config.MMax0
	if mMax0 == 0 {
		mMax0 = config.M * 2
	}

	layout := NodeLayout{}
	offset := uint32(0)

	layout.LockOffset = offset
	offset += 4

	layout.LevelOffset = offset
	offset += 4

	layout.L0CountOffset = offset
	offset += 4

	layout.UpperCountsOffset = offset
	offset += config.MaxLevel * 4

	layout.VectorOffset = offset
	offset += config.Dims * 4

	layout.L0NeighborsOffset = offset
	offset += mMax0 * 4

	layout.UpperNeighborsOffset = offset
	offset += config.MaxLevel * config.M * 4

	layout.NodeSize = (offset + 63) &^ 63

	return layout
}

type Storage struct {
	file   *os.File
	data   []byte
	config IndexConfig
	layout NodeLayout
	mu     sync.RWMutex
}

func NewStorage(path string, config IndexConfig, initialNodes uint32) (*Storage, error) {
	if config.Dims == 0 {
		return nil, fmt.Errorf("hnsw: Dims must be > 0")
	}
	if config.M == 0 {
		return nil, fmt.Errorf("hnsw: M must be > 0")
	}
	if config.MaxLevel == 0 {
		return nil, fmt.Errorf("hnsw: MaxLevel must be > 0")
	}

	if config.MMax0 == 0 {
		config.MMax0 = config.M * 2
	}

	layout := NewNodeLayout(config)
	totalSize := uint64(HeaderSize) + uint64(initialNodes)*uint64(layout.NodeSize)

	file, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o666)
	if err != nil {
		return nil, err
	}

	info, err := file.Stat()
	if err != nil {
		file.Close()
		return nil, err
	}

	var mmapSize int
	if info.Size() < int64(totalSize) {
		if err := file.Truncate(int64(totalSize)); err != nil {
			file.Close()
			return nil, err
		}
		mmapSize = int(totalSize)
	} else {
		mmapSize = int(info.Size())
	}

	data, err := unix.Mmap(
		int(file.Fd()),
		0,
		mmapSize,
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

	if info.Size() == 0 || s.readUint32(0) == 0 {
		s.writeHeader()
	} else {
		if err := s.validateHeader(); err != nil {
			unix.Munmap(data)
			file.Close()
			return nil, err
		}
	}

	return s, nil
}

func (s *Storage) writeHeader() {
	copy(s.data[0:4], Magic)
	s.writeUint32(4, 1)
	s.writeUint32(8, s.config.Dims)
	s.writeUint32(12, s.config.M)
	s.writeUint32(16, s.config.MMax0)
	s.writeUint32(20, s.config.MaxLevel)
	s.writeUint32(36, 0)
	s.writeUint32(32, (uint32(len(s.data))-HeaderSize)/s.layout.NodeSize)
}

func (s *Storage) validateHeader() error {
	magic := string(s.data[0:4])
	if magic != Magic {
		return fmt.Errorf("invalid hnsw file: bad magic %q", magic)
	}
	if s.readUint32(4) != 1 {
		return fmt.Errorf("unsupported hnsw version: %d", s.readUint32(4))
	}
	if s.readUint32(8) != s.config.Dims {
		return fmt.Errorf(
			"dims mismatch: file has %d, config has %d",
			s.readUint32(8),
			s.config.Dims,
		)
	}
	if s.readUint32(12) != s.config.M {
		return fmt.Errorf("M mismatch: file has %d, config has %d", s.readUint32(12), s.config.M)
	}
	if s.readUint32(16) != s.config.MMax0 {
		return fmt.Errorf("MMax0 mismatch: file has %d, config has %d", s.readUint32(16), s.config.MMax0)
	}
	if s.readUint32(20) != s.config.MaxLevel {
		return fmt.Errorf("MaxLevel mismatch: file has %d, config has %d", s.readUint32(20), s.config.MaxLevel)
	}
	return nil
}

func (s *Storage) readUint32(offset uint32) uint32 {
	return binary.LittleEndian.Uint32(s.data[offset : offset+4])
}

func (s *Storage) writeUint32(offset uint32, val uint32) {
	binary.LittleEndian.PutUint32(s.data[offset:offset+4], val)
}

func (s *Storage) getNodeData(id uint32) []byte {
	offset := uint64(HeaderSize) + uint64(id)*uint64(s.layout.NodeSize)
	end := offset + uint64(s.layout.NodeSize)
	if end > uint64(len(s.data)) {
		panic("hnsw: node id out of bounds")
	}
	return s.data[offset:end]
}

func (s *Storage) ReadLock() {
	s.mu.RLock()
}

func (s *Storage) ReadUnlock() {
	s.mu.RUnlock()
}

func (s *Storage) GetVector(id uint32) []float32 {
	data := s.getNodeData(id)
	vecData := data[s.layout.VectorOffset : s.layout.VectorOffset+(s.config.Dims*4)]
	return unsafe.Slice((*float32)(unsafe.Pointer(&vecData[0])), s.config.Dims)
}

func (s *Storage) GetNeighbors(id uint32, layer int) []uint32 {
	data := s.getNodeData(id)
	var neighborsData []byte
	var count int

	if layer == 0 {
		count = int(
			binary.LittleEndian.Uint32(data[s.layout.L0CountOffset : s.layout.L0CountOffset+4]),
		)
		neighborsData = data[s.layout.L0NeighborsOffset : s.layout.L0NeighborsOffset+(s.config.MMax0*4)]
	} else {
		if layer > int(s.config.MaxLevel) {
			panic("hnsw: layer out of bounds")
		}
		countsOffset := s.layout.UpperCountsOffset + uint32(layer-1)*4
		count = int(binary.LittleEndian.Uint32(data[countsOffset : countsOffset+4]))
		layerOffset := s.layout.UpperNeighborsOffset + uint32(layer-1)*s.config.M*4
		neighborsData = data[layerOffset : layerOffset+(s.config.M*4)]
	}

	return unsafe.Slice((*uint32)(unsafe.Pointer(&neighborsData[0])), count)
}

func (s *Storage) SetNeighbors(id uint32, layer int, neighbors []uint32) {
	data := s.getNodeData(id)
	if layer == 0 {
		binary.LittleEndian.PutUint32(
			data[s.layout.L0CountOffset:s.layout.L0CountOffset+4],
			uint32(len(neighbors)),
		)
		neighborsData := data[s.layout.L0NeighborsOffset : s.layout.L0NeighborsOffset+(s.config.MMax0*4)]
		copy(unsafe.Slice((*uint32)(unsafe.Pointer(&neighborsData[0])), s.config.MMax0), neighbors)
	} else {
		if layer > int(s.config.MaxLevel) {
			panic("hnsw: layer out of bounds")
		}
		countsOffset := s.layout.UpperCountsOffset + uint32(layer-1)*4
		binary.LittleEndian.PutUint32(data[countsOffset:countsOffset+4], uint32(len(neighbors)))
		layerOffset := s.layout.UpperNeighborsOffset + uint32(layer-1)*s.config.M*4
		neighborsData := data[layerOffset : layerOffset+(s.config.M*4)]
		copy(unsafe.Slice((*uint32)(unsafe.Pointer(&neighborsData[0])), s.config.M), neighbors)
	}
}

func (s *Storage) LockNode(id uint32) {
	data := s.getNodeData(id)
	lockPtr := (*uint32)(unsafe.Pointer(&data[s.layout.LockOffset]))
	for !atomic.CompareAndSwapUint32(lockPtr, 0, 1) {
	}
}

func (s *Storage) UnlockNode(id uint32) {
	data := s.getNodeData(id)
	lockPtr := (*uint32)(unsafe.Pointer(&data[s.layout.LockOffset]))
	atomic.StoreUint32(lockPtr, 0)
}

func (s *Storage) addNode() (uint32, error) {
	nodeCount := s.readUint32(28)
	allocated := s.readUint32(32)

	if nodeCount >= allocated {
		newAllocated := allocated * 2
		if newAllocated == 0 {
			newAllocated = 1000
		}
		if err := s.grow(newAllocated); err != nil {
			return 0, err
		}
	}

	id := nodeCount
	s.writeUint32(28, id+1)
	return id, nil
}

func (s *Storage) grow(newAllocated uint32) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	newSize := uint64(HeaderSize) + uint64(newAllocated)*uint64(s.layout.NodeSize)

	if err := s.file.Truncate(int64(newSize)); err != nil {
		return fmt.Errorf("truncate: %w", err)
	}

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

	oldData := s.data
	s.data = data
	s.writeUint32(32, newAllocated)

	if oldData != nil {
		unix.Munmap(oldData)
	}

	return nil
}

func (s *Storage) setLevel(id uint32, level int) {
	data := s.getNodeData(id)
	binary.LittleEndian.PutUint32(data[s.layout.LevelOffset:s.layout.LevelOffset+4], uint32(level))
}

func (s *Storage) setVector(id uint32, vec []float32) {
	data := s.getNodeData(id)
	vecData := data[s.layout.VectorOffset : s.layout.VectorOffset+(s.config.Dims*4)]
	copy(unsafe.Slice((*float32)(unsafe.Pointer(&vecData[0])), s.config.Dims), vec)
}
