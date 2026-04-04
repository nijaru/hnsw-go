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

const (
	HeaderSize = 64
	Magic      = "HNSW"
)

type IndexConfig struct {
	Dims     uint32
	M        uint32
	MMax0    uint32
	MaxLevel uint32
	Probes   uint32
}

type NodeLayout struct {
	GraphNodeSize     uint32
	LockOffset        uint32
	LevelOffset       uint32
	L0CountOffset     uint32
	L0NeighborsOffset uint32
	UpperOffsetOffset uint32
	MetaOffsetOffset  uint32
	MetaLenOffset     uint32
	VectorSize        uint32
}

func NewNodeLayout(config IndexConfig) NodeLayout {
	mMax0 := config.MMax0
	if mMax0 == 0 {
		mMax0 = config.M * 2
	}

	var l NodeLayout
	offset := uint32(0)

	l.LockOffset = offset
	offset += 4

	l.LevelOffset = offset
	offset += 4

	l.L0CountOffset = offset
	offset += 4

	l.L0NeighborsOffset = offset
	offset += mMax0 * 4

	l.UpperOffsetOffset = offset
	offset += 4

	l.MetaOffsetOffset = offset
	offset += 4

	l.MetaLenOffset = offset
	offset += 4

	l.GraphNodeSize = (offset + 63) &^ 63
	l.VectorSize = (config.Dims*4 + 63) &^ 63

	return l
}

type Storage struct {
	path        string
	graphFile   *os.File
	graphData   []byte
	vecFile     *os.File
	vecData     []byte
	upperFile   *os.File
	upperData   []byte
	delFile     *os.File
	delData     []byte
	metaFile    *os.File
	metaData    []byte
	config      IndexConfig
	layout      NodeLayout
	vectorSlice []float32
	vecLen      uint32
	allocMu     sync.Mutex
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

	graphSize := uint64(HeaderSize) + uint64(initialNodes)*uint64(layout.GraphNodeSize)
	vecSize := uint64(initialNodes) * uint64(layout.VectorSize)
	// Initial upper data size: ~1/16th of nodes * avg upper size
	upperSize := uint64(initialNodes) / 16 * uint64(config.MaxLevel) * uint64(config.M+1) * 4
	if upperSize < 4096 {
		upperSize = 4096
	}
	delSize := uint64((initialNodes + 7) / 8)
	if delSize < 4096 {
		delSize = 4096
	}
	metaSize := uint64(initialNodes) * 64 // 64 bytes avg meta
	if metaSize < 4096 {
		metaSize = 4096
	}

	graphFile, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o666)
	if err != nil {
		return nil, fmt.Errorf("graph file: %w", err)
	}

	vecPath := path + ".vec"
	vecFile, err := os.OpenFile(vecPath, os.O_RDWR|os.O_CREATE, 0o666)
	if err != nil {
		graphFile.Close()
		return nil, fmt.Errorf("vector file: %w", err)
	}

	upperPath := path + ".upper"
	upperFile, err := os.OpenFile(upperPath, os.O_RDWR|os.O_CREATE, 0o666)
	if err != nil {
		vecFile.Close()
		graphFile.Close()
		return nil, fmt.Errorf("upper file: %w", err)
	}

	delPath := path + ".del"
	delFile, err := os.OpenFile(delPath, os.O_RDWR|os.O_CREATE, 0o666)
	if err != nil {
		upperFile.Close()
		vecFile.Close()
		graphFile.Close()
		return nil, fmt.Errorf("deleted file: %w", err)
	}

	metaPath := path + ".meta"
	metaFile, err := os.OpenFile(metaPath, os.O_RDWR|os.O_CREATE, 0o666)
	if err != nil {
		delFile.Close()
		upperFile.Close()
		vecFile.Close()
		graphFile.Close()
		return nil, fmt.Errorf("meta file: %w", err)
	}

	graphInfo, err := graphFile.Stat()
	if err != nil {
		metaFile.Close()
		delFile.Close()
		upperFile.Close()
		vecFile.Close()
		graphFile.Close()
		return nil, err
	}

	vecInfo, err := vecFile.Stat()
	if err != nil {
		metaFile.Close()
		delFile.Close()
		upperFile.Close()
		vecFile.Close()
		graphFile.Close()
		return nil, err
	}

	upperInfo, err := upperFile.Stat()
	if err != nil {
		metaFile.Close()
		delFile.Close()
		upperFile.Close()
		vecFile.Close()
		graphFile.Close()
		return nil, err
	}

	delInfo, err := delFile.Stat()
	if err != nil {
		metaFile.Close()
		delFile.Close()
		upperFile.Close()
		vecFile.Close()
		graphFile.Close()
		return nil, err
	}

	metaInfo, err := metaFile.Stat()
	if err != nil {
		metaFile.Close()
		delFile.Close()
		upperFile.Close()
		vecFile.Close()
		graphFile.Close()
		return nil, err
	}

	graphMmapSize := max(int(graphInfo.Size()), int(graphSize))
	if graphInfo.Size() < int64(graphSize) {
		if err := graphFile.Truncate(int64(graphSize)); err != nil {
			metaFile.Close()
			delFile.Close()
			upperFile.Close()
			vecFile.Close()
			graphFile.Close()
			return nil, fmt.Errorf("graph truncate: %w", err)
		}
		graphMmapSize = int(graphSize)
	}

	vecMmapSize := max(int(vecInfo.Size()), int(vecSize))
	if vecInfo.Size() < int64(vecSize) {
		if err := vecFile.Truncate(int64(vecSize)); err != nil {
			metaFile.Close()
			delFile.Close()
			upperFile.Close()
			vecFile.Close()
			graphFile.Close()
			return nil, fmt.Errorf("vector truncate: %w", err)
		}
		vecMmapSize = int(vecSize)
	}

	upperMmapSize := max(int(upperInfo.Size()), int(upperSize))
	if upperInfo.Size() < int64(upperSize) {
		if err := upperFile.Truncate(int64(upperSize)); err != nil {
			metaFile.Close()
			delFile.Close()
			upperFile.Close()
			vecFile.Close()
			graphFile.Close()
			return nil, fmt.Errorf("upper truncate: %w", err)
		}
		upperMmapSize = int(upperSize)
	}

	delMmapSize := max(int(delInfo.Size()), int(delSize))
	if delInfo.Size() < int64(delSize) {
		if err := delFile.Truncate(int64(delSize)); err != nil {
			metaFile.Close()
			delFile.Close()
			upperFile.Close()
			vecFile.Close()
			graphFile.Close()
			return nil, fmt.Errorf("deleted truncate: %w", err)
		}
		delMmapSize = int(delSize)
	}

	metaMmapSize := max(int(metaInfo.Size()), int(metaSize))
	if metaInfo.Size() < int64(metaSize) {
		if err := metaFile.Truncate(int64(metaSize)); err != nil {
			metaFile.Close()
			delFile.Close()
			upperFile.Close()
			vecFile.Close()
			graphFile.Close()
			return nil, fmt.Errorf("meta truncate: %w", err)
		}
		metaMmapSize = int(metaSize)
	}

	graphData, err := unix.Mmap(
		int(graphFile.Fd()), 0, graphMmapSize,
		unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED,
	)
	if err != nil {
		metaFile.Close()
		delFile.Close()
		upperFile.Close()
		vecFile.Close()
		graphFile.Close()
		return nil, fmt.Errorf("graph mmap: %w", err)
	}

	vecData, err := unix.Mmap(
		int(vecFile.Fd()), 0, vecMmapSize,
		unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED,
	)
	if err != nil {
		unix.Munmap(graphData)
		metaFile.Close()
		delFile.Close()
		upperFile.Close()
		vecFile.Close()
		graphFile.Close()
		return nil, fmt.Errorf("vector mmap: %w", err)
	}

	upperData, err := unix.Mmap(
		int(upperFile.Fd()), 0, upperMmapSize,
		unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED,
	)
	if err != nil {
		unix.Munmap(vecData)
		unix.Munmap(graphData)
		metaFile.Close()
		delFile.Close()
		upperFile.Close()
		vecFile.Close()
		graphFile.Close()
		return nil, fmt.Errorf("upper mmap: %w", err)
	}

	delData, err := unix.Mmap(
		int(delFile.Fd()), 0, delMmapSize,
		unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED,
	)
	if err != nil {
		unix.Munmap(upperData)
		unix.Munmap(vecData)
		unix.Munmap(graphData)
		metaFile.Close()
		delFile.Close()
		upperFile.Close()
		vecFile.Close()
		graphFile.Close()
		return nil, fmt.Errorf("deleted mmap: %w", err)
	}

	metaData, err := unix.Mmap(
		int(metaFile.Fd()), 0, metaMmapSize,
		unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED,
	)
	if err != nil {
		unix.Munmap(delData)
		unix.Munmap(upperData)
		unix.Munmap(vecData)
		unix.Munmap(graphData)
		metaFile.Close()
		delFile.Close()
		upperFile.Close()
		vecFile.Close()
		graphFile.Close()
		return nil, fmt.Errorf("meta mmap: %w", err)
	}

	s := &Storage{
		path:      path,
		graphFile: graphFile,
		graphData: graphData,
		vecFile:   vecFile,
		vecData:   vecData,
		upperFile: upperFile,
		upperData: upperData,
		delFile:   delFile,
		delData:   delData,
		metaFile:  metaFile,
		metaData:  metaData,
		config:    config,
		layout:    layout,
		vecLen:    config.Dims,
	}

	if graphInfo.Size() == 0 || s.readUint32(0) == 0 {
		s.writeHeader(initialNodes, uint32(upperMmapSize), uint32(metaMmapSize))
	} else {
		if err := s.validateHeader(); err != nil {
			unix.Munmap(metaData)
			unix.Munmap(delData)
			unix.Munmap(upperData)
			unix.Munmap(vecData)
			unix.Munmap(graphData)
			metaFile.Close()
			delFile.Close()
			upperFile.Close()
			vecFile.Close()
			graphFile.Close()
			return nil, err
		}
	}

	s.vectorSlice = unsafe.Slice(
		(*float32)(unsafe.Pointer(&s.vecData[0])),
		uint64(len(s.vecData))/4,
	)

	return s, nil
}

func (s *Storage) Sync() error {
	var errs []error
	if s.graphData != nil {
		if err := unix.Msync(s.graphData, unix.MS_SYNC); err != nil {
			errs = append(errs, err)
		}
	}
	if s.vecData != nil {
		if err := unix.Msync(s.vecData, unix.MS_SYNC); err != nil {
			errs = append(errs, err)
		}
	}
	if s.upperData != nil {
		if err := unix.Msync(s.upperData, unix.MS_SYNC); err != nil {
			errs = append(errs, err)
		}
	}
	if s.delData != nil {
		if err := unix.Msync(s.delData, unix.MS_SYNC); err != nil {
			errs = append(errs, err)
		}
	}
	if s.metaData != nil {
		if err := unix.Msync(s.metaData, unix.MS_SYNC); err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("sync errors: %v", errs)
	}
	return nil
}

func (s *Storage) Close() error {
	var errs []error
	if s.graphData != nil {
		if err := unix.Msync(s.graphData, unix.MS_SYNC); err != nil {
			errs = append(errs, err)
		}
		if err := unix.Munmap(s.graphData); err != nil {
			errs = append(errs, err)
		}
		s.graphData = nil
	}
	if s.vecData != nil {
		if err := unix.Msync(s.vecData, unix.MS_SYNC); err != nil {
			errs = append(errs, err)
		}
		if err := unix.Munmap(s.vecData); err != nil {
			errs = append(errs, err)
		}
		s.vecData = nil
	}
	if s.upperData != nil {
		if err := unix.Msync(s.upperData, unix.MS_SYNC); err != nil {
			errs = append(errs, err)
		}
		if err := unix.Munmap(s.upperData); err != nil {
			errs = append(errs, err)
		}
		s.upperData = nil
	}
	if s.delData != nil {
		if err := unix.Msync(s.delData, unix.MS_SYNC); err != nil {
			errs = append(errs, err)
		}
		if err := unix.Munmap(s.delData); err != nil {
			errs = append(errs, err)
		}
		s.delData = nil
	}
	if s.metaData != nil {
		if err := unix.Msync(s.metaData, unix.MS_SYNC); err != nil {
			errs = append(errs, err)
		}
		if err := unix.Munmap(s.metaData); err != nil {
			errs = append(errs, err)
		}
		s.metaData = nil
	}
	if s.graphFile != nil {
		if err := s.graphFile.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if s.vecFile != nil {
		if err := s.vecFile.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if s.upperFile != nil {
		if err := s.upperFile.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if s.delFile != nil {
		if err := s.delFile.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if s.metaFile != nil {
		if err := s.metaFile.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("close errors: %v", errs)
	}
	return nil
}

func (s *Storage) IsDeleted(id uint32) bool {
	byteIdx := id / 8
	if int(byteIdx) >= len(s.delData) {
		return false
	}
	bitIdx := id % 8
	return (s.delData[byteIdx] & (1 << bitIdx)) != 0
}

func (s *Storage) SetDeleted(id uint32, deleted bool) {
	byteIdx := id / 8
	if int(byteIdx) >= len(s.delData) {
		if !deleted {
			return
		}
		newSize := max(uint32(len(s.delData)*2), (id/8+4096)&^4095)
		if err := s.growDeleted(newSize); err != nil {
			panic(err)
		}
	}
	bitIdx := id % 8
	if deleted {
		if (s.delData[byteIdx] & (1 << bitIdx)) == 0 {
			s.delData[byteIdx] |= (1 << bitIdx)
			count := s.readUint32(48)
			s.writeUint32(48, count+1)
		}
	} else {
		if (s.delData[byteIdx] & (1 << bitIdx)) != 0 {
			s.delData[byteIdx] &^= (1 << bitIdx)
			count := s.readUint32(48)
			if count > 0 {
				s.writeUint32(48, count-1)
			}
		}
	}
}

func (s *Storage) growDeleted(newSize uint32) error {
	if err := s.delFile.Truncate(int64(newSize)); err != nil {
		return fmt.Errorf("deleted truncate: %w", err)
	}
	delData, err := unix.Mmap(
		int(s.delFile.Fd()), 0, int(newSize),
		unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED,
	)
	if err != nil {
		return fmt.Errorf("deleted mmap: %w", err)
	}
	oldDel := s.delData
	s.delData = delData
	if oldDel != nil {
		unix.Munmap(oldDel)
	}
	return nil
}

func (s *Storage) writeHeader(initialNodes, initialUpperSize, initialMetaSize uint32) {
	copy(s.graphData[0:4], Magic)
	s.writeUint32(4, 5) // Version 5
	s.writeUint32(8, s.config.Dims)
	s.writeUint32(12, s.config.M)
	s.writeUint32(16, s.config.MMax0)
	s.writeUint32(20, s.config.MaxLevel)
	s.writeUint32(24, 0)
	s.writeUint32(28, 0)
	s.writeUint32(32, initialNodes)
	s.writeUint32(36, 0)
	s.writeUint32(40, initialUpperSize)
	s.writeUint32(44, 4)
	s.writeUint32(48, 0)
	s.writeUint32(52, initialMetaSize)
	s.writeUint32(56, 4)
}

func (s *Storage) validateHeader() error {
	magic := string(s.graphData[0:4])
	if magic != Magic {
		return fmt.Errorf("invalid hnsw file: bad magic %q", magic)
	}
	ver := s.readUint32(4)
	if ver < 1 || ver > 5 {
		return fmt.Errorf("unsupported hnsw version: %d", ver)
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
		return fmt.Errorf(
			"MMax0 mismatch: file has %d, config has %d",
			s.readUint32(16),
			s.config.MMax0,
		)
	}
	if s.readUint32(20) != s.config.MaxLevel {
		return fmt.Errorf(
			"MaxLevel mismatch: file has %d, config has %d",
			s.readUint32(20),
			s.config.MaxLevel,
		)
	}
	return nil
}

func (s *Storage) readUint32(offset uint64) uint32 {
	return binary.LittleEndian.Uint32(s.graphData[offset : offset+4])
}

func (s *Storage) writeUint32(offset uint64, val uint32) {
	binary.LittleEndian.PutUint32(s.graphData[offset:offset+4], val)
}

func (s *Storage) writeUint32Upper(offset uint64, val uint32) {
	binary.LittleEndian.PutUint32(s.upperData[offset:offset+4], val)
}

func (s *Storage) GetMaxLevel(id uint32) int {
	offset := uint64(HeaderSize) + uint64(id)*uint64(s.layout.GraphNodeSize)
	return int(s.readUint32(offset))
}

func (s *Storage) getGraphNode(id uint32) []byte {
	offset := uint64(HeaderSize) + uint64(id)*uint64(s.layout.GraphNodeSize)
	end := offset + uint64(s.layout.GraphNodeSize)
	if end > uint64(len(s.graphData)) {
		panic("hnsw: node id out of bounds")
	}
	return s.graphData[offset:end]
}

func (s *Storage) GetVector(id uint32) []float32 {
	vecOffset := uint64(id) * uint64(s.layout.VectorSize) / 4
	end := vecOffset + uint64(s.config.Dims)
	if end*4 > uint64(len(s.vecData)) {
		panic("hnsw: vector id out of bounds")
	}
	return s.vectorSlice[vecOffset : vecOffset+uint64(s.config.Dims)]
}

func (s *Storage) GetMetadata(id uint32) []byte {
	data := s.getGraphNode(id)
	offset := binary.LittleEndian.Uint32(
		data[s.layout.MetaOffsetOffset : s.layout.MetaOffsetOffset+4],
	)
	if offset == 0 {
		return nil
	}
	length := binary.LittleEndian.Uint32(data[s.layout.MetaLenOffset : s.layout.MetaLenOffset+4])
	return s.metaData[offset : offset+length]
}

func (s *Storage) SetMetadata(id uint32, meta []byte) error {
	if len(meta) == 0 {
		return nil
	}

	size := uint32(len(meta))

	s.allocMu.Lock()
	used := s.readUint32(56)
	allocated := s.readUint32(52)

	if used+size > allocated {
		newAllocated := max(allocated*2, used+size+4096)
		if err := s.growMeta(newAllocated); err != nil {
			s.allocMu.Unlock()
			return err
		}
	}

	offset := used
	copy(s.metaData[offset:offset+size], meta)
	s.writeUint32(56, used+size)
	s.allocMu.Unlock()

	data := s.getGraphNode(id)
	binary.LittleEndian.PutUint32(
		data[s.layout.MetaOffsetOffset:s.layout.MetaOffsetOffset+4],
		offset,
	)
	binary.LittleEndian.PutUint32(data[s.layout.MetaLenOffset:s.layout.MetaLenOffset+4], size)
	return nil
}

func (s *Storage) growMeta(newAllocated uint32) error {
	if err := s.metaFile.Truncate(int64(newAllocated)); err != nil {
		return fmt.Errorf("meta truncate: %w", err)
	}

	metaData, err := unix.Mmap(
		int(s.metaFile.Fd()), 0, int(newAllocated),
		unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED,
	)
	if err != nil {
		return fmt.Errorf("meta mmap: %w", err)
	}

	oldMeta := s.metaData
	s.metaData = metaData
	s.writeUint32(52, newAllocated)

	if oldMeta != nil {
		unix.Munmap(oldMeta)
	}
	return nil
}

func (s *Storage) GetNeighbors(id uint32, layer int) []uint32 {
	data := s.getGraphNode(id)
	var neighborsData []byte
	var count int

	if layer == 0 {
		count = int(
			binary.LittleEndian.Uint32(data[s.layout.L0CountOffset : s.layout.L0CountOffset+4]),
		)
		neighborsData = data[s.layout.L0NeighborsOffset : s.layout.L0NeighborsOffset+(s.config.MMax0*4)]
	} else {
		offset := binary.LittleEndian.Uint32(data[s.layout.UpperOffsetOffset : s.layout.UpperOffsetOffset+4])
		if offset == 0 {
			return nil
		}
		level := int(binary.LittleEndian.Uint32(data[s.layout.LevelOffset : s.layout.LevelOffset+4]))
		if layer > level {
			return nil
		}

		// upperData layout: [L]counts, [L*M]neighbors
		countsOffset := offset + uint32(layer-1)*4
		count = int(binary.LittleEndian.Uint32(s.upperData[countsOffset : countsOffset+4]))

		allCountsSize := uint32(level) * 4
		layerNbOffset := offset + allCountsSize + uint32(layer-1)*s.config.M*4
		neighborsData = s.upperData[layerNbOffset : layerNbOffset+(s.config.M*4)]
	}

	return unsafe.Slice((*uint32)(unsafe.Pointer(&neighborsData[0])), count)
}

func (s *Storage) SetNeighbors(id uint32, layer int, neighbors []uint32) {
	data := s.getGraphNode(id)
	var neighborsData []byte

	if layer == 0 {
		binary.LittleEndian.PutUint32(
			data[s.layout.L0CountOffset:s.layout.L0CountOffset+4],
			uint32(len(neighbors)),
		)
		neighborsData = data[s.layout.L0NeighborsOffset : s.layout.L0NeighborsOffset+(s.config.MMax0*4)]
		copy(unsafe.Slice((*uint32)(unsafe.Pointer(&neighborsData[0])), s.config.MMax0), neighbors)
	} else {
		offset := binary.LittleEndian.Uint32(data[s.layout.UpperOffsetOffset : s.layout.UpperOffsetOffset+4])
		if offset == 0 {
			panic("hnsw: setting upper neighbors on node with level 0")
		}
		level := int(binary.LittleEndian.Uint32(data[s.layout.LevelOffset : s.layout.LevelOffset+4]))
		if layer > level {
			panic("hnsw: layer > node level")
		}

		countsOffset := offset + uint32(layer-1)*4
		binary.LittleEndian.PutUint32(s.upperData[countsOffset:countsOffset+4], uint32(len(neighbors)))

		allCountsSize := uint32(level) * 4
		layerNbOffset := offset + allCountsSize + uint32(layer-1)*s.config.M*4
		neighborsData = s.upperData[layerNbOffset : layerNbOffset+(s.config.M*4)]
		copy(unsafe.Slice((*uint32)(unsafe.Pointer(&neighborsData[0])), s.config.M), neighbors)
	}
}

func (s *Storage) allocateUpper(id uint32, level int) error {
	if level <= 0 {
		return nil
	}

	size := uint32(level) * (1 + s.config.M) * 4

	s.allocMu.Lock()
	used := s.readUint32(44)
	allocated := s.readUint32(40)

	if used+size > allocated {
		newAllocated := max(allocated*2, used+size+4096)
		if err := s.growUpper(newAllocated); err != nil {
			s.allocMu.Unlock()
			return err
		}
	}

	offset := used
	s.writeUint32(44, used+size)
	s.allocMu.Unlock()

	data := s.getGraphNode(id)
	binary.LittleEndian.PutUint32(
		data[s.layout.UpperOffsetOffset:s.layout.UpperOffsetOffset+4],
		offset,
	)
	return nil
}

func (s *Storage) growUpper(newAllocated uint32) error {
	if err := s.upperFile.Truncate(int64(newAllocated)); err != nil {
		return fmt.Errorf("upper truncate: %w", err)
	}

	upperData, err := unix.Mmap(
		int(s.upperFile.Fd()), 0, int(newAllocated),
		unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED,
	)
	if err != nil {
		return fmt.Errorf("upper mmap: %w", err)
	}

	oldUpper := s.upperData
	s.upperData = upperData
	s.writeUint32(40, newAllocated)

	if oldUpper != nil {
		unix.Munmap(oldUpper)
	}
	return nil
}

func (s *Storage) LockNode(id uint32) {
	data := s.getGraphNode(id)
	lockPtr := (*uint32)(unsafe.Pointer(&data[s.layout.LockOffset]))
	for !atomic.CompareAndSwapUint32(lockPtr, 0, 1) {
	}
}

func (s *Storage) UnlockNode(id uint32) {
	data := s.getGraphNode(id)
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
	newGraphSize := uint64(HeaderSize) + uint64(newAllocated)*uint64(s.layout.GraphNodeSize)
	newVecSize := uint64(newAllocated) * uint64(s.layout.VectorSize)

	if err := s.graphFile.Truncate(int64(newGraphSize)); err != nil {
		return fmt.Errorf("graph truncate: %w", err)
	}

	if err := s.vecFile.Truncate(int64(newVecSize)); err != nil {
		return fmt.Errorf("vector truncate: %w", err)
	}

	graphData, err := unix.Mmap(
		int(s.graphFile.Fd()), 0, int(newGraphSize),
		unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED,
	)
	if err != nil {
		return fmt.Errorf("graph mmap: %w", err)
	}

	vecData, err := unix.Mmap(
		int(s.vecFile.Fd()), 0, int(newVecSize),
		unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED,
	)
	if err != nil {
		unix.Munmap(graphData)
		return fmt.Errorf("vector mmap: %w", err)
	}

	oldGraph := s.graphData
	oldVec := s.vecData
	s.graphData = graphData
	s.vecData = vecData
	s.writeUint32(32, newAllocated)
	s.vectorSlice = unsafe.Slice(
		(*float32)(unsafe.Pointer(&s.vecData[0])),
		uint64(len(s.vecData))/4,
	)

	if oldGraph != nil {
		unix.Munmap(oldGraph)
	}
	if oldVec != nil {
		unix.Munmap(oldVec)
	}

	return nil
}

func (s *Storage) setLevel(id uint32, level int) {
	data := s.getGraphNode(id)
	binary.LittleEndian.PutUint32(data[s.layout.LevelOffset:s.layout.LevelOffset+4], uint32(level))
}

func (s *Storage) setVector(id uint32, vec []float32) {
	dst := s.GetVector(id)
	copy(dst, vec)
}

func (s *Storage) ResetNode(id uint32) {
	data := s.getGraphNode(id)
	for i := range data {
		data[i] = 0
	}
}
