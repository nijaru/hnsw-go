package hnsw

import (
	"math/bits"
	"slices"
)

// AllowList is a precomputed set of node IDs that search may return.
// IDs must be sorted ascending and unique when used.
type AllowList struct {
	IDs  []uint32
	Bits []uint64
}

// NewAllowIDsSorted builds an allow-list from sorted, unique IDs.
func NewAllowIDsSorted(ids []uint32) AllowList {
	return AllowList{IDs: ids}
}

// NewAllowBitset builds an allow-list from a bitset where bit N allows node N.
func NewAllowBitset(bits []uint64) AllowList {
	return AllowList{Bits: bits}
}

// Len reports the number of allowed IDs in the list.
func (a AllowList) Len() int {
	if len(a.IDs) > 0 {
		return len(a.IDs)
	}

	count := 0
	for _, word := range a.Bits {
		count += bits.OnesCount64(word)
	}
	return count
}

// Contains reports whether id is present in the allow-list.
func (a AllowList) Contains(id uint32) bool {
	if len(a.Bits) > 0 {
		word := id / 64
		if int(word) >= len(a.Bits) {
			return false
		}
		mask := uint64(1) << (id % 64)
		return (a.Bits[word] & mask) != 0
	}
	if len(a.IDs) == 0 {
		return false
	}
	_, ok := slices.BinarySearch(a.IDs, id)
	return ok
}

// ForEach calls fn for every allowed ID until fn returns false.
func (a AllowList) ForEach(fn func(uint32) bool) {
	if len(a.Bits) > 0 {
		for wordIdx, word := range a.Bits {
			base := uint32(wordIdx) * 64
			for word != 0 {
				bit := bits.TrailingZeros64(word)
				if !fn(base + uint32(bit)) {
					return
				}
				word &= word - 1
			}
		}
		return
	}

	for _, id := range a.IDs {
		if !fn(id) {
			return
		}
	}
}
