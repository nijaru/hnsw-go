package hnsw

import (
	"cmp"
	"slices"
	"sync"
)

// TermIndex is an embedded inverted index mapping string terms (like tenants or tags)
// to node IDs. It provides fast, zero-alloc allow-list generation for filtered search.
type TermIndex struct {
	mu    sync.RWMutex
	terms map[string][]uint32
}

func NewTermIndex() *TermIndex {
	return &TermIndex{
		terms: make(map[string][]uint32),
	}
}

// Add associates a node ID with one or more terms.
func (ti *TermIndex) Add(id uint32, terms ...string) {
	ti.mu.Lock()
	defer ti.mu.Unlock()

	for _, term := range terms {
		ids := ti.terms[term]
		if len(ids) == 0 || ids[len(ids)-1] < id {
			ti.terms[term] = append(ids, id)
		} else if ids[len(ids)-1] != id {
			idx, found := slices.BinarySearch(ids, id)
			if !found {
				ti.terms[term] = slices.Insert(ids, idx, id)
			}
		}
	}
}

// Remove drops a node ID from specific terms. It preserves sort order.
func (ti *TermIndex) Remove(id uint32, terms ...string) {
	ti.mu.Lock()
	defer ti.mu.Unlock()

	for _, term := range terms {
		ids := ti.terms[term]
		for i, existing := range ids {
			if existing == id {
				copy(ids[i:], ids[i+1:])
				ti.terms[term] = ids[:len(ids)-1]
				break
			}
		}
	}
}

// Allow returns an AllowList containing IDs that possess the given term.
func (ti *TermIndex) Allow(term string) AllowList {
	ti.mu.RLock()
	defer ti.mu.RUnlock()

	ids := ti.terms[term]
	if len(ids) == 0 {
		return AllowList{}
	}

	clone := make([]uint32, len(ids))
	copy(clone, ids)
	return NewAllowIDsSorted(clone)
}

// AllowIntersect returns an AllowList of IDs that possess ALL given terms (AND).
func (ti *TermIndex) AllowIntersect(terms ...string) AllowList {
	if len(terms) == 0 {
		return AllowList{}
	}

	ti.mu.RLock()
	defer ti.mu.RUnlock()

	var smallest []uint32
	var smallestTerm string
	for _, term := range terms {
		ids := ti.terms[term]
		if len(ids) == 0 {
			return AllowList{} // If any term is empty, the intersection is empty.
		}
		if smallest == nil || len(ids) < len(smallest) {
			smallest = ids
			smallestTerm = term
		}
	}

	intersected := make([]uint32, 0, len(smallest))
	for _, id := range smallest {
		hasAll := true
		for _, term := range terms {
			if term == smallestTerm {
				continue
			}
			ids := ti.terms[term]
			if _, ok := slices.BinarySearch(ids, id); !ok {
				hasAll = false
				break
			}
		}
		if hasAll {
			intersected = append(intersected, id)
		}
	}

	return NewAllowIDsSorted(intersected)
}

// AllowUnion returns an AllowList of IDs that possess ANY of the given terms (OR).
func (ti *TermIndex) AllowUnion(terms ...string) AllowList {
	if len(terms) == 0 {
		return AllowList{}
	}

	ti.mu.RLock()
	defer ti.mu.RUnlock()

	var allLists [][]uint32
	totalSize := 0
	for _, term := range terms {
		if ids := ti.terms[term]; len(ids) > 0 {
			allLists = append(allLists, ids)
			totalSize += len(ids)
		}
	}

	if len(allLists) == 0 {
		return AllowList{}
	}
	if len(allLists) == 1 {
		clone := make([]uint32, len(allLists[0]))
		copy(clone, allLists[0])
		return NewAllowIDsSorted(clone)
	}

	union := make([]uint32, 0, totalSize)
	indices := make([]int, len(allLists))
	for {
		minID := ^uint32(0)
		for i, lst := range allLists {
			if indices[i] < len(lst) {
				if id := lst[indices[i]]; id < minID {
					minID = id
				}
			}
		}

		if minID == ^uint32(0) {
			break // all lists exhausted
		}

		union = append(union, minID)
		for i, lst := range allLists {
			if indices[i] < len(lst) && lst[indices[i]] == minID {
				indices[i]++
			}
		}
	}

	return NewAllowIDsSorted(union)
}

// numericEntry represents a single stored value and its associated node ID.
type numericEntry struct {
	val float64
	id  uint32
}

// RangeIndex is an embedded filter index mapping field names to sorted float64 values.
// It allows fast generation of allow-lists for numeric or time range filters.
type RangeIndex struct {
	mu     sync.RWMutex
	fields map[string][]numericEntry
}

func NewRangeIndex() *RangeIndex {
	return &RangeIndex{
		fields: make(map[string][]numericEntry),
	}
}

// Add associates a node ID with a numeric value in a specific field.
// It maintains the slice in ascending order by value, then by ID.
func (ri *RangeIndex) Add(id uint32, field string, val float64) {
	ri.mu.Lock()
	defer ri.mu.Unlock()

	entries := ri.fields[field]
	entry := numericEntry{val: val, id: id}

	idx, _ := slices.BinarySearchFunc(entries, entry, func(a, b numericEntry) int {
		if c := cmp.Compare(a.val, b.val); c != 0 {
			return c
		}
		return cmp.Compare(a.id, b.id)
	})

	entries = slices.Insert(entries, idx, entry)
	ri.fields[field] = entries
}

// Remove drops a node ID from a specific field.
func (ri *RangeIndex) Remove(id uint32, field string) {
	ri.mu.Lock()
	defer ri.mu.Unlock()

	entries := ri.fields[field]
	for i, e := range entries {
		if e.id == id {
			ri.fields[field] = slices.Delete(entries, i, i+1)
			break // Assuming 1 value per field per ID
		}
	}
}

// AllowRange returns an AllowList of IDs whose values fall in [minVal, maxVal].
func (ri *RangeIndex) AllowRange(field string, minVal, maxVal float64) AllowList {
	ri.mu.RLock()
	defer ri.mu.RUnlock()

	entries := ri.fields[field]
	if len(entries) == 0 {
		return AllowList{}
	}

	startIdx, _ := slices.BinarySearchFunc(
		entries,
		numericEntry{val: minVal},
		func(a, b numericEntry) int {
			return cmp.Compare(a.val, b.val)
		},
	)

	var ids []uint32
	for i := startIdx; i < len(entries); i++ {
		if entries[i].val > maxVal {
			break
		}
		ids = append(ids, entries[i].id)
	}

	if len(ids) == 0 {
		return AllowList{}
	}

	// IDs are sorted by (val, id) natively, but for the AllowList they must be strictly sorted by ID.
	slices.Sort(ids)
	return NewAllowIDsSorted(ids)
}
