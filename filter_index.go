package hnsw

import (
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
		if len(ids) == 0 || ids[len(ids)-1] != id {
			// Note: expects IDs to be added in ascending order to maintain sorting.
			// The caller (like BatchInsert) naturally provides ascending IDs.
			ti.terms[term] = append(ids, id)
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
