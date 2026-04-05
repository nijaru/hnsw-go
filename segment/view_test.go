package segment

import (
	"testing"

	"github.com/omendb/hnsw-go"
)

func TestSegmentViewStableIDMapping(t *testing.T) {
	head := newSegmentBinding(&hnsw.Index{}, []uint32{0, 1, 2})
	frozen := newSegmentBinding(&hnsw.Index{}, []uint32{3, 4})

	catalog := newSegmentCatalog()
	view, err := catalog.publish(head, frozen)
	if err != nil {
		t.Fatalf("publish failed: %v", err)
	}

	segment, local, ok := view.resolve(4)
	if !ok {
		t.Fatal("expected global id 4 to resolve")
	}
	if segment != 1 || local != 1 {
		t.Fatalf("global id 4 resolved to segment=%d local=%d", segment, local)
	}

	global, ok := view.globalID(1, 1)
	if !ok {
		t.Fatal("expected frozen segment local id 1 to resolve")
	}
	if global != 4 {
		t.Fatalf("segment 1 local 1 mapped to %d, want 4", global)
	}
}

func TestSegmentViewPublicationIsAtomic(t *testing.T) {
	head1 := newSegmentBinding(&hnsw.Index{}, []uint32{0, 1})
	frozen := newSegmentBinding(&hnsw.Index{}, []uint32{2})

	catalog := newSegmentCatalog()
	previous, err := catalog.publish(head1, frozen)
	if err != nil {
		t.Fatalf("initial publish failed: %v", err)
	}

	head2 := newSegmentBinding(&hnsw.Index{}, []uint32{3, 4})
	current, err := catalog.publish(head2, frozen)
	if err != nil {
		t.Fatalf("second publish failed: %v", err)
	}

	if previous == current {
		t.Fatal("expected a new immutable view on republish")
	}

	if _, _, ok := previous.resolve(3); ok {
		t.Fatal("old snapshot unexpectedly resolved a new head id")
	}

	segment, local, ok := current.resolve(3)
	if !ok {
		t.Fatal("new snapshot did not resolve a new head id")
	}
	if segment != 0 || local != 0 {
		t.Fatalf("new head id resolved to segment=%d local=%d", segment, local)
	}
}

func TestSegmentViewRejectsDuplicateGlobalIDs(t *testing.T) {
	head := newSegmentBinding(&hnsw.Index{}, []uint32{0, 1})
	frozen := newSegmentBinding(&hnsw.Index{}, []uint32{1, 2})

	if _, err := newSegmentView(1, head, frozen); err == nil {
		t.Fatal("expected duplicate global ids to be rejected")
	}
}
