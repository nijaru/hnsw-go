package filter

import (
	"reflect"
	"testing"

	"github.com/omendb/hnsw-go"
)

func TestTermIndex(t *testing.T) {
	ti := NewTermIndex()

	ti.Add(1, "tagA", "tagB")
	ti.Add(2, "tagA")
	ti.Add(3, "tagB", "tagC")
	ti.Add(4, "tagA", "tagB", "tagC")

	tests := []struct {
		name   string
		method string
		args   []string
		want   []uint32
	}{
		{
			name:   "single tag A",
			method: "Allow",
			args:   []string{"tagA"},
			want:   []uint32{1, 2, 4},
		},
		{
			name:   "single tag B",
			method: "Allow",
			args:   []string{"tagB"},
			want:   []uint32{1, 3, 4},
		},
		{
			name:   "single tag C",
			method: "Allow",
			args:   []string{"tagC"},
			want:   []uint32{3, 4},
		},
		{
			name:   "intersect A and B",
			method: "AllowIntersect",
			args:   []string{"tagA", "tagB"},
			want:   []uint32{1, 4},
		},
		{
			name:   "intersect B and C",
			method: "AllowIntersect",
			args:   []string{"tagB", "tagC"},
			want:   []uint32{3, 4},
		},
		{
			name:   "intersect empty",
			method: "AllowIntersect",
			args:   []string{"tagA", "tagZ"},
			want:   []uint32{},
		},
		{
			name:   "union A and C",
			method: "AllowUnion",
			args:   []string{"tagA", "tagC"},
			want:   []uint32{1, 2, 3, 4},
		},
		{
			name:   "union missing tag",
			method: "AllowUnion",
			args:   []string{"tagZ", "tagX"},
			want:   []uint32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got hnsw.AllowList
			switch tt.method {
			case "Allow":
				got = ti.Allow(tt.args[0])
			case "AllowIntersect":
				got = ti.AllowIntersect(tt.args...)
			case "AllowUnion":
				got = ti.AllowUnion(tt.args...)
			}

			if len(tt.want) == 0 {
				if got.Len() != 0 {
					t.Errorf("expected empty, got %v", got.IDs)
				}
				return
			}
			if !reflect.DeepEqual(got.IDs, tt.want) {
				t.Errorf("expected %v, got %v", tt.want, got.IDs)
			}
		})
	}

	// Test Remove
	ti.Remove(4, "tagA", "tagB", "tagC")
	got := ti.Allow("tagA")
	want := []uint32{1, 2}
	if !reflect.DeepEqual(got.IDs, want) {
		t.Errorf("after remove expected %v, got %v", want, got.IDs)
	}

	// Re-add and check sort order is maintained
	ti.Add(4, "tagA")
	got = ti.Allow("tagA")
	want = []uint32{1, 2, 4}
	if !reflect.DeepEqual(got.IDs, want) {
		t.Errorf("after re-add expected %v, got %v", want, got.IDs)
	}
}

func TestRangeIndex(t *testing.T) {
	ri := NewRangeIndex()

	ri.Add(1, "score", 1.5)
	ri.Add(2, "score", 3.0)
	ri.Add(3, "score", 4.5)
	ri.Add(4, "score", 2.0)
	ri.Add(5, "score", 5.0)

	tests := []struct {
		name string
		min  float64
		max  float64
		want []uint32
	}{
		{
			name: "full range",
			min:  1.0,
			max:  6.0,
			want: []uint32{1, 2, 3, 4, 5},
		},
		{
			name: "exact match",
			min:  3.0,
			max:  3.0,
			want: []uint32{2},
		},
		{
			name: "partial range",
			min:  2.0,
			max:  4.5,
			want: []uint32{2, 3, 4},
		},
		{
			name: "out of bounds",
			min:  10.0,
			max:  20.0,
			want: []uint32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ri.AllowRange("score", tt.min, tt.max)
			if len(tt.want) == 0 {
				if got.Len() != 0 {
					t.Errorf("expected empty, got %v", got.IDs)
				}
				return
			}
			if !reflect.DeepEqual(got.IDs, tt.want) {
				t.Errorf("expected %v, got %v", tt.want, got.IDs)
			}
		})
	}

	// Test Remove
	ri.Remove(4, "score")
	got := ri.AllowRange("score", 1.0, 5.0)
	want := []uint32{1, 2, 3, 5}
	if !reflect.DeepEqual(got.IDs, want) {
		t.Errorf("after remove expected %v, got %v", want, got.IDs)
	}

	// Test Add updating order properly
	ri.Add(6, "score", 0.5)
	got = ri.AllowRange("score", 0.0, 2.0)
	want = []uint32{1, 6}
	if !reflect.DeepEqual(got.IDs, want) {
		t.Errorf("after insert expected %v, got %v", want, got.IDs)
	}
}
