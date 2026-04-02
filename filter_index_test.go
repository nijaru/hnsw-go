package hnsw

import (
	"reflect"
	"testing"
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
			var got AllowList
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
