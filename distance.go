package hnsw

import (
	"math"
)

// DistanceFunc is a function type for calculating distance between two vectors.
type DistanceFunc func(a, b []float32) float32

// L2 calculates the Euclidean distance between two vectors.
// We use a manually unrolled loop with multiple accumulators to break
// dependency chains and maximize instruction-level parallelism (ILP).
func L2(a, b []float32) float32 {
	var s0, s1, s2, s3 float32
	n := len(a)
	i := 0
	for ; i <= n-4; i += 4 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
	}
	sum := s0 + s1 + s2 + s3
	// Remainder
	for ; i < n; i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

// Cosine calculates the Cosine similarity between two vectors.
// It returns a value where smaller is "closer" (1 - cosine).
func Cosine(a, b []float32) float32 {
	var dot, magA, magB float32
	for i := range a {
		dot += a[i] * b[i]
		magA += a[i] * a[i]
		magB += b[i] * b[i]
	}
	if magA == 0 || magB == 0 {
		return 1.0
	}
	// Return 1 - cosine to keep "distance" semantics (smaller is closer).
	return 1.0 - (dot / (float32(math.Sqrt(float64(magA))) * float32(math.Sqrt(float64(magB)))))
}

// Dot calculates the dot product between two vectors.
// We return its negative to keep "distance" semantics (smaller is closer).
func Dot(a, b []float32) float32 {
	var dot float32
	for i := range a {
		dot += a[i] * b[i]
	}
	return -dot
}
