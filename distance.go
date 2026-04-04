package hnsw

import (
	"math"
)

type DistanceFunc func(a, b []float32) float32

func L2(a, b []float32) float32 {
	n := len(a)
	if n == 0 || len(b) < n {
		return 0
	}
	_ = b[n-1] // bounds check elimination
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	for ; i <= n-8; i += 8 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]
		d4 := a[i+4] - b[i+4]
		d5 := a[i+5] - b[i+5]
		d6 := a[i+6] - b[i+6]
		d7 := a[i+7] - b[i+7]
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
		s4 += d4 * d4
		s5 += d5 * d5
		s6 += d6 * d6
		s7 += d7 * d7
	}
	sum := (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
	for ; i < n; i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

func L2Two(a, b1, b2 []float32) (float32, float32) {
	n := len(a)
	if n == 0 || len(b1) < n || len(b2) < n {
		return 0, 0
	}
	_ = b1[n-1]
	_ = b2[n-1]

	var s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 float32
	var s2_0, s2_1, s2_2, s2_3, s2_4, s2_5, s2_6, s2_7 float32

	i := 0
	for ; i <= n-8; i += 8 {
		q0 := a[i]
		q1 := a[i+1]
		q2 := a[i+2]
		q3 := a[i+3]
		q4 := a[i+4]
		q5 := a[i+5]
		q6 := a[i+6]
		q7 := a[i+7]

		d1_0 := q0 - b1[i]
		d1_1 := q1 - b1[i+1]
		d1_2 := q2 - b1[i+2]
		d1_3 := q3 - b1[i+3]
		d1_4 := q4 - b1[i+4]
		d1_5 := q5 - b1[i+5]
		d1_6 := q6 - b1[i+6]
		d1_7 := q7 - b1[i+7]

		d2_0 := q0 - b2[i]
		d2_1 := q1 - b2[i+1]
		d2_2 := q2 - b2[i+2]
		d2_3 := q3 - b2[i+3]
		d2_4 := q4 - b2[i+4]
		d2_5 := q5 - b2[i+5]
		d2_6 := q6 - b2[i+6]
		d2_7 := q7 - b2[i+7]

		s1_0 += d1_0 * d1_0
		s1_1 += d1_1 * d1_1
		s1_2 += d1_2 * d1_2
		s1_3 += d1_3 * d1_3
		s1_4 += d1_4 * d1_4
		s1_5 += d1_5 * d1_5
		s1_6 += d1_6 * d1_6
		s1_7 += d1_7 * d1_7

		s2_0 += d2_0 * d2_0
		s2_1 += d2_1 * d2_1
		s2_2 += d2_2 * d2_2
		s2_3 += d2_3 * d2_3
		s2_4 += d2_4 * d2_4
		s2_5 += d2_5 * d2_5
		s2_6 += d2_6 * d2_6
		s2_7 += d2_7 * d2_7
	}

	sum1 := (s1_0 + s1_1) + (s1_2 + s1_3) + (s1_4 + s1_5) + (s1_6 + s1_7)
	sum2 := (s2_0 + s2_1) + (s2_2 + s2_3) + (s2_4 + s2_5) + (s2_6 + s2_7)

	for ; i < n; i++ {
		q := a[i]
		diff1 := q - b1[i]
		diff2 := q - b2[i]
		sum1 += diff1 * diff1
		sum2 += diff2 * diff2
	}

	return sum1, sum2
}

func Cosine(a, b []float32) float32 {
	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 1.0
	}
	sim := dot / float32(math.Sqrt(float64(normA)*float64(normB)))
	return 1.0 - sim
}

func Dot(a, b []float32) float32 {
	var dot float32
	for i := range a {
		dot += a[i] * b[i]
	}
	return -dot
}
