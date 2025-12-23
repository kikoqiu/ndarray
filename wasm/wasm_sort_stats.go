//go:build wasm

package main

import (
	"math"
	"sort"
	"unsafe"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

/**
 * Sorting and Indexing (O(N log K) or O(N))
 */

// ArgSort_F64 returns the indices that would sort the array.
//
//go:wasmexport ArgSort_F64
func ArgSort_F64(inPtr, idxPtr unsafe.Pointer, n int32) {
	data := ptrToF64Slice(inPtr, n)
	idx := ptrToI32Slice(idxPtr, n)

	for i := range idx {
		idx[i] = int32(i)
	}

	sort.SliceStable(idx, func(i, j int) bool {
		return data[idx[i]] < data[idx[j]]
	})
}

// TopK_F64 finds the largest or smallest K elements and their indices.
// largest: 1 for largest, 0 for smallest.
//
//go:wasmexport TopK_F64
func TopK_F64(inPtr, valOutPtr, idxOutPtr unsafe.Pointer, n, k int32, largest int32) {
	N, K := int(n), int(k)
	data := ptrToF64Slice(inPtr, n)
	valOut := ptrToF64Slice(valOutPtr, k)
	idxOut := ptrToI32Slice(idxOutPtr, k)

	type pair struct {
		val float64
		idx int32
	}
	items := make([]pair, N)
	for i := 0; i < N; i++ {
		items[i] = pair{data[i], int32(i)}
	}

	if largest == 1 {
		sort.Slice(items, func(i, j int) bool { return items[i].val > items[j].val })
	} else {
		sort.Slice(items, func(i, j int) bool { return items[i].val < items[j].val })
	}

	for i := 0; i < K; i++ {
		valOut[i] = items[i].val
		idxOut[i] = items[i].idx
	}
}

/**
 * High-Dimensional Statistics (O(N * D^2))
 */

// Covariance_F64 computes the D x D Covariance matrix.
//
// Parameters:
// - inPtr: Pointer to the input data (N rows * D columns)
// - outPtr: Pointer to the output buffer (D * D elements)
// - n: Number of observations (rows)
// - d: Number of variables (columns)
//
//go:wasmexport Covariance_F64
func Covariance_F64(inPtr, outPtr unsafe.Pointer, n, d int32) {
	N, D := int(n), int(d)
	data := ptrToF64Slice(inPtr, n*d)
	out := ptrToF64Slice(outPtr, d*d)

	// 1. Wrap the input data as a Row-Major Dense matrix (N x D).
	m := mat.NewDense(N, D, data)

	// 2. Use a temporary SymDense for the calculation.
	// Passing 'nil' for storage lets Gonum allocate temporary memory,
	// ensuring the calculation is isolated from the output buffer.
	tmpCov := mat.NewSymDense(D, nil)
	stat.CovarianceMatrix(tmpCov, m, nil)

	// 3. Populate the output buffer as a full D x D matrix.
	// We wrap the output pointer as a 'mat.Dense'.
	// The Copy() method ensures that both the upper and lower triangles
	// are filled, resulting in a complete D*D matrix for JS/NumPy.
	res := mat.NewDense(D, D, out)
	res.Copy(tmpCov)
}

// Correlation_F64 computes the D x D Pearson correlation matrix.
//
// Parameters:
// - inPtr: Pointer to the input data (N rows * D columns)
// - outPtr: Pointer to the output buffer (D * D elements)
// - n: Number of observations (rows)
// - d: Number of variables (columns)
//
//go:wasmexport Correlation_F64
func Correlation_F64(inPtr, outPtr unsafe.Pointer, n, d int32) {
	N, D := int(n), int(d)
	data := ptrToF64Slice(inPtr, n*d)
	out := ptrToF64Slice(outPtr, d*d)

	// 1. Wrap input as N x D matrix.
	m := mat.NewDense(N, D, data)

	// 2. Calculate correlation into a temporary symmetric matrix.
	tmpCorr := mat.NewSymDense(D, nil)
	stat.CorrelationMatrix(tmpCorr, m, nil)

	// 3. Export the result to the output pointer.
	// By copying SymDense into a standard Dense matrix, we ensure the
	// underlying 'out' slice is fully populated with D*D values.
	res := mat.NewDense(D, D, out)
	res.Copy(tmpCorr)
}

/**
 * Spatial Geometry and Distances (O(M * N * D))
 */

// PairwiseDist_F64 computes the Euclidean distance between two sets of vectors.
// A is M x D, B is N x D, result is M x N.
//
//go:wasmexport PairwiseDist_F64
func PairwiseDist_F64(aPtr, bPtr, outPtr unsafe.Pointer, m, n, d int32) {
	M, N, D := int(m), int(n), int(d)
	a := ptrToF64Slice(aPtr, m*d)
	b := ptrToF64Slice(bPtr, n*d)
	out := ptrToF64Slice(outPtr, m*n)

	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var sum float64
			for k := 0; k < D; k++ {
				diff := a[i*D+k] - b[j*D+k]
				sum += diff * diff
			}
			out[i*N+j] = math.Sqrt(sum)
		}
	}
}

// PairwiseDist_F32 performs the distance calculation for float32.
//
//go:wasmexport PairwiseDist_F32
func PairwiseDist_F32(aPtr, bPtr, outPtr unsafe.Pointer, m, n, d int32) {
	M, N, D := int(m), int(n), int(d)
	a := ptrToF32Slice(aPtr, m*d)
	b := ptrToF32Slice(bPtr, n*d)
	out := ptrToF32Slice(outPtr, m*n)

	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var sum float32
			for k := 0; k < D; k++ {
				diff := a[i*D+k] - b[j*D+k]
				sum += diff * diff
			}
			out[i*N+j] = float32(math.Sqrt(float64(sum)))
		}
	}
}

/**
 * Iterative Algorithms
 */

// KMeans_F64 performs K-Means clustering.
// data: N x D, centroids: K x D (initial centroids provided as input).
// Returns: Number of iterations performed.
//
//go:wasmexport KMeans_F64
func KMeans_F64(dataPtr, centroidsPtr, labelsPtr unsafe.Pointer, n, d, k, maxIter int32) int32 {
	N, D, K, MaxI := int(n), int(d), int(k), int(maxIter)
	data := ptrToF64Slice(dataPtr, n*d)
	centroids := ptrToF64Slice(centroidsPtr, k*d)
	labels := ptrToI32Slice(labelsPtr, n)

	counts := make([]int, K)
	newCentroids := make([]float64, K*D)

	for iter := 0; iter < MaxI; iter++ {
		changed := false

		// 1. Assignment Step
		for i := 0; i < N; i++ {
			minDist := math.MaxFloat64
			var bestK int32 = 0
			for c := 0; c < K; c++ {
				distSq := 0.0
				for dim := 0; dim < D; dim++ {
					diff := data[i*D+dim] - centroids[c*D+dim]
					distSq += diff * diff
				}
				if distSq < minDist {
					minDist = distSq
					bestK = int32(c)
				}
			}
			if labels[i] != bestK {
				labels[i] = bestK
				changed = true
			}
		}

		if !changed && iter > 0 {
			return int32(iter)
		}

		// 2. Update Step
		for i := range newCentroids {
			newCentroids[i] = 0
		}
		for i := range counts {
			counts[i] = 0
		}

		for i := 0; i < N; i++ {
			c := int(labels[i])
			counts[c]++
			for dim := 0; dim < D; dim++ {
				newCentroids[c*D+dim] += data[i*D+dim]
			}
		}

		for c := 0; c < K; c++ {
			if counts[c] > 0 {
				for dim := 0; dim < D; dim++ {
					centroids[c*D+dim] = newCentroids[c*D+dim] / float64(counts[c])
				}
			}
		}
	}
	return maxIter
}

// KMeans_F32 performs K-Means clustering for float32.
//
//go:wasmexport KMeans_F32
func KMeans_F32(dataPtr, centroidsPtr, labelsPtr unsafe.Pointer, n, d, k, maxIter int32) int32 {
	N, D, K, MaxI := int(n), int(d), int(k), int(maxIter)
	data := ptrToF32Slice(dataPtr, n*d)
	centroids := ptrToF32Slice(centroidsPtr, k*d)
	labels := ptrToI32Slice(labelsPtr, n)

	counts := make([]int, K)
	newCentroids := make([]float32, K*D)

	for iter := 0; iter < MaxI; iter++ {
		changed := false
		for i := 0; i < N; i++ {
			minDist := float32(math.MaxFloat32)
			var bestK int32 = 0
			for c := 0; c < K; c++ {
				distSq := float32(0.0)
				for dim := 0; dim < D; dim++ {
					diff := data[i*D+dim] - centroids[c*D+dim]
					distSq += diff * diff
				}
				if distSq < minDist {
					minDist = distSq
					bestK = int32(c)
				}
			}
			if labels[i] != bestK {
				labels[i] = bestK
				changed = true
			}
		}
		if !changed && iter > 0 {
			return int32(iter)
		}

		for i := range newCentroids {
			newCentroids[i] = 0
		}
		for i := range counts {
			counts[i] = 0
		}
		for i := 0; i < N; i++ {
			c := int(labels[i])
			counts[c]++
			for dim := 0; dim < D; dim++ {
				newCentroids[c*D+dim] += data[i*D+dim]
			}
		}
		for c := 0; c < K; c++ {
			if counts[c] > 0 {
				for dim := 0; dim < D; dim++ {
					centroids[c*D+dim] = newCentroids[c*D+dim] / float32(counts[c])
				}
			}
		}
	}
	return maxIter
}
