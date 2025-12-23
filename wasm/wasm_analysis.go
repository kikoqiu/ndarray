//go:build wasm

package main

import (
	"unsafe"

	"gonum.org/v1/gonum/mat"
)

/**
 * Advanced Matrix Properties and Norms (O(n^2) to O(n^3))
 */

// MatrixNorm_F64 computes different types of matrix norms.
// normType: 1 (The maximum absolute column sum), 2 (Frobenius), Infinity (The maximum absolute row sum)
//
//go:wasmexport MatrixNorm_F64
func MatrixNorm_F64(aPtr unsafe.Pointer, r, c int32, normType float64) float64 {
	R, C := int(r), int(c)
	a := mat.NewDense(R, C, ptrToF64Slice(aPtr, r*c))
	return mat.Norm(a, normType)
}

// MatrixNorm_F32 computes matrix norms for Float32.
// normType: 1 (The maximum absolute column sum), 2 (Frobenius), Infinity (The maximum absolute row sum)
//
//go:wasmexport MatrixNorm_F32
func MatrixNorm_F32(aPtr unsafe.Pointer, r, c int32, normType float64) float32 {
	R, C := int(r), int(c)
	a32 := ptrToF32Slice(aPtr, r*c)
	a64 := make([]float64, R*C)
	for i, v := range a32 {
		a64[i] = float64(v)
	}
	a := mat.NewDense(R, C, a64)
	return float32(mat.Norm(a, normType))
}

// Cond_F64 estimates the reciprocal condition number of matrix A.
// norm: 1 (1-norm) or Infinity (Infinity norm).
//
//go:wasmexport Cond_F64
func Cond_F64(aPtr unsafe.Pointer, n int32, norm float64) float64 {
	N := int(n)
	a := mat.NewDense(N, N, ptrToF64Slice(aPtr, n*n))
	return mat.Cond(a, norm)
}

// Rank_F64 computes the rank of an M x N matrix using SVD.
// epsilon: tolerance for singular values (if <= 0, 1e-14 is used).
//
//go:wasmexport Rank_F64
func Rank_F64(aPtr unsafe.Pointer, m, n int32, epsilon float64) int32 {
	M, N := int(m), int(n)
	a := mat.NewDense(M, N, ptrToF64Slice(aPtr, m*n))
	var svd mat.SVD
	if ok := svd.Factorize(a, mat.SVDThin); !ok {
		return -1
	}
	if epsilon <= 0 {
		epsilon = 1e-14
	}
	return int32(svd.Rank(epsilon))
}

/**
 * Specialized Eigenvalue Problems
 */

// EigenSym_F64 computes eigenvalues and vectors for symmetric matrices.
// Returns: 0 on success, 1 on failure.
//
//go:wasmexport EigenSym_F64
func EigenSym_F64(aPtr unsafe.Pointer, n int32, valPtr, vecPtr unsafe.Pointer) int32 {
	N := int(n)
	a := mat.NewSymDense(N, ptrToF64Slice(aPtr, n*n))
	var es mat.EigenSym
	if ok := es.Factorize(a, vecPtr != nil); !ok {
		return 1
	}

	es.Values(ptrToF64Slice(valPtr, n))

	if vecPtr != nil {
		vData := ptrToF64Slice(vecPtr, n*n)
		v := mat.NewDense(N, N, vData)
		es.VectorsTo(v)
	}
	return 0
}

/**
 * Structural Operations
 */

// Kronecker_F64 computes the Kronecker product C = A ⊗ B.
// A(ma, na), B(mb, nb) -> C(ma*mb, na*nb).
//
//go:wasmexport Kronecker_F64
func Kronecker_F64(aPtr, bPtr, outPtr unsafe.Pointer, ma, na, mb, nb int32) {
	MA, NA, MB, NB := int(ma), int(na), int(mb), int(nb)
	a := ptrToF64Slice(aPtr, ma*na)
	b := ptrToF64Slice(bPtr, mb*nb)
	out := ptrToF64Slice(outPtr, (ma*mb)*(na*nb))

	// Implementation of O(ma*na*mb*nb) loop
	for i := 0; i < MA; i++ {
		for j := 0; j < NA; j++ {
			va := a[i*NA+j]
			for ib := 0; ib < MB; ib++ {
				for jb := 0; jb < NB; jb++ {
					// Index in result matrix C
					row := i*MB + ib
					col := j*NB + jb
					out[row*(NA*NB)+col] = va * b[ib*NB+jb]
				}
			}
		}
	}
}

// Kronecker_F32 computes the Kronecker product for Float32.
//
//go:wasmexport Kronecker_F32
func Kronecker_F32(aPtr, bPtr, outPtr unsafe.Pointer, ma, na, mb, nb int32) {
	MA, NA, MB, NB := int(ma), int(na), int(mb), int(nb)
	a := ptrToF32Slice(aPtr, ma*na)
	b := ptrToF32Slice(bPtr, mb*nb)
	out := ptrToF32Slice(outPtr, (ma*mb)*(na*nb))

	for i := 0; i < MA; i++ {
		for j := 0; j < NA; j++ {
			va := a[i*NA+j]
			for ib := 0; ib < MB; ib++ {
				for jb := 0; jb < NB; jb++ {
					row := i*MB + ib
					col := j*NB + jb
					out[row*(NA*NB)+col] = va * b[ib*NB+jb]
				}
			}
		}
	}
}
