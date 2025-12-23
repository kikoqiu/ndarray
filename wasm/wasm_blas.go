//go:build wasm

package main

import (
	"unsafe"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/blas/gonum"
	"gonum.org/v1/gonum/mat"
)

// init initializes the Gonum BLAS implementation.
func init() {
	impl := gonum.Implementation{}
	blas64.Use(impl)
	blas32.Use(impl)
}

/**
 * BLAS Level 3 Operations (Matrix-Matrix, O(n^3))
 */

// MatMul_F64 performs C = A * B.
// m, n, k are dimensions for A(m*n) and B(n*k).
//
//go:wasmexport MatMul_F64
func MatMul_F64(aPtr, bPtr, outPtr unsafe.Pointer, m, n, k int32) {
	M, N, K := int(m), int(n), int(k)
	aData := ptrToF64Slice(aPtr, m*n)
	bData := ptrToF64Slice(bPtr, n*k)
	cData := ptrToF64Slice(outPtr, m*k)

	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1.0,
		blas64.General{Rows: M, Cols: N, Data: aData, Stride: N},
		blas64.General{Rows: N, Cols: K, Data: bData, Stride: K},
		0.0,
		blas64.General{Rows: M, Cols: K, Data: cData, Stride: K},
	)
}

// MatMul_F32 performs C = A * B.
//
//go:wasmexport MatMul_F32
func MatMul_F32(aPtr, bPtr, outPtr unsafe.Pointer, m, n, k int32) {
	M, N, K := int(m), int(n), int(k)
	aData := ptrToF32Slice(aPtr, m*n)
	bData := ptrToF32Slice(bPtr, n*k)
	cData := ptrToF32Slice(outPtr, m*k)

	blas32.Gemm(blas.NoTrans, blas.NoTrans, 1.0,
		blas32.General{Rows: M, Cols: N, Data: aData, Stride: N},
		blas32.General{Rows: N, Cols: K, Data: bData, Stride: K},
		0.0,
		blas32.General{Rows: M, Cols: K, Data: cData, Stride: K},
	)
}

// MatMulBatch_F64 performs batch matrix multiplications.
//
//go:wasmexport MatMulBatch_F64
func MatMulBatch_F64(aPtr, bPtr, outPtr unsafe.Pointer, batchSize, m, n, k int32) {
	M, N, K := int(m), int(n), int(k)
	aStride, bStride, cStride := M*N, N*K, M*K

	for i := 0; i < int(batchSize); i++ {
		a := ptrToF64Slice(unsafe.Add(aPtr, uintptr(i*aStride*8)), m*n)
		b := ptrToF64Slice(unsafe.Add(bPtr, uintptr(i*bStride*8)), n*k)
		c := ptrToF64Slice(unsafe.Add(outPtr, uintptr(i*cStride*8)), m*k)

		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1.0,
			blas64.General{Rows: M, Cols: N, Data: a, Stride: N},
			blas64.General{Rows: N, Cols: K, Data: b, Stride: K},
			0.0,
			blas64.General{Rows: M, Cols: K, Data: c, Stride: K},
		)
	}
}

// MatMulBatch_F32 performs batch matrix multiplications.
//
//go:wasmexport MatMulBatch_F32
func MatMulBatch_F32(aPtr, bPtr, outPtr unsafe.Pointer, batchSize, m, n, k int32) {
	M, N, K := int(m), int(n), int(k)
	aStride, bStride, cStride := M*N, N*K, M*K

	for i := 0; i < int(batchSize); i++ {
		a := ptrToF32Slice(unsafe.Add(aPtr, uintptr(i*aStride*4)), m*n)
		b := ptrToF32Slice(unsafe.Add(bPtr, uintptr(i*bStride*4)), n*k)
		c := ptrToF32Slice(unsafe.Add(outPtr, uintptr(i*cStride*4)), m*k)

		blas32.Gemm(blas.NoTrans, blas.NoTrans, 1.0,
			blas32.General{Rows: M, Cols: N, Data: a, Stride: N},
			blas32.General{Rows: N, Cols: K, Data: b, Stride: K},
			0.0,
			blas32.General{Rows: M, Cols: K, Data: c, Stride: K},
		)
	}
}

/**
 * Matrix Functions (O(n^3))
 */

// MatrixPower_F64 computes A^k (Matrix Power).
// Returns: 0 on success, 1 if k < 0 or A is not square.
//
//go:wasmexport MatrixPower_F64
func MatrixPower_F64(aPtr, outPtr unsafe.Pointer, n, k int32) int32 {
	if k < 0 {
		return 1
	}
	N := int(n)
	a := mat.NewDense(N, N, ptrToF64Slice(aPtr, n*n))
	out := mat.NewDense(N, N, ptrToF64Slice(outPtr, n*n))

	// Pow panics if n < 0 or matrix is not square.
	// Since we use NewDense with N x N, it is square.
	out.Pow(a, int(k))
	return 0
}

// Syrk_F64 performs the symmetric rank-k operation: C = alpha*A*A^T + beta*C.
// Often used for calculating covariance matrices.
//
//go:wasmexport Syrk_F64
func Syrk_F64(aPtr, cPtr unsafe.Pointer, n, k int32) {
	N, K := int(n), int(k)
	a := ptrToF64Slice(aPtr, n*k)
	cData := ptrToF64Slice(cPtr, n*n)
	cMat := mat.NewDense(N, N, cData)

	// Uplo is specified in the Symmetric struct
	blas64.Syrk(blas.NoTrans, 1.0,
		blas64.General{Rows: N, Cols: K, Data: a, Stride: K},
		0.0,
		blas64.Symmetric{N: N, Data: cData, Stride: N, Uplo: blas.Upper},
	)

	// Copy upper triangle to lower triangle
	for i := 0; i < N; i++ {
		for j := i + 1; j < N; j++ {
			cMat.Set(j, i, cMat.At(i, j))
		}
	}
}

// Syrk_F32 performs the symmetric rank-k operation.
//
//go:wasmexport Syrk_F32
func Syrk_F32(aPtr, cPtr unsafe.Pointer, n, k int32) {
	N, K := int(n), int(k)
	a := ptrToF32Slice(aPtr, n*k)
	c := ptrToF32Slice(cPtr, n*n)

	blas32.Syrk(blas.NoTrans, 1.0,
		blas32.General{Rows: N, Cols: K, Data: a, Stride: K},
		0.0,
		blas32.Symmetric{N: N, Data: c, Stride: N, Uplo: blas.Upper},
	)
}

// Trsm_F64 solves the triangular matrix equation A*X = B.
//
//go:wasmexport Trsm_F64
func Trsm_F64(aPtr, bPtr unsafe.Pointer, m, n, lower int32) {
	M, N := int(m), int(n)
	aData := ptrToF64Slice(aPtr, m*m)
	bData := ptrToF64Slice(bPtr, m*n)
	uplo := blas.Upper
	if lower == 1 {
		uplo = blas.Lower
	}

	// Uplo and Diag are specified in the Triangular struct
	blas64.Trsm(blas.Left, blas.NoTrans, 1.0,
		blas64.Triangular{N: M, Data: aData, Stride: M, Uplo: uplo, Diag: blas.NonUnit},
		blas64.General{Rows: M, Cols: N, Data: bData, Stride: N},
	)
}

// Trsm_F32 solves the triangular matrix equation A*X = B.
//
//go:wasmexport Trsm_F32
func Trsm_F32(aPtr, bPtr unsafe.Pointer, m, n, lower int32) {
	M, N := int(m), int(n)
	aData := ptrToF32Slice(aPtr, m*m)
	bData := ptrToF32Slice(bPtr, m*n)
	uplo := blas.Upper
	if lower == 1 {
		uplo = blas.Lower
	}

	blas32.Trsm(blas.Left, blas.NoTrans, 1.0,
		blas32.Triangular{N: M, Data: aData, Stride: M, Uplo: uplo, Diag: blas.NonUnit},
		blas32.General{Rows: M, Cols: N, Data: bData, Stride: N},
	)
}

/**
 * BLAS Level 2 Operations (Matrix-Vector, O(n^2))
 */

// MatVecMul_F64 performs y = A * x.
//
//go:wasmexport MatVecMul_F64
func MatVecMul_F64(aPtr, xPtr, yPtr unsafe.Pointer, m, n int32) {
	M, N := int(m), int(n)
	a := ptrToF64Slice(aPtr, m*n)
	x := ptrToF64Slice(xPtr, n)
	y := ptrToF64Slice(yPtr, m)

	blas64.Gemv(blas.NoTrans, 1.0,
		blas64.General{Rows: M, Cols: N, Data: a, Stride: N},
		blas64.Vector{N: N, Inc: 1, Data: x},
		0.0,
		blas64.Vector{N: M, Inc: 1, Data: y},
	)
}

// MatVecMul_F32 performs y = A * x.
//
//go:wasmexport MatVecMul_F32
func MatVecMul_F32(aPtr, xPtr, yPtr unsafe.Pointer, m, n int32) {
	M, N := int(m), int(n)
	a := ptrToF32Slice(aPtr, m*n)
	x := ptrToF32Slice(xPtr, n)
	y := ptrToF32Slice(yPtr, m)

	blas32.Gemv(blas.NoTrans, 1.0,
		blas32.General{Rows: M, Cols: N, Data: a, Stride: N},
		blas32.Vector{N: N, Inc: 1, Data: x},
		0.0,
		blas32.Vector{N: M, Inc: 1, Data: y},
	)
}

// Ger_F64 performs the rank-1 update (outer product): A = alpha*x*y^T + A.
//
//go:wasmexport Ger_F64
func Ger_F64(xPtr, yPtr, aPtr unsafe.Pointer, m, n int32) {
	M, N := int(m), int(n)
	x := ptrToF64Slice(xPtr, m)
	y := ptrToF64Slice(yPtr, n)
	a := ptrToF64Slice(aPtr, m*n)

	blas64.Ger(1.0,
		blas64.Vector{N: M, Inc: 1, Data: x},
		blas64.Vector{N: N, Inc: 1, Data: y},
		blas64.General{Rows: M, Cols: N, Data: a, Stride: N},
	)
}

// Ger_F32 performs the rank-1 update (outer product).
//
//go:wasmexport Ger_F32
func Ger_F32(xPtr, yPtr, aPtr unsafe.Pointer, m, n int32) {
	M, N := int(m), int(n)
	x := ptrToF32Slice(xPtr, m)
	y := ptrToF32Slice(yPtr, n)
	a := ptrToF32Slice(aPtr, m*n)

	blas32.Ger(1.0,
		blas32.Vector{N: M, Inc: 1, Data: x},
		blas32.Vector{N: N, Inc: 1, Data: y},
		blas32.General{Rows: M, Cols: N, Data: a, Stride: N},
	)
}
