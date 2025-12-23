//go:build wasm

package main

import (
	"unsafe"

	"gonum.org/v1/gonum/mat"
)

/**
 * Matrix Factorizations and Linear Solvers (O(n^3))
 */

// LU_F64 performs LU decomposition of an M x N matrix.
//
//go:wasmexport LU_F64
func LU_F64(aPtr unsafe.Pointer, m, n int32) int32 {
	M, N := int(m), int(n)
	aData := ptrToF64Slice(aPtr, m*n)
	a := mat.NewDense(M, N, aData)

	var lu mat.LU
	lu.Factorize(a)
	return 0
}

// QR_F64 computes the QR decomposition: A = Q * R.
//
//go:wasmexport QR_F64
func QR_F64(aPtr unsafe.Pointer, m, n int32, qPtr, rPtr unsafe.Pointer) int32 {
	M, N := int(m), int(n)
	aData := ptrToF64Slice(aPtr, m*n)
	a := mat.NewDense(M, N, aData)

	var qr mat.QR
	qr.Factorize(a)

	if qPtr != nil {
		qData := ptrToF64Slice(qPtr, m*m)
		q := mat.NewDense(M, M, qData)
		qr.QTo(q)
	}
	if rPtr != nil {
		rData := ptrToF64Slice(rPtr, m*n)
		r := mat.NewDense(M, N, rData)
		qr.RTo(r)
	}
	return 0
}

// Cholesky_F64 computes the Cholesky factorization A = L * L^T.
// Matrix A must be symmetric and positive-definite.
//
//go:wasmexport Cholesky_F64
func Cholesky_F64(aPtr, lPtr unsafe.Pointer, n int32) int32 {
	N := int(n)
	aData := ptrToF64Slice(aPtr, n*n)
	a := mat.NewSymDense(N, aData)

	var chol mat.Cholesky
	if ok := chol.Factorize(a); !ok {
		return 1 // Not positive-definite
	}

	if lPtr != nil {
		lData := ptrToF64Slice(lPtr, n*n)
		lTri := mat.NewTriDense(N, mat.Lower, lData)
		chol.LTo(lTri) // Fixed: Extract L from Cholesky object
	}
	return 0
}

// SVD_F64 computes Singular Value Decomposition: A = U * S * V^T.
//
//go:wasmexport SVD_F64
func SVD_F64(aPtr unsafe.Pointer, m, n int32, uPtr, sPtr, vPtr unsafe.Pointer) int32 {
	M, N := int(m), int(n)
	aData := ptrToF64Slice(aPtr, m*n)
	a := mat.NewDense(M, N, aData)

	var svd mat.SVD
	if ok := svd.Factorize(a, mat.SVDFull); !ok {
		return 1
	}

	if sPtr != nil {
		sLen := minInt(M, N)
		sData := ptrToF64Slice(sPtr, int32(sLen))
		svd.Values(sData)
	}
	if uPtr != nil {
		uData := ptrToF64Slice(uPtr, m*m)
		u := mat.NewDense(M, M, uData)
		svd.UTo(u)
	}
	if vPtr != nil {
		vData := ptrToF64Slice(vPtr, n*n)
		v := mat.NewDense(N, N, vData)
		svd.VTo(v)
	}
	return 0
}

// SolveLinear_F64 solves the linear system Ax = B.
//
//go:wasmexport SolveLinear_F64
func SolveLinear_F64(aPtr, bPtr, outPtr unsafe.Pointer, n, k int32) int32 {
	N, K := int(n), int(k)
	a := mat.NewDense(N, N, ptrToF64Slice(aPtr, n*n))
	b := mat.NewDense(N, K, ptrToF64Slice(bPtr, n*k))
	res := mat.NewDense(N, K, ptrToF64Slice(outPtr, n*k))

	if err := res.Solve(a, b); err != nil {
		return 1
	}
	return 0
}

// Invert_F64 computes the inverse of a square matrix.
//
//go:wasmexport Invert_F64
func Invert_F64(aPtr, outPtr unsafe.Pointer, n int32) int32 {
	N := int(n)
	a := mat.NewDense(N, N, ptrToF64Slice(aPtr, n*n))
	out := mat.NewDense(N, N, ptrToF64Slice(outPtr, n*n))

	if err := out.Inverse(a); err != nil {
		return 1
	}
	return 0
}

// PInverse_F64 computes the Moore-Penrose pseudoinverse using SVD.
//
//go:wasmexport PInverse_F64
func PInverse_F64(aPtr, outPtr unsafe.Pointer, m, n int32) int32 {
	M, N := int(m), int(n)
	a := mat.NewDense(M, N, ptrToF64Slice(aPtr, m*n))
	out := mat.NewDense(N, M, ptrToF64Slice(outPtr, n*m))

	var svd mat.SVD
	if ok := svd.Factorize(a, mat.SVDFull); !ok {
		return 1
	}

	// Get U, S, and V
	u := mat.NewDense(M, M, nil)
	svd.UTo(u)
	s := svd.Values(nil)
	v := mat.NewDense(N, N, nil)
	svd.VTo(v)

	// Calculate S+
	sPlus := mat.NewDense(N, M, nil)
	tol := 1e-15 // Tolerance for non-zero singular values
	k := minInt(M, N)
	for i := 0; i < k; i++ {
		val := s[i]
		if val > tol {
			sPlus.Set(i, i, 1/val)
		}
	}

	// Calculate V * S+
	vsPlus := mat.NewDense(N, M, nil)
	vsPlus.Mul(v, sPlus)

	// Calculate (V * S+) * U^T
	out.Mul(vsPlus, u.T())

	return 0
}

// Det_F64 computes the determinant of a square matrix.
//
//go:wasmexport Det_F64
func Det_F64(aPtr unsafe.Pointer, n int32) float64 {
	N := int(n)
	a := mat.NewDense(N, N, ptrToF64Slice(aPtr, n*n))
	return mat.Det(a)
}

// LogDet_F64 computes the log-determinant (sign, logabsdet).
//
//go:wasmexport LogDet_F64
func LogDet_F64(aPtr unsafe.Pointer, n int32, signLogPtr unsafe.Pointer) {
	N := int(n)
	a := mat.NewDense(N, N, ptrToF64Slice(aPtr, n*n))
	det, sign := mat.LogDet(a)
	rst := ptrToF64Slice(signLogPtr, 2)
	rst[0] = det
	rst[1] = sign
}

// Eigen_F64 computes the eigenvalues and eigenvectors of a general square matrix.
//
//go:wasmexport Eigen_F64
func Eigen_F64(aPtr unsafe.Pointer, n int32, eigvalsPtr, eigvecsPtr unsafe.Pointer) {
	N := int(n)
	a := mat.NewDense(N, N, ptrToF64Slice(aPtr, n*n))

	var eig mat.Eigen
	if ok := eig.Factorize(a, mat.EigenRight); !ok {
		// Factorization might fail for numerically unstable matrices.
		// For now, we don't have a robust error reporting mechanism here,
		// but in a real-world scenario, a status code should be returned.
		return
	}

	// Write eigenvalues (interleaved real, imag)
	eigvalsData := ptrToF64Slice(eigvalsPtr, int32(n*2))
	vals := eig.Values(nil)
	for i, val := range vals {
		eigvalsData[i*2] = real(val)
		eigvalsData[i*2+1] = imag(val)
	}

	// Write eigenvectors if requested
	if eigvecsPtr != nil {
		eigvecsData := ptrToF64Slice(eigvecsPtr, int32(n*n*2))
		vecs := mat.NewCDense(N, N, nil)
		eig.VectorsTo(vecs)

		// Transpose and interleave the complex eigenvectors
		for i := 0; i < N; i++ {
			for j := 0; j < N; j++ {
				val := vecs.At(i, j)
				offset := (i*N + j) * 2
				eigvecsData[offset] = real(val)
				eigvecsData[offset+1] = imag(val)
			}
		}
	}
}

/**
 * Float32 Implementations (using Float64 casting)
 */

// SolveLinear_F32 performs SolveLinear with F32 input/output.
//
//go:wasmexport SolveLinear_F32
func SolveLinear_F32(aPtr, bPtr, outPtr unsafe.Pointer, n, k int32) int32 {
	N, K := int(n), int(k)
	a32 := ptrToF32Slice(aPtr, n*n)
	b32 := ptrToF32Slice(bPtr, n*k)

	a64 := make([]float64, N*N)
	for i := range a32 {
		a64[i] = float64(a32[i])
	}
	b64 := make([]float64, N*K)
	for i := range b32 {
		b64[i] = float64(b32[i])
	}

	res := mat.NewDense(N, K, nil)
	if err := res.Solve(mat.NewDense(N, N, a64), mat.NewDense(N, K, b64)); err != nil {
		return 1
	}

	out32 := ptrToF32Slice(outPtr, n*k)
	for i, v := range res.RawMatrix().Data {
		out32[i] = float32(v)
	}
	return 0
}

// Invert_F32 computes matrix inverse for Float32.
//
//go:wasmexport Invert_F32
func Invert_F32(aPtr, outPtr unsafe.Pointer, n int32) int32 {
	N := int(n)
	a32 := ptrToF32Slice(aPtr, n*n)
	a64 := make([]float64, N*N)
	for i := range a32 {
		a64[i] = float64(a32[i])
	}

	out := mat.NewDense(N, N, nil)
	if err := out.Inverse(mat.NewDense(N, N, a64)); err != nil {
		return 1
	}

	out32 := ptrToF32Slice(outPtr, n*n)
	for i, v := range out.RawMatrix().Data {
		out32[i] = float32(v)
	}
	return 0
}

// Det_F32 computes the determinant for Float32.
//
//go:wasmexport Det_F32
func Det_F32(aPtr unsafe.Pointer, n int32) float32 {
	N := int(n)
	a32 := ptrToF32Slice(aPtr, n*n)
	a64 := make([]float64, N*N)
	for i := range a32 {
		a64[i] = float64(a32[i])
	}
	return float32(mat.Det(mat.NewDense(N, N, a64)))
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
