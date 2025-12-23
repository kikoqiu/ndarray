//go:build wasm

package main

import (
	"unsafe"

	"gonum.org/v1/gonum/dsp/fourier"
)

/**
 * 1D Fourier Transformations (O(n log n))
 * Gonum dsp/fourier natively supports float64 only.
 */

// FFT1D_F64 performs a 1D Complex-to-Complex Fast Fourier Transform.
//
//go:wasmexport FFT1D_F64
func FFT1D_F64(ptr unsafe.Pointer, n int32) {
	N := int(n)
	coeffs := unsafe.Slice((*complex128)(ptr), N)

	fft := fourier.NewCmplxFFT(N)
	fft.Coefficients(coeffs, coeffs)
}

// IFFT1D_F64 performs a 1D Inverse Complex-to-Complex Fast Fourier Transform.
//
//go:wasmexport IFFT1D_F64
func IFFT1D_F64(ptr unsafe.Pointer, n int32) {
	N := int(n)
	coeffs := unsafe.Slice((*complex128)(ptr), N)

	fft := fourier.NewCmplxFFT(N)
	fft.Sequence(coeffs, coeffs)

	scale := 1.0 / float64(N)
	for i := range coeffs {
		coeffs[i] *= complex(scale, 0)
	}
}

// RFFT1D_F64 performs a 1D Real-to-Complex Fast Fourier Transform.
// outPtr must have length (n/2 + 1) * 2.
//
//go:wasmexport RFFT1D_F64
func RFFT1D_F64(inPtr, outPtr unsafe.Pointer, n int32) {
	N := int(n)
	in := unsafe.Slice((*float64)(inPtr), N)
	outLen := N/2 + 1
	out := unsafe.Slice((*complex128)(outPtr), outLen)

	fft := fourier.NewFFT(N)
	fft.Coefficients(out, in)
}

// RIFFT1D_F64 performs a 1D Complex-to-Real Inverse Fast Fourier Transform.
// inPtr must have length (n/2 + 1) * 2.
//
//go:wasmexport RIFFT1D_F64
func RIFFT1D_F64(inPtr, outPtr unsafe.Pointer, n int32) {
	N := int(n)
	inLen := N/2 + 1
	coeffs := unsafe.Slice((*complex128)(inPtr), inLen)
	out := unsafe.Slice((*float64)(outPtr), N)

	fft := fourier.NewFFT(N)
	fft.Sequence(out, coeffs)

	scale := 1.0 / float64(N)
	for i := range out {
		out[i] *= scale
	}
}

/**
 * 2D Fourier Transformations (O(N^2 log N))
 */

// FFT2D_F64 performs a 2D Complex-to-Complex FFT using Row-Column decomposition.
//
//go:wasmexport FFT2D_F64
func FFT2D_F64(ptr unsafe.Pointer, rows, cols int32) {
	R, C := int(rows), int(cols)
	data := unsafe.Slice((*complex128)(ptr), R*C)

	// Transform Rows
	fftC := fourier.NewCmplxFFT(C)
	for i := 0; i < R; i++ {
		offset := i * C
		row := data[offset : offset+C]
		fftC.Coefficients(row, row)
	}

	// Transform Columns
	fftR := fourier.NewCmplxFFT(R)
	colBuf := make([]complex128, R)
	for j := 0; j < C; j++ {
		// Extract column
		for i := 0; i < R; i++ {
			colBuf[i] = data[i*C+j]
		}
		// Transform
		fftR.Coefficients(colBuf, colBuf)
		// Place back into data
		for i := 0; i < R; i++ {
			data[i*C+j] = colBuf[i]
		}
	}
}

// IFFT2D_F64 performs a 2D Inverse Complex-to-Complex FFT.
//
//go:wasmexport IFFT2D_F64
func IFFT2D_F64(ptr unsafe.Pointer, rows, cols int32) {
	R, C := int(rows), int(cols)
	data := unsafe.Slice((*complex128)(ptr), R*C)

	// Inverse Columns
	fftR := fourier.NewCmplxFFT(R)
	colBuf := make([]complex128, R)
	for j := 0; j < C; j++ {
		// Extract column
		for i := 0; i < R; i++ {
			colBuf[i] = data[i*C+j]
		}
		// Transform
		fftR.Sequence(colBuf, colBuf)
		// Place back into data
		for i := 0; i < R; i++ {
			data[i*C+j] = colBuf[i]
		}
	}

	// Inverse Rows and Scale
	fftC := fourier.NewCmplxFFT(C)
	scale := complex(1.0/float64(R*C), 0)
	for i := 0; i < R; i++ {
		offset := i * C
		row := data[offset : offset+C]
		fftC.Sequence(row, row)
		for j := 0; j < C; j++ {
			row[j] *= scale
		}
	}
}

/**
 * Other Transformations
 */

// DCT_F64 performs a Discrete Cosine Transform (Type II).
//
//go:wasmexport DCT_F64
func DCT_F64(inPtr, outPtr unsafe.Pointer, n int32) {
	N := int(n)
	in := ptrToF64Slice(inPtr, n)
	out := ptrToF64Slice(outPtr, n)
	dct := fourier.NewDCT(N)
	dct.Transform(out, in)
}

/**
 * Spatial Operations (O(N^2 * K^2))
 * Manually implemented, supporting both F64 and F32.
 */

// Conv2D_F64 performs 2D spatial convolution.
//
//go:wasmexport Conv2D_F64
func Conv2D_F64(inPtr, kernelPtr, outPtr unsafe.Pointer, h, w, kh, kw, stride, padding int32) {
	H, W, KH, KW, S, P := int(h), int(w), int(kh), int(kw), int(stride), int(padding)
	img := ptrToF64Slice(inPtr, h*w)
	kernel := ptrToF64Slice(kernelPtr, kh*kw)
	outH := (H-KH+2*P)/S + 1
	outW := (W-KW+2*P)/S + 1
	out := ptrToF64Slice(outPtr, int32(outH*outW))

	for i := 0; i < outH; i++ {
		for j := 0; j < outW; j++ {
			var sum float64
			for ki := 0; ki < KH; ki++ {
				for kj := 0; kj < KW; kj++ {
					ii, jj := i*S+ki-P, j*S+kj-P
					if ii >= 0 && ii < H && jj >= 0 && jj < W {
						// Note: Convolution involves flipping the kernel;
						// for correlation, use kernel[ki*KW+kj]
						sum += img[ii*W+jj] * kernel[(KH-1-ki)*KW+(KW-1-kj)]
					}
				}
			}
			out[i*outW+j] = sum
		}
	}
}

// Conv2D_F32 performs 2D spatial convolution for float32.
//
//go:wasmexport Conv2D_F32
func Conv2D_F32(inPtr, kernelPtr, outPtr unsafe.Pointer, h, w, kh, kw, stride, padding int32) {
	H, W, KH, KW, S, P := int(h), int(w), int(kh), int(kw), int(stride), int(padding)
	img := ptrToF32Slice(inPtr, h*w)
	kernel := ptrToF32Slice(kernelPtr, kh*kw)
	outH := (H-KH+2*P)/S + 1
	outW := (W-KW+2*P)/S + 1
	out := ptrToF32Slice(outPtr, int32(outH*outW))

	for i := 0; i < outH; i++ {
		for j := 0; j < outW; j++ {
			var sum float32
			for ki := 0; ki < KH; ki++ {
				for kj := 0; kj < KW; kj++ {
					ii, jj := i*S+ki-P, j*S+kj-P
					if ii >= 0 && ii < H && jj >= 0 && jj < W {
						sum += img[ii*W+jj] * kernel[(KH-1-ki)*KW+(KW-1-kj)]
					}
				}
			}
			out[i*outW+j] = sum
		}
	}
}

// CrossCorrelate2D_F64 performs 2D cross-correlation (no kernel flip).
//
//go:wasmexport CrossCorrelate2D_F64
func CrossCorrelate2D_F64(inPtr, kernelPtr, outPtr unsafe.Pointer, h, w, kh, kw, stride, padding int32) {
	H, W, KH, KW, S, P := int(h), int(w), int(kh), int(kw), int(stride), int(padding)
	img := ptrToF64Slice(inPtr, h*w)
	kernel := ptrToF64Slice(kernelPtr, kh*kw)
	outH := (H-KH+2*P)/S + 1
	outW := (W-KW+2*P)/S + 1
	out := ptrToF64Slice(outPtr, int32(outH*outW))

	for i := 0; i < outH; i++ {
		for j := 0; j < outW; j++ {
			var sum float64
			for ki := 0; ki < KH; ki++ {
				for kj := 0; kj < KW; kj++ {
					ii, jj := i*S+ki-P, j*S+kj-P
					if ii >= 0 && ii < H && jj >= 0 && jj < W {
						sum += img[ii*W+jj] * kernel[ki*KW+kj]
					}
				}
			}
			out[i*outW+j] = sum
		}
	}
}

// CrossCorrelate2D_F32 performs 2D cross-correlation for float32.
//
//go:wasmexport CrossCorrelate2D_F32
func CrossCorrelate2D_F32(inPtr, kernelPtr, outPtr unsafe.Pointer, h, w, kh, kw, stride, padding int32) {
	H, W, KH, KW, S, P := int(h), int(w), int(kh), int(kw), int(stride), int(padding)
	img := ptrToF32Slice(inPtr, h*w)
	kernel := ptrToF32Slice(kernelPtr, kh*kw)
	outH := (H-KH+2*P)/S + 1
	outW := (W-KW+2*P)/S + 1
	out := ptrToF32Slice(outPtr, int32(outH*outW))

	for i := 0; i < outH; i++ {
		for j := 0; j < outW; j++ {
			var sum float32
			for ki := 0; ki < KH; ki++ {
				for kj := 0; kj < KW; kj++ {
					ii, jj := i*S+ki-P, j*S+kj-P
					if ii >= 0 && ii < H && jj >= 0 && jj < W {
						sum += img[ii*W+jj] * kernel[ki*KW+kj]
					}
				}
			}
			out[i*outW+j] = sum
		}
	}
}
