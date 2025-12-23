import { NDArray } from './ndarray_core.js';
import { NDWasm, fromWasm } from './ndwasm.js';
/**
 * NDWasmSignal: Signal Processing & Transformations
 * Handles O(n log n) frequency domain transforms and O(n^2 * k^2) spatial filters.
 * @namespace NDWasmSignal
 */
export const NDWasmSignal = {
    /**
     * 1D Complex-to-Complex Fast Fourier Transform.
     * The input array must have its last dimension of size 2 (real and imaginary parts).
     * The transform is performed in-place.
     * Complexity: O(n log n)
     * @memberof NDWasmSignal
     * @param {NDArray} a - Complex input signal, with shape [..., 2].
     * @returns {NDArray} - Complex result, with the same shape as input.
     */
    fft(a) {
        if (a.ndim !== 2 || a.shape[1] !== 2) {
            throw new Error("Input to fft must be a 1D complex array with shape [n, 2].");
        }

        const n = a.size / 2;
        const suffix = NDWasm.runtime._getSuffix(a.dtype);
        const wComplex = a.toWasm(NDWasm.runtime);

        try {
            NDWasm.runtime.exports[`FFT1D${suffix}`](wComplex.ptr, n);
            return fromWasm(wComplex, a.shape, a.dtype);
        } finally {
            wComplex.dispose();
        }
    },

    /**
     * 1D Inverse Complex-to-Complex Fast Fourier Transform.
     * The input array must have its last dimension of size 2 (real and imaginary parts).
     * The transform is performed in-place.
     * Complexity: O(n log n)
     * @memberof NDWasmSignal
     * @param {NDArray} a - Complex frequency-domain signal, with shape [..., 2].
     * @returns {NDArray} - Complex time-domain result, with the same shape as input.
     */
    ifft(a) {
        if (a.ndim !== 2 || a.shape[1] !== 2) {
            throw new Error("Input to ifft must be a 1D complex array with shape [n, 2].");
        }
        
        const n = a.size / 2;
        const suffix = NDWasm.runtime._getSuffix(a.dtype);
        const wComplex = a.toWasm(NDWasm.runtime);

        try {
            NDWasm.runtime.exports[`IFFT1D${suffix}`](wComplex.ptr, n);
            return fromWasm(wComplex, a.shape, a.dtype);
        } finally {
            wComplex.dispose();
        }
    },

    /**
     * 1D Real-to-Complex Fast Fourier Transform (Optimized for real input).
     * The output is a complex array with shape [n/2 + 1, 2].
     * Complexity: O(n log n)
     * @memberof NDWasmSignal
     * @param {NDArray} a - Real input signal.
     * @returns {NDArray} - Complex result of shape [n/2 + 1, 2].
     */
    rfft(a) {
        if (a.ndim !== 1) {
            throw new Error("Input to rfft must be a 1D real array.");
        }
        const n = a.size;
        const outLen = Math.floor(n / 2) + 1;
        const suffix = NDWasm.runtime._getSuffix(a.dtype);
        const wa = a.toWasm(NDWasm.runtime);
        const wOut = NDWasm.runtime.createBuffer(outLen * 2, a.dtype);

        try {
            NDWasm.runtime.exports[`RFFT1D${suffix}`](wa.ptr, wOut.ptr, n);
            return fromWasm(wOut, [outLen, 2], a.dtype);
        } finally {
            wa.dispose();
            wOut.dispose();
        }
    },

    /**
     * 1D Complex-to-Real Inverse Fast Fourier Transform.
     * The input must be a complex array of shape [k, 2], where k is n/2 + 1.
     * @memberof NDWasmSignal
     * @param {NDArray} a - Complex frequency signal of shape [n/2 + 1, 2].
     * @param {number} n - Length of the original real signal.
     * @returns {NDArray} Real-valued time domain signal.
     */
    rifft(a, n) {
        if (a.ndim !== 2 || a.shape[1] !== 2) {
            throw new Error("Input to rifft must be a complex array with shape [k, 2].");
        }
        const suffix = NDWasm.runtime._getSuffix(a.dtype);
        const wa = a.toWasm(NDWasm.runtime);
        const wo = NDWasm.runtime.createBuffer(n, a.dtype);

        try {
            NDWasm.runtime.exports[`RIFFT1D${suffix}`](wa.ptr, wo.ptr, n);
            return fromWasm(wo, [n], a.dtype);
        } finally {
            wa.dispose();
            wo.dispose();
        }
    },

    /**
     * 2D Complex-to-Complex Fast Fourier Transform.
     * The input array must be 3D with shape [rows, cols, 2].
     * The transform is performed in-place.
     * Complexity: O(rows * cols * log(rows * cols))
     * @memberof NDWasmSignal
     * @param {NDArray} a - 2D Complex input signal, with shape [rows, cols, 2].
     * @returns {NDArray} - 2D Complex result, with the same shape as input.
     */
    fft2(a) {
        if (a.ndim !== 3 || a.shape[2] !== 2) {
            throw new Error("fft2 requires a 3D array with shape [rows, cols, 2].");
        }
        const [rows, cols] = a.shape;
        const suffix = NDWasm.runtime._getSuffix(a.dtype);
        const wComplex = a.toWasm(NDWasm.runtime);

        try {
            NDWasm.runtime.exports[`FFT2D${suffix}`](wComplex.ptr, rows, cols);
            return fromWasm(wComplex, a.shape, a.dtype);
        } finally {
            wComplex.dispose();
        }
    },

    /**
     * 2D Inverse Complex-to-Complex Fast Fourier Transform.
     * The input array must be 3D with shape [rows, cols, 2].
     * The transform is performed in-place.
     * @memberof NDWasmSignal
     * @param {NDArray} a - 2D Complex frequency-domain signal, with shape [rows, cols, 2].
     * @returns {NDArray} - 2D Complex time-domain result, with the same shape as input.
     */
    ifft2(a) {
        if (a.ndim !== 3 || a.shape[2] !== 2) {
            throw new Error("ifft2 requires a 3D array with shape [rows, cols, 2].");
        }
        const [rows, cols] = a.shape;
        const suffix = NDWasm.runtime._getSuffix(a.dtype);
        const wComplex = a.toWasm(NDWasm.runtime);

        try {
            NDWasm.runtime.exports[`IFFT2D${suffix}`](wComplex.ptr, rows, cols);
            return fromWasm(wComplex, a.shape, a.dtype);
        } finally {
            wComplex.dispose();
        }
    },

    /**
     * 1D Discrete Cosine Transform (Type II).
     * Complexity: O(n log n)
     * @memberof NDWasmSignal
     * @param {NDArray} a - Input signal.
     * @returns {NDArray} DCT result of same shape.
     */
    dct(a) {
        if (a.size < 2) {
            return a.copy();
        }
        const n = a.size;
        const suffix = NDWasm.runtime._getSuffix(a.dtype);
        return NDWasm._compute([a], a.shape, a.dtype, (aPtr, outPtr) => {
            return NDWasm.runtime.exports[`DCT${suffix}`](aPtr, outPtr, n);
        });
    },

    /**
     * 2D Spatial Convolution.
     * Complexity: O(img_h * img_w * kernel_h * kernel_w)
     * @memberof NDWasmSignal
     * @param {NDArray} img - 2D Image/Matrix.
     * @param {NDArray} kernel - 2D Filter kernel.
     * @param {number} stride - Step size (default 1).
     * @param {number} padding - Zero-padding size (default 0).
     * @returns {NDArray} Convolved result.
     */
    conv2d(img, kernel, stride = 1, padding = 0) {
        if (img.ndim !== 2 || kernel.ndim !== 2) throw new Error("Inputs must be 2D.");
        const [h, w] = img.shape;
        const [kh, kw] = kernel.shape;
        const oh = Math.floor((h - kh + 2 * padding) / stride) + 1;
        const ow = Math.floor((w - kw + 2 * padding) / stride) + 1;
        const suffix = NDWasm.runtime._getSuffix(img.dtype);

        return NDWasm._compute([img, kernel], [oh, ow], img.dtype, (iPtr, kPtr, outPtr) => {
            return NDWasm.runtime.exports[`Conv2D${suffix}`](iPtr, kPtr, outPtr, h, w, kh, kw, stride, padding);
        });
    },

    /**
     * 2D Spatial Cross-Correlation.
     * Similar to convolution but without flipping the kernel.
     * Complexity: O(img_h * img_w * kernel_h * kernel_w)
     * @memberof NDWasmSignal
     * @param {NDArray} img - 2D Image/Matrix.
     * @param {NDArray} kernel - 2D Filter kernel.
     * @param {number} stride - Step size.
     * @param {number} padding - Zero-padding size.
     * @returns {NDArray} Cross-correlated result.
     */
    correlate2d(img, kernel, stride = 1, padding = 0) {
        const [h, w] = img.shape;
        const [kh, kw] = kernel.shape;
        const oh = Math.floor((h - kh + 2 * padding) / stride) + 1;
        const ow = Math.floor((w - kw + 2 * padding) / stride) + 1;
        const suffix = NDWasm.runtime._getSuffix(img.dtype);

        return NDWasm._compute([img, kernel], [oh, ow], img.dtype, (iPtr, kPtr, outPtr) => {
            return NDWasm.runtime.exports[`CrossCorrelate2D${suffix}`](iPtr, kPtr, outPtr, h, w, kh, kw, stride, padding);
        });
    }
};
