import { NDArray } from './ndarray_core.js';
import { NDWasm, fromWasm } from './ndwasm.js';
/**
 * NDWasmDecomp: Decompositions & Solvers
 * Handles O(n^3) matrix factorizations and linear system solutions.
 * @namespace NDWasmDecomp
 */
export const NDWasmDecomp = {
    /**
     * Solves a system of linear equations: Ax = B for x.
     * Complexity: O(n^3)
     * @memberof NDWasmDecomp
     * @param {NDArray} a - Square coefficient matrix of shape [n, n].
     * @param {NDArray} b - Right-hand side matrix or vector of shape [n, k].
     * @returns {NDArray} Solution matrix x of shape [n, k].
     */
    solve(a, b) {
        if (a.shape[0] !== a.shape[1] || a.shape[0] !== b.shape[0]) {
            throw new Error("Dimension mismatch for linear solver: A must be square and match B's rows.");
        }
        const n = a.shape[0];
        const k = b.ndim === 1 ? 1 : b.shape[1];
        const suffix = NDWasm.runtime._getSuffix(a.dtype);

        return NDWasm._compute([a, b], [n, k], a.dtype, (aPtr, bPtr, outPtr) => {
            return NDWasm.runtime.exports[`SolveLinear${suffix}`](aPtr, bPtr, outPtr, n, k);
        });
    },

    /**
     * Computes the multiplicative inverse of a square matrix.
     * Complexity: O(n^3)
     * @memberof NDWasmDecomp
     * @param {NDArray} a - Square matrix to invert of shape [n, n].
     * @returns {NDArray} The inverted matrix of shape [n, n].
     */
    inv(a) {
        if (a.shape[0] !== a.shape[1]) throw new Error("Matrix must be square to invert.");
        const n = a.shape[0];
        const suffix = NDWasm.runtime._getSuffix(a.dtype);

        return NDWasm._compute([a], a.shape, a.dtype, (aPtr, outPtr) => {
            return NDWasm.runtime.exports[`Invert${suffix}`](aPtr, outPtr, n);
        });
    },

    /**
     * Computes the Singular Value Decomposition (SVD): A = U * S * V^T.
     * Complexity: O(m * n * min(m, n))
     * @memberof NDWasmDecomp
     * @param {NDArray} a - Input matrix of shape [m, n].
     * @returns {{u: NDArray, s: NDArray, v: NDArray}}
     */
    svd(a) {
        const [m, n] = a.shape;
        const k = Math.min(m, n);
        const suffix = NDWasm.runtime._getSuffix(a.dtype);

        const wa = a.toWasm(NDWasm.runtime);
        const ws = NDWasm.runtime.createBuffer(k, a.dtype);
        const wu = NDWasm.runtime.createBuffer(m * m, a.dtype);
        const wv = NDWasm.runtime.createBuffer(n * n, a.dtype);

        try {
            const status = NDWasm.runtime.exports[`SVD${suffix}`](wa.ptr, m, n, wu.ptr, ws.ptr, wv.ptr);
            if (status !== 0) throw new Error("SVD computation failed.");

            return {
                u: fromWasm(wu, [m, m], a.dtype),
                s: fromWasm(ws, [k], a.dtype),
                v: fromWasm(wv, [n, n], a.dtype)
            };
        } finally {
            [wa, ws, wu, wv].forEach(b => b.dispose());
        }
    },

    /**
     * Computes the QR decomposition: A = Q * R.
     * Complexity: O(n^3)
     * @memberof NDWasmDecomp
     * @param {NDArray} a - Input matrix of shape [m, n].
     * @returns {{q: NDArray, r: NDArray}}
     */
    qr(a) {
        const [m, n] = a.shape;
        const suffix = NDWasm.runtime._getSuffix(a.dtype);
        const wa = a.toWasm(NDWasm.runtime);
        const wq = NDWasm.runtime.createBuffer(m * m, a.dtype);
        const wr = NDWasm.runtime.createBuffer(m * n, a.dtype);

        try {
            NDWasm.runtime.exports[`QR${suffix}`](wa.ptr, m, n, wq.ptr, wr.ptr);
            return {
                q: fromWasm(wq, [m, m], a.dtype),
                r: fromWasm(wr, [m, n], a.dtype)
            };
        } finally {
            [wa, wq, wr].forEach(b => b.dispose());
        }
    },

    /**
     * Computes the Cholesky decomposition of a symmetric, positive-definite matrix: A = L * L^T.
     * Complexity: O(n^3)
     * @memberof NDWasmDecomp
     * @param {NDArray} a - Symmetric positive-definite matrix of shape [n, n].
     * @returns {NDArray} Lower triangular matrix L of shape [n, n].
     */
    cholesky(a) {
        const n = a.shape[0];
        const suffix = NDWasm.runtime._getSuffix(a.dtype);
        return NDWasm._compute([a], [n, n], a.dtype, (aPtr, outPtr) => {
            return NDWasm.runtime.exports[`Cholesky${suffix}`](aPtr, outPtr, n);
        });
    },

    /**
     * Computes the LU decomposition of a matrix: A = P * L * U.
     * The result is stored in-place in the output matrix.
     * @memberof NDWasmDecomp
     * @param {NDArray} a - Input matrix of shape [m, n].
     * @returns {NDArray} LU matrix of shape [m, n].
     */
    lu(a) {
        const [m, n] = a.shape;
        const suffix = NDWasm.runtime._getSuffix(a.dtype);
        const wa = a.toWasm(NDWasm.runtime);
        try {
            NDWasm.runtime.exports[`LU${suffix}`](wa.ptr, m, n);
            return fromWasm(wa, a.shape, a.dtype);
        } finally {
            wa.dispose();
        }
    },

    /**
     * Computes the Moore-Penrose pseudo-inverse of a matrix using SVD.
     * Complexity: O(n^3)
     * @memberof NDWasmDecomp
     * @param {NDArray} a - Input matrix of shape [m, n].
     * @returns {NDArray} Pseudo-inverted matrix of shape [n, m].
     */
    pinv(a) {
        const [m, n] = a.shape;
        const suffix = NDWasm.runtime._getSuffix(a.dtype);
        return NDWasm._compute([a], [n, m], a.dtype, (aPtr, outPtr) => {
            return NDWasm.runtime.exports[`PInverse${suffix}`](aPtr, outPtr, m, n);
        });
    },

    /**
     * Computes the determinant of a square matrix.
     * Complexity: O(n^3)
     * @memberof NDWasmDecomp
     * @param {NDArray} a - Square matrix of shape [n, n].
     * @returns {number} The determinant.
     */
    det(a) {
        if (a.shape[0] !== a.shape[1]) throw new Error("Matrix must be square.");
        const suffix = NDWasm.runtime._getSuffix(a.dtype);
        const wa = a.toWasm(NDWasm.runtime);
        try {
            return NDWasm.runtime.exports[`Det${suffix}`](wa.ptr, a.shape[0]);
        } finally {
            wa.dispose();
        }
    },

    /**
     * Computes the log-determinant for improved numerical stability.
     * Complexity: O(n^3)
     * @memberof NDWasmDecomp
     * @param {NDArray} a - Square matrix of shape [n, n].
     * @returns {{sign: number, logAbsDet: number}}
     */
    logDet(a) {
        const n = a.shape[0];
        const suffix = NDWasm.runtime._getSuffix(a.dtype);
        const wa = a.toWasm(NDWasm.runtime);
        const wDetSign = NDWasm.runtime.createBuffer(2, a.dtype);

        try {
            NDWasm.runtime.exports[`LogDet${suffix}`](wa.ptr, n, wDetSign.ptr);
            return {
                sign: wDetSign.view[1],
                logAbsDet: wDetSign.view[0]
            };
        } finally {
            [wa, wDetSign].forEach(b => b.dispose());
        }
    },

    /**
     * Computes the eigenvalues and eigenvectors of a general square matrix.
     * Eigenvalues and eigenvectors can be complex numbers.
     * The results are returned in an interleaved format where each complex number (a + bi)
     * is represented by two consecutive float64 values (a, b).
     *
     * @param {NDArray} a - Input square matrix of shape `[n, n]`. Must be float64.
     * @returns {{values: NDArray, vectors: NDArray}} An object containing:
     *   - `values`: Complex eigenvalues as an NDArray of shape `[n, 2]`, where `[i, 0]` is real and `[i, 1]` is imaginary.
     *   - `vectors`: Complex right eigenvectors as an NDArray of shape `[n, n, 2]`, where `[i, j, 0]` is real and `[i, j, 1]` is imaginary.
     *              (Note: these are column vectors, such that `A * v = lambda * v`).
     * @throws {Error} If WASM runtime is not loaded, input is not a square matrix, or input dtype is not float64.
     * @memberof NDWasmDecomp
     */
    eigen(a) {
        if (!NDWasm.runtime?.isLoaded) throw new Error("WasmRuntime not loaded.");
        if (a.shape[0] !== a.shape[1]) throw new Error("Matrix must be square for eigen decomposition.");
        if (a.dtype !== 'float64') {
            throw new Error("Eigen decomposition currently only supports 'float64' input dtype.");
        }
        const n = a.shape[0];

        let wa, weigvals, weigvecs;
        try {
            wa = a.toWasm(NDWasm.runtime);
            // Eigenvalues: n complex numbers -> n * 2 float64
            weigvals = NDWasm.runtime.createBuffer(n * 2, 'float64');
            // Eigenvectors: n*n complex numbers -> n*n * 2 float64
            weigvecs = NDWasm.runtime.createBuffer(n * n * 2, 'float64');
            
            NDWasm.runtime.exports.Eigen_F64(wa.ptr, n, weigvals.ptr, weigvecs.ptr);

            return {
                values: fromWasm(weigvals, [n, 2], 'float64'),
                vectors: fromWasm(weigvecs, [n, n, 2], 'float64')
            };
        } finally {
            [wa, weigvals, weigvecs].forEach(b => b?.dispose());
        }
    }
};
