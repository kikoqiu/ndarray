import { NDArray } from './ndarray_core.js';
import { NDWasm, fromWasm } from './ndwasm.js';
/**
 * NDWasmAnalysis: Advanced Analysis, Stats, Spatial & Random
 * Handles O(n log n) sorting, O(n*d^2) statistics, O(n^3) matrix properties,
 * spatial clustering, and high-performance random sampling.
 * @namespace NDWasmAnalysis
 */
export const NDWasmAnalysis = {
    // --- 1. Sorting & Searching (O(n log n)) ---

    /**
     * Returns the indices that would sort an array.
     * @memberof NDWasmAnalysis
     * @param {NDArray} a - Input array.
     * @returns {NDArray} Indices as Int32 NDArray.
     */
    argsort(a) {
        const wa = a.toWasm(NDWasm.runtime);
        const wi = NDWasm.runtime.createBuffer(a.size, 'int32');
        try {
            const suffix = NDWasm.runtime._getSuffix(a.dtype);
            NDWasm.runtime.exports[`ArgSort${suffix}`](wa.ptr, wi.ptr, a.size);
            return fromWasm(wi, a.shape, 'int32');
        } finally {
            wa.dispose();
            wi.dispose();
        }
    },

    /**
     * Finds the largest or smallest K elements and their indices.
     * Complexity: O(n log k)
     * @memberof NDWasmAnalysis
     * @param {NDArray} a - Input array.
     * @param {number} k - Number of elements to return.
     * @param {boolean} largest - If true, find max elements; else min.
     * @returns {{values: NDArray, indices: NDArray}}
     */
    topk(a, k, largest = true) {
        const wa = a.toWasm(NDWasm.runtime);
        const wv = NDWasm.runtime.createBuffer(k, a.dtype);
        const wi = NDWasm.runtime.createBuffer(k, 'int32');
        try {
            const suffix = NDWasm.runtime._getSuffix(a.dtype);
            NDWasm.runtime.exports[`TopK${suffix}`](wa.ptr, wv.ptr, wi.ptr, a.size, k, largest ? 1 : 0);
            return {
                values: fromWasm(wv, [k], a.dtype),
                indices: fromWasm(wi, [k], 'int32')
            };
        } finally {
            [wa, wv, wi].forEach(b => b.dispose());
        }
    },

    // --- 2. Statistics & Matrix Properties (O(n^2) to O(n^3)) ---

    /**
     * Computes the covariance matrix for a dataset of shape [n_samples, n_features].
     * @memberof NDWasmAnalysis
     * @param {NDArray} a - Data matrix.
     * @returns {NDArray} Covariance matrix of shape [d, d].
     */
    cov(a) {
        const [n, d] = a.shape;
        const suffix = NDWasm.runtime._getSuffix(a.dtype);
        return NDWasm._compute([a], [d, d], a.dtype, (aP, oP) => {
            return NDWasm.runtime.exports[`Covariance${suffix}`](aP, oP, n, d);
        });
    },

    /**
     * Computes the Pearson correlation matrix for a dataset of shape [n_samples, n_features].
     * @memberof NDWasmAnalysis
     * @param {NDArray} a - Data matrix.
     * @returns {NDArray} Correlation matrix of shape [d, d].
     */
    corr(a) {
        const [n, d] = a.shape;
        const suffix = NDWasm.runtime._getSuffix(a.dtype);
        return NDWasm._compute([a], [d, d], a.dtype, (aP, oP) => {
            return NDWasm.runtime.exports[`Correlation${suffix}`](aP, oP, n, d);
        });
    },

    /**
     * Computes the matrix norm.
     * @memberof NDWasmAnalysis
     * @param {NDArray} a - Input matrix.
     * @param {number} type - 1 (The maximum absolute column sum), 2 (Frobenius), Infinity (The maximum absolute row sum)
     * @returns {number} The norm value.
     */
    norm(a, type = 2) {
        const suffix = NDWasm.runtime._getSuffix(a.dtype);
        const wa = a.toWasm(NDWasm.runtime);
        try {
            return NDWasm.runtime.exports[`MatrixNorm${suffix}`](wa.ptr, a.shape[0], a.ndim ==1 ? 1 : a.shape[1], type);
        } finally {
            wa.dispose();
        }
    },

    /**
     * Computes the rank of a matrix using SVD.
     * @memberof NDWasmAnalysis
     * @param {NDArray} a - Input matrix.
     * @param {number} tol - Tolerance for singular values (0 for 1e-14).
     * @returns {number} Integer rank of the matrix.
     */
    rank(a, tol = 0) {
        const wa = a.toWasm(NDWasm.runtime);
        try {
            const suffix = NDWasm.runtime._getSuffix(a.dtype);
            return NDWasm.runtime.exports[`Rank${suffix}`](wa.ptr, a.shape[0], a.shape[1], tol);
        } finally {
            wa.dispose();
        }
    },
    
    /**
     * estimates the reciprocal condition number of matrix a.
     * @memberof NDWasmAnalysis
     * @param {NDArray} a - Input matrix.
     * @param {number} norm - norm: 1 (1-norm) or Infinity (Infinity norm).
     * @returns {number} result.
    */
    cond(a, norm = 1) {
        const wa = a.toWasm(NDWasm.runtime);
        try {
            const suffix = NDWasm.runtime._getSuffix(a.dtype);
            return NDWasm.runtime.exports[`Cond${suffix}`](wa.ptr, a.shape[0], norm);
        } finally {
            wa.dispose();
        }
    },

    /**
     * Eigenvalue decomposition for symmetric matrices.
     * @memberof NDWasmAnalysis
     * @param {NDArray} a - Symmetric square matrix.
     * @param {boolean} computeVectors - Whether to return eigenvectors.
     * @returns {{values: NDArray, vectors: NDArray|null}}
     */
    eigenSym(a, computeVectors = true) {
        const n = a.shape[0];
        const suffix = NDWasm.runtime._getSuffix(a.dtype);
        const wa = a.toWasm(NDWasm.runtime);
        const wv = NDWasm.runtime.createBuffer(n, a.dtype);
        const we = computeVectors ? NDWasm.runtime.createBuffer(n * n, a.dtype) : { ptr: 0, dispose: () => {} };
        try {
            NDWasm.runtime.exports[`EigenSym${suffix}`](wa.ptr, n, wv.ptr, we.ptr);
            return {
                values: fromWasm(wv, [n], a.dtype),
                vectors: computeVectors ? fromWasm(we, [n, n], a.dtype) : null
            };
        } finally {
            [wa, wv, we].forEach(b => b.dispose());
        }
    },

    // --- 3. Spatial & Iterative (O(m*n*d)) ---

    /**
     * Computes pairwise Euclidean distances between two sets of vectors.
     * @memberof NDWasmAnalysis
     * @param {NDArray} a - Matrix of shape [m, d].
     * @param {NDArray} b - Matrix of shape [n, d].
     * @returns {NDArray} Distance matrix of shape [m, n].
     */
    pairwiseDist(a, b) {
        const [m, d] = a.shape;
        const n = b.shape[0];
        const suffix = NDWasm.runtime._getSuffix(a.dtype);
        return NDWasm._compute([a, b], [m, n], a.dtype, (aP, bP, oP) => {
            return NDWasm.runtime.exports[`PairwiseDist${suffix}`](aP, bP, oP, m, n, d);
        });
    },

    /**
     * Performs K-Means clustering in WASM memory.
     * @memberof NDWasmAnalysis
     * @param {NDArray} data - Data of shape [n_samples, d_features].
     * @param {number} k - Number of clusters.
     * @param {number} maxIter - Maximum iterations.
     * @returns {{centroids: NDArray, labels: NDArray, iterations: number}}
     */
    kmeans(data, k, maxIter = 100) {
        const [n, d] = data.shape;
        const suffix = NDWasm.runtime._getSuffix(data.dtype);
        const wd = data.toWasm(NDWasm.runtime);
        const wc = NDWasm.runtime.createBuffer(k * d, data.dtype); // Initial centroids
        const wl = NDWasm.runtime.createBuffer(n, 'int32');
        try {
            const actualIters = NDWasm.runtime.exports[`KMeans${suffix}`](wd.ptr, wc.ptr, wl.ptr, n, d, k, maxIter);
            return {
                centroids: fromWasm(wc, [k, d], data.dtype),
                labels: fromWasm(wl, [n], 'int32'),
                iterations: actualIters
            };
        } finally {
            [wd, wc, wl].forEach(b => b.dispose());
        }
    },

    // --- 4. Advanced Structural (O(n^2 * m^2)) ---

    /**
     * Computes the Kronecker product C = A ⊗ B.
     * @memberof NDWasmAnalysis
     */
    kronecker(a, b) {
        const outShape = [a.shape[0] * b.shape[0], a.shape[1] * b.shape[1]];
        const suffix = NDWasm.runtime._getSuffix(a.dtype);
        return NDWasm._compute([a, b], outShape, a.dtype, (aP, bP, oP) => {
            return NDWasm.runtime.exports[`Kronecker${suffix}`](aP, bP, oP, a.shape[0], a.shape[1], b.shape[0], b.shape[1]);
        });
    },

};
