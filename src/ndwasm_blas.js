import { NDArray } from './ndarray_core.js';
import { NDWasm, fromWasm } from './ndwasm.js';
/** 
 * NDWasmBlas: BLAS (Basic Linear Algebra Subprograms)
 * Handles O(n^2) and O(n^3) matrix-matrix and matrix-vector operations.
 * @namespace NDWasmBlas
 */
export const NDWasmBlas = {

    /**
     * Calculates the trace of a 2D square matrix (sum of diagonal elements).
     * Complexity: O(n)
     * @param {NDArray} a
     * @returns {number} The sum of the diagonal elements.
     * @throws {Error} If the array is not 2D or not a square matrix.
     */
    trace(a) {
        // 1. Validation: Must be 2D
        if (a.ndim !== 2) {
            throw new Error(`Trace is only defined for 2D matrices, but this array is ${a.ndim}D.`);
        }

        // 2. Validation: Must be square (N x N)
        const rows = a.shape[0];
        const cols = a.shape[1];
        if (rows !== cols) {
            throw new Error(`Trace is only defined for square matrices, but this matrix is ${rows}x${cols}.`);
        }

        let sum = 0;
        const data = a.data;
        const offset = a.offset;
        const s0 = a.strides[0]; // Row stride
        const s1 = a.strides[1]; // Column stride

        // 3. Calculation: Sum A[i, i]
        // Memory index for element [i, i] is: offset + i * strides[0] + i * strides[1]
        // We can optimize the loop by pre-calculating the step size: s0 + s1
        const step = s0 + s1;
        for (let i = 0; i < rows; i++) {
            sum += data[offset + i * step];
        }

        return sum;
    },
    

    /**
     * General Matrix Multiplication (GEMM): C = A * B.
     * Complexity: O(m * n * k)
     * @memberof NDWasmBlas
     * @param {NDArray} a - Left matrix of shape [m, n].
     * @param {NDArray} b - Right matrix of shape [n, k].
     * @returns {NDArray} Result matrix of shape [m, k].
     */
    matmul(a, b) {
        if (a.shape[1] !== b.shape[0]) {
            throw new Error(`Matrix inner dimensions must match: ${a.shape[1]} != ${b.shape[0]}`);
        }
        if(b.ndim!==2 && b.ndim!==1){
            throw new Error(`Right operand must be a 2D matrix (or 1D vector), but got ${b.ndim}D.`);
        }
        const m = a.shape[0];
        const n = a.shape[1];
        const k = b.ndim === 2 ? b.shape[1] : 1;
        const outShape = [m, k];
        const suffix = NDWasm.runtime._getSuffix(a.dtype);

        return NDWasm._compute([a, b], outShape, a.dtype, (aPtr, bPtr, outPtr) => {
            return NDWasm.runtime.exports[`MatMul${suffix}`](aPtr, bPtr, outPtr, m, n, k);
        });
    },

    /**
     * matPow computes A^k (Matrix Power).
     * Matrix Functions (O(n^3))
     * @memberof NDWasmBlas
     * @param {NDArray} a - Matrix of shape [n, n].
     * @returns {NDArray} Result matrix of shape [n, n].
     */
    matPow(a, k) {
        if (a.shape[0] !== a.shape[1]) {
            throw new Error(`Matrix must be square: ${a.shape[0]} != ${a.shape[1]}`);
        }
        const n = a.shape[0];
        const outShape = [n, n];
        const suffix = NDWasm.runtime._getSuffix(a.dtype);

        return NDWasm._compute([a], outShape, a.dtype, (aPtr,  outPtr) => {
            return NDWasm.runtime.exports[`MatrixPower${suffix}`](aPtr, outPtr, n, k);
        });
    },


    /**
     * Batched Matrix Multiplication: C[i] = A[i] * B[i].
     * Common in deep learning inference.
     * Complexity: O(batch * m * n * k)
     * @memberof NDWasmBlas
     * @param {NDArray} a - Batch of matrices of shape [batch, m, n].
     * @param {NDArray} b - Batch of matrices of shape [batch, n, k].
     * @returns {NDArray} Result batch of shape [batch, m, k].
     */
    matmulBatch(a, b) {
        if (a.ndim !== 3 || b.ndim !== 3 || a.shape[0] !== b.shape[0]) {
            throw new Error("Input must be 3D batches with same batch size.");
        }
        if (a.shape[2] !== b.shape[1]) {
            throw new Error("Batch matrix inner dimensions must match.");
        }
        const batch = a.shape[0];
        const m = a.shape[1];
        const n = a.shape[2];
        const k = b.shape[2];
        const outShape = [batch, m, k];
        const suffix = NDWasm.runtime._getSuffix(a.dtype);

        return NDWasm._compute([a, b], outShape, a.dtype, (aPtr, bPtr, outPtr) => {
            return NDWasm.runtime.exports[`MatMulBatch${suffix}`](aPtr, bPtr, outPtr, batch, m, n, k);
        });
    },

    /**
     * Symmetric Rank-K Update: C = alpha * A * A^T + beta * C.
     * Used for efficiently computing covariance matrices or Gram matrices.
     * Complexity: O(n^2 * k)
     * @memberof NDWasmBlas
     * @param {NDArray} a - Input matrix of shape [n, k].
     * @returns {NDArray} Symmetric result matrix of shape [n, n].
     */
    syrk(a) {
        const n = a.shape[0];
        const k = a.shape[1];
        const outShape = [n, n];
        const suffix = NDWasm.runtime._getSuffix(a.dtype);

        return NDWasm._compute([a], outShape, a.dtype, (aPtr, outPtr) => {
            return NDWasm.runtime.exports[`Syrk${suffix}`](aPtr, outPtr, n, k);
        });
    },

    /**
     * Triangular System Solver: Solves A * X = B for X, where A is a triangular matrix.
     * Complexity: O(m^2 * n)
     * @memberof NDWasmBlas
     * @param {NDArray} a - Triangular matrix of shape [m, m].
     * @param {NDArray} b - Right-hand side matrix/vector of shape [m, n].
     * @returns {NDArray} Solution matrix X of shape [m, n].
     */
    trsm(a, b, lower = false) {
        if (a.shape[0] !== a.shape[1] || a.shape[0] !== b.shape[0]) {
            throw new Error("Dimension mismatch for triangular solver.");
        }
        const m = a.shape[0];
        const n = b.ndim === 1 ? 1 : b.shape[1];
        const suffix = NDWasm.runtime._getSuffix(a.dtype);

        // We use _compute but must ensure b is copied to the output buffer first 
        // because Go's TRSM often operates on the B matrix storage.
        const wa = a.toWasm(NDWasm.runtime);
        const wb = b.toWasm(NDWasm.runtime); // wb already contains b's data
        try {
            NDWasm.runtime.exports[`Trsm${suffix}`](wa.ptr, wb.ptr, m, n, lower ? 1 : 0);
            return fromWasm(wb, b.shape, b.dtype);
        } finally {
            wa.dispose();
            wb.dispose();
        }
    },

    /**
     * Matrix-Vector Multiplication: y = A * x.
     * Complexity: O(m * n)
     * @memberof NDWasmBlas
     * @param {NDArray} a - Matrix of shape [m, n].
     * @param {NDArray} x - Vector of shape [n].
     * @returns {NDArray} Result vector of shape [m].
     */
    matVecMul(a, x) {
        if (a.shape[1] !== x.size) {
            throw new Error("Matrix-Vector dimension mismatch.");
        }
        const m = a.shape[0];
        const n = a.shape[1];
        const suffix = NDWasm.runtime._getSuffix(a.dtype);

        return NDWasm._compute([a, x], [m], a.dtype, (aPtr, xPtr, outPtr) => {
            return NDWasm.runtime.exports[`MatVecMul${suffix}`](aPtr, xPtr, outPtr, m, n);
        });
    },

    /**
     * Vector Outer Product (Rank-1 Update): A = x * y^T.
     * Complexity: O(m * n)
     * @memberof NDWasmBlas
     * @param {NDArray} x - Vector of shape [m].
     * @param {NDArray} y - Vector of shape [n].
     * @returns {NDArray} Result matrix of shape [m, n].
     */
    ger(x, y) {
        const m = x.size;
        const n = y.size;
        const outShape = [m, n];
        const suffix = NDWasm.runtime._getSuffix(x.dtype);

        return NDWasm._compute([x, y], outShape, x.dtype, (xPtr, yPtr, outPtr) => {
            return NDWasm.runtime.exports[`Ger${suffix}`](xPtr, yPtr, outPtr, m, n);
        });
    }
};
