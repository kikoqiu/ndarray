/**
 * File: ndwasm_decomp.test.js
 * Responsibility: Test suite for Decompositions & Solvers (LU, SVD, QR, Cholesky, etc.)
 */
const ndarray = require('../dist/ndarray.cjs');
const { NDWasm, WasmRuntime, random } = ndarray;

describe('NDWasmDecomp (WASM)', () => {
    
    beforeAll(async () => {
        const runtime = new WasmRuntime();
        await runtime.init({
            execUrl: 'dist/wasm_exec.js',
            wasmUrl: 'dist/ndarray_plugin.wasm'
        });
        NDWasm.bind(runtime);
    }, 30000);

    // --- 1. solve (Ax = B) ---
    describe('solve', () => {
        test('case 1: identity system', () => {
            const a = ndarray.eye(3);
            const b = ndarray.array([[1], [2], [3]]);
            const x = a.solve(b);
            expect(x.data).toEqual(new Float64Array([1, 2, 3]));
        });

        test('case 2: simple 2x2 system', () => {
            // 3x + y = 9, x + 2y = 8 => x=2, y=3
            const a = ndarray.array([[3, 1], [1, 2]]);
            const b = ndarray.array([[9], [8]]);
            const x = a.solve(b);
            expect(x.get(0, 0)).toBeCloseTo(2);
            expect(x.get(1, 0)).toBeCloseTo(3);
        });

        test('case 3: multiple right-hand sides', () => {
            const a = ndarray.array([[2, 1], [1, 2]]);
            const b = ndarray.array([[3, 3], [3, 3]]); // Two identical B columns
            const x = a.solve(b);
            expect(x.shape).toEqual(new Int32Array([2, 2]));
            expect(x.get(0, 0)).toBeCloseTo(1);
            expect(x.get(0, 1)).toBeCloseTo(1);
        });

        test('case 4: error on non-square matrix', () => {
            const a = ndarray.zeros([3, 2]);
            const b = ndarray.zeros([3, 1]);
            expect(() => a.solve(b)).toThrow(/must be square/);
        });

        test('case 5: dimension mismatch', () => {
            const a = ndarray.eye(3);
            const b = ndarray.zeros([2, 1]);
            expect(() => a.solve(b)).toThrow(/match B's rows/);
        });
    });

    // --- 2. inv (Matrix Inverse) ---
    describe('inv', () => {
        test('case 1: 2x2 inverse', () => {
            const a = ndarray.array([[4, 7], [2, 6]]);
            const aInv = a.inv();
            // Expected: [[0.6, -0.7], [-0.2, 0.4]]
            expect(aInv.get(0, 0)).toBeCloseTo(0.6);
            expect(aInv.get(0, 1)).toBeCloseTo(-0.7);
        });

        test('case 2: identity inverse', () => {
            const a = ndarray.eye(4);
            const aInv = a.inv();
            aInv.data.forEach((val, i) => expect(val).toBeCloseTo(a.data[i]));
        });

        test('case 3: A * A_inv approx Identity', () => {
            const a = ndarray.array([[1, 2], [3, 4]]);
            const aInv = a.inv();
            const res = a.matmul(aInv);
            expect(res.get(0, 0)).toBeCloseTo(1);
            expect(res.get(0, 1)).toBeCloseTo(0);
            expect(res.get(1, 0)).toBeCloseTo(0);
            expect(res.get(1, 1)).toBeCloseTo(1);
        });

        test('case 4: diagonal matrix inverse', () => {
            const a = ndarray.array([[2, 0], [0, 5]]);
            const aInv = a.inv();
            expect(aInv.get(0, 0)).toBeCloseTo(0.5);
            expect(aInv.get(1, 1)).toBeCloseTo(0.2);
        });

        test('case 5: error on non-square', () => {
            const a = ndarray.zeros([2, 3]);
            expect(() => a.inv()).toThrow(/must be square/);
        });
    });

    // --- 3. svd (Singular Value Decomposition) ---
    describe('svd', () => {
        test('case 1: square matrix reconstruction (A = U S V^T)', () => {
            const a = ndarray.array([[1, 2], [3, 4]]);
            const { u, s, v } = a.svd();
            // Note: v from WASM is often V, so reconstruction is U @ diag(S) @ V^T
            const sDiag = ndarray.zeros([2, 2]);
            sDiag.set(s.get(0), 0, 0);
            sDiag.set(s.get(1), 1, 1);
            const recon = u.matmul(sDiag).matmul(v.transpose());
            recon.data.forEach((val, i) => expect(val).toBeCloseTo(a.data[i]));
        });

        test('case 2: singular values of identity', () => {
            const a = ndarray.eye(3);
            const { s } = a.svd();
            s.data.forEach(v => expect(v).toBeCloseTo(1));
        });

        test('case 3: tall matrix shape [3, 2]', () => {
            const a = ndarray.zeros([3, 2]);
            const { u, s, v } = a.svd();
            expect(u.shape).toEqual(new Int32Array([3, 3]));
            expect(s.shape).toEqual(new Int32Array([2]));
            expect(v.shape).toEqual(new Int32Array([2, 2]));
        });

        test('case 4: rank-1 matrix SVD', () => {
            const a = ndarray.ones([2, 2]);
            const { s } = a.svd();
            const sortedS = Array.from(s.data).sort((a,b) => b-a);
            expect(sortedS[0]).toBeCloseTo(2);
            expect(sortedS[1]).toBeCloseTo(0);
        });

        test('case 5: wide matrix reconstruction [2, 3]', () => {
            const a = ndarray.array([[1, 2, 3], [4, 5, 6]]);
            const { u, s, v } = a.svd();
            const sDiag = ndarray.zeros([2, 3]);
            sDiag.set(s.get(0), 0, 0);
            sDiag.set(s.get(1), 1, 1);
            const recon = u.matmul(sDiag).matmul(v.transpose());
            recon.data.forEach((val, i) => expect(val).toBeCloseTo(a.data[i]));
        });
    });

    // --- 4. qr (QR Decomposition) ---
    describe('qr', () => {
        test('case 1: square reconstruction (A = QR)', () => {
            const a = ndarray.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]]);
            const { q, r } = a.qr();
            const recon = q.matmul(r);
            recon.data.forEach((val, i) => expect(val).toBeCloseTo(a.data[i]));
        });

        test('case 2: Q is orthogonal (Q^T Q = I)', () => {
            const a = random.random([4, 4]);
            const { q } = a.qr();
            const res = q.transpose().matmul(q);
            const identity = ndarray.eye(4);
            res.data.forEach((val, i) => expect(val).toBeCloseTo(identity.data[i]));
        });

        test('case 3: R is upper triangular', () => {
            const a = random.random([3, 3]);
            const { r } = a.qr();
            expect(r.get(1, 0)).toBeCloseTo(0);
            expect(r.get(2, 0)).toBeCloseTo(0);
            expect(r.get(2, 1)).toBeCloseTo(0);
        });

        test('case 4: rectangular matrix QR [4, 2]', () => {
            const a = random.random([4, 2]);
            const { q, r } = a.qr();
            expect(q.shape).toEqual(new Int32Array([4, 4]));
            expect(r.shape).toEqual(new Int32Array([4, 2]));
            const recon = q.matmul(r);
            recon.data.forEach((val, i) => expect(val).toBeCloseTo(a.data[i]));
        });

        test('case 5: identity matrix QR', () => {
            const a = ndarray.eye(3);
            const { q, r } = a.qr();
            expect(q.data).toEqual(a.data);
            expect(r.data).toEqual(a.data);
        });
    });

    // --- 5. cholesky ---
    describe('cholesky', () => {
        test('case 1: 2x2 positive definite reconstruction (A = L L^T)', () => {
            const a = ndarray.array([[4, 12], [12, 37]]);
            const l = a.cholesky();
            const recon = l.matmul(l.transpose());
            recon.data.forEach((val, i) => expect(val).toBeCloseTo(a.data[i]));
        });

        test('case 2: identity matrix', () => {
            const a = ndarray.eye(3);
            const l = a.cholesky();
            expect(l.data).toEqual(a.data);
        });

        test('case 3: lower triangularity check', () => {
            const a = ndarray.array([[10, 5], [5, 10]]);
            const l = a.cholesky();
            expect(l.get(0, 1)).toBe(0);
        });

        test('case 4: 3x3 diagonal matrix', () => {
            const a = ndarray.array([[4, 0, 0], [0, 9, 0], [0, 0, 16]]);
            const l = a.cholesky();
            expect(l.get(0, 0)).toBeCloseTo(2);
            expect(l.get(1, 1)).toBeCloseTo(3);
            expect(l.get(2, 2)).toBeCloseTo(4);
        });

        test('case 5: large symmetric positive definite', () => {
            const a = ndarray.array([[10, 2, 3], [2, 10, 4], [3, 4, 10]]);
            const l = a.cholesky();
            const recon = l.matmul(l.transpose());
            recon.data.forEach((val, i) => expect(val).toBeCloseTo(a.data[i]));
        });
    });

    // --- 6. lu (LU Decomposition) ---
    describe('lu', () => {
        test('case 1: matrix', () => {
            const a = ndarray.array([[1, 2], [3, 4]]);
            const lu = a.lu();
            expect(lu.shape).toEqual(new Int32Array([2, 2]));
        });
    });

    // --- 7. pinv (Pseudo-inverse) ---
    describe('pinv', () => {
        test('case 1: match inv for square invertible', () => {
            const a = ndarray.array([[1, 2], [3, 4]]);
            const pinv = a.pinv();
            const inv = a.inv();
            pinv.data.forEach((val, i) => expect(val).toBeCloseTo(inv.data[i]));
        });

        test('case 2: tall matrix [3, 2] pseudo-inverse', () => {
            const a = ndarray.array([[1, 0], [0, 1], [0, 1]]);
            const pinv = a.pinv();
            expect(pinv.shape).toEqual(new Int32Array([2, 3]));
            // (A^T A)^-1 A^T
            const res = pinv.matmul(a); // Should be I(2x2)
            expect(res.get(0, 0)).toBeCloseTo(1);
            expect(res.get(1, 1)).toBeCloseTo(1);
        });

        test('case 3: wide matrix [2, 3] pseudo-inverse', () => {
            const a = ndarray.array([[1, 0, 0], [0, 1, 1]]);
            const pinv = a.pinv();
            expect(pinv.shape).toEqual(new Int32Array([3, 2]));
            const res = a.matmul(pinv); // Should be I(2x2)
            expect(res.get(0, 0)).toBeCloseTo(1);
            expect(res.get(1, 1)).toBeCloseTo(1);
        });

        test('case 4: rank-deficient matrix', () => {
            const a = ndarray.array([[1, 1], [1, 1]]);
            const pinv = a.pinv();
            // A @ pinv @ A should be A
            const res = a.matmul(pinv).matmul(a);
            res.data.forEach((v, i) => expect(v).toBeCloseTo(a.data[i]));
        });

        test('case 5: identity pseudo-inverse', () => {
            const a = ndarray.eye(3);
            const pinv = a.pinv();
            expect(pinv.data).toEqual(a.data);
        });
    });

    // --- 8. det (Determinant) ---
    describe('det', () => {
        test('case 1: 2x2 simple det', () => {
            const a = ndarray.array([[4, 6], [3, 8]]);
            expect(a.det()).toBeCloseTo(14); // 32 - 18
        });

        test('case 2: identity det', () => {
            const a = ndarray.eye(5);
            expect(a.det()).toBe(1);
        });

        test('case 3: singular matrix det', () => {
            const a = ndarray.array([[1, 2], [2, 4]]);
            expect(a.det()).toBeCloseTo(0);
        });

        test('case 4: scaling property', () => {
            const a = ndarray.array([[1, 2], [3, 4]]); // det = -2
            const scaled = a.mul(2); // det = -2 * 2^2 = -8
            expect(scaled.det()).toBeCloseTo(-8);
        });

        test('case 5: triangle matrix det', () => {
            const a = ndarray.array([[2, 3, 4], [0, 5, 6], [0, 0, 2]]);
            expect(a.det()).toBeCloseTo(20); // 2 * 5 * 2
        });
    });

    // --- 9. logDet ---
    describe('logDet', () => {
        test('case 1: identity log-det', () => {
            const a = ndarray.eye(3);
            const { sign, logAbsDet } = a.logDet();
            expect(sign).toBe(1);
            expect(logAbsDet).toBeCloseTo(0);
        });

        test('case 2: negative determinant sign', () => {
            const a = ndarray.array([[0, 1], [1, 0]]); // det = -1
            const { sign, logAbsDet } = a.logDet();
            expect(sign).toBe(-1);
            expect(logAbsDet).toBeCloseTo(0);
        });

        test('case 3: large diagonal matrix', () => {
            const a = ndarray.array([[Math.exp(2), 0], [0, Math.exp(3)]]);
            const { logAbsDet } = a.logDet();
            expect(logAbsDet).toBeCloseTo(5);
        });

        test('case 4: singular matrix log-det', () => {
            const a = ndarray.zeros([2, 2]);
            const { logAbsDet } = a.logDet();
            expect(logAbsDet).toBe(-Infinity);
        });

        test('case 5: consistency with det()', () => {
            const a = ndarray.array([[1, 5], [2, 3]]); // det = -7
            const d = a.det();
            const { sign, logAbsDet } = a.logDet();
            expect(sign * Math.exp(logAbsDet)).toBeCloseTo(d);
        });
    });

    // --- 10. eigen (Eigen Decomposition) ---
    describe('eigen', () => {
        test('case 1: simple 2x2 matrix with real eigenvalues', () => {
            // Matrix [[1, -1], [1, 3]] has a repeated eigenvalue of 2
            const a = ndarray.array([[1, -1], [1, 3]]);
            const { values, vectors } = a.eigen();
            
            // Eigenvalues are complex, returned as [real, imag]
            // For this matrix, they are {2, 2}.
            expect(values.shape).toEqual(new Int32Array([2, 2]));
            expect(values.get(0, 0)).toBeCloseTo(2);
            expect(values.get(0, 1)).toBeCloseTo(0); // imag part
            expect(values.get(1, 0)).toBeCloseTo(2);
            expect(values.get(1, 1)).toBeCloseTo(0); // imag part

            expect(vectors.shape).toEqual(new Int32Array([2, 2, 2]));
        });

        test('case 2: symmetric matrix (guaranteed real eigenvalues)', () => {
            const a = ndarray.array([[2, 1], [1, 2]]);
            const { values } = a.eigen(); // Eigenvalues are 1 and 3
            
            // Sort eigenvalues to have a consistent order for testing
            const real_eigenvalues = values.slice(null, 0).copy().data.slice().sort();
            
            expect(real_eigenvalues[0]).toBeCloseTo(1);
            expect(real_eigenvalues[1]).toBeCloseTo(3);

            // Imaginary parts should be zero
            expect(values.get(0, 1)).toBeCloseTo(0);
            expect(values.get(1, 1)).toBeCloseTo(0);
        });

        test('case 3: matrix with complex eigenvalues', () => {
            // Matrix [[0, -1], [1, 0]] has eigenvalues {i, -i}
            const a = ndarray.array([[0, -1], [1, 0]]);
            const { values } = a.eigen();

            // The eigenvalues are 0+1i and 0-1i. Order might vary.
            const v1_real = values.get(0, 0);
            const v1_imag = values.get(0, 1);
            const v2_real = values.get(1, 0);
            const v2_imag = values.get(1, 1);
            
            expect(v1_real).toBeCloseTo(0);
            expect(v2_real).toBeCloseTo(0);
            expect(Math.abs(v1_imag)).toBeCloseTo(1);
            expect(Math.abs(v2_imag)).toBeCloseTo(1);
            expect(v1_imag).toBeCloseTo(-v2_imag); // Should be conjugates
        });

        test('case 4: verify eigenvector equation (A*v = lambda*v)', () => {
            const a = ndarray.array([[6, -1], [2, 3]]); // E-vals: 5, 4
            const { values, vectors } = a.eigen();
            
            // First eigenvector
            const lambda1_real = values.get(0, 0);
            const v1_real = vectors.slice(null, 0, 0); // First column, real part
            
            // A * v
            const Av1 = a.matVecMul(v1_real);
            // lambda * v
            const lambda1v1 = v1_real.mul(lambda1_real);

            Av1.data.forEach((val, i) => {
                expect(val).toBeCloseTo(lambda1v1.data[i]);
            });
        });

    });
});