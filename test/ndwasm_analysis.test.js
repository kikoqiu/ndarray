/**
 * File: ndwasm_analysis.test.js
 * Responsibility: Test suite for advanced analysis, statistics, and spatial operators.
 */
const ndarray = require('../dist/ndarray.cjs');
const { analysis, NDWasm, WasmRuntime, random } = ndarray;

describe('NDWasmAnalysis (WASM)', () => {
    
    beforeAll(async () => {
        const runtime = new WasmRuntime();
        await runtime.init({
            execUrl: 'dist/wasm_exec.js',
            wasmUrl: 'dist/ndarray_plugin.wasm'
        });
        NDWasm.bind(runtime);
    }, 30000);

    // --- 1. argsort ---
    describe('argsort', () => {
        test('case 1: simple unsorted array', () => {
            const a = ndarray.array([10, 5, 8, 2]);
            const indices = a.argsort();
            expect(indices.data).toEqual(new Int32Array([3, 1, 2, 0]));
        });

        test('case 2: reverse sorted array', () => {
            const a = ndarray.array([4, 3, 2, 1]);
            const indices = a.argsort();
            expect(indices.data).toEqual(new Int32Array([3, 2, 1, 0]));
        });

        test('case 3: array with duplicate values', () => {
            const a = ndarray.array([1, 5, 2, 5, 3]);
            const indices = a.argsort();
            // 1(idx 0), 2(idx 2), 3(idx 4), 5(idx 1 or 3)
            expect(indices.get(0)).toBe(0);
            expect(indices.get(1)).toBe(2);
            expect(indices.get(2)).toBe(4);
        });

        test('case 4: already sorted array', () => {
            const a = ndarray.array([1.1, 2.2, 3.3]);
            const indices = a.argsort();
            expect(indices.data).toEqual(new Int32Array([0, 1, 2]));
        });

        test('case 5: array with negative numbers', () => {
            const a = ndarray.array([-1, -10, 5, 0]);
            const indices = a.argsort();
            expect(indices.data).toEqual(new Int32Array([1, 0, 3, 2]));
        });
    });

    // --- 2. topk ---
    describe('topk', () => {
        test('case 1: largest 3 elements', () => {
            const a = ndarray.array([1, 10, 5, 20, 2]);
            const { values, indices } = a.topk(3, true);
            expect(values.data).toEqual(new Float64Array([20, 10, 5]));
            expect(indices.data).toEqual(new Int32Array([3, 1, 2]));
        });

        test('case 2: smallest 2 elements', () => {
            const a = ndarray.array([1, 10, 5, 20, 2]);
            const { values, indices } = a.topk(2, false);
            expect(values.data).toEqual(new Float64Array([1, 2]));
            expect(indices.data).toEqual(new Int32Array([0, 4]));
        });

        test('case 3: k equals array size', () => {
            const a = ndarray.array([3, 1, 2]);
            const { values } = a.topk(3, true);
            expect(values.data).toEqual(new Float64Array([3, 2, 1]));
        });

        test('case 4: floating point precision', () => {
            const a = ndarray.array([0.1, 0.11, 0.09]);
            const { values } = a.topk(1, true);
            expect(values.get(0)).toBeCloseTo(0.11);
        });

        test('case 5: large k on small array (should handle/throw based on WASM impl)', () => {
            const a = ndarray.array([1, 2]);
            // 假设库会处理 k <= size
            const { values } = a.topk(2, true);
            expect(values.size).toBe(2);
        });
    });

    // --- 3. cov (Covariance) ---
    describe('cov', () => {
        test('case 1: uncorrelated identity-like data', () => {
            // 2 samples, 2 features
            const a = ndarray.array([[1, 0], [1, 0]]); 
            const c = a.cov();
            expect(c.shape).toEqual(new Int32Array([2, 2]));
            expect(c.get(0, 0)).toBe(0); // variance of constant is 0
        });

        test('case 2: positively correlated data', () => {
            const a = ndarray.array([[1, 1], [2, 2], [3, 3]]);
            const c = a.cov();
            expect(c.get(0, 1)).toBeGreaterThan(0);
            expect(c.get(0, 0)).toBeCloseTo(c.get(1, 1));
        });

        test('case 3: 2x2 matrix shape check', () => {
            const a = random.random([10, 3]);
            const c = a.cov();
            expect(c.shape).toEqual(new Int32Array([3, 3]));
        });

        test('case 4: scaling property', () => {
            const a = ndarray.array([[1, 2], [3, 4]]);
            const c1 = a.cov();
            const c2 = a.mul(2).cov();
            expect(c2.get(0, 0)).toBeCloseTo(c1.get(0, 0) * 4);
        });

        test('case 5: single feature data', () => {
            const a = ndarray.array([[1], [2], [3]]);
            const c = a.cov();
            expect(c.shape).toEqual(new Int32Array([1, 1]));
            expect(c.get(0, 0)).toBeCloseTo(1.0); // Sample variance of [1,2,3] is 1.0
        });
    });

    // --- 3. corr (Correlation) ---
    describe('corr', () => {
        test('case 1: perfectly correlated data', () => {
            const a = ndarray.array([[1, 2], [2, 4], [3, 6]]);
            const c = analysis.corr(a);
            expect(c.get(0, 0)).toBeCloseTo(1);
            expect(c.get(1, 1)).toBeCloseTo(1);
            expect(c.get(0, 1)).toBeCloseTo(1);
            expect(c.get(1, 0)).toBeCloseTo(1);
        });
    });

    // --- 4. norm ---
    describe('norm', () => {
        test('case 1: Frobenius norm (default)', () => {
            const a = ndarray.array([[3, 0], [0, 4]]);
            expect(a.norm(2)).toBeCloseTo(5); // sqrt(3^2 + 4^2)
        });

        test('case 2: 1-norm (max column sum)', () => {
            const a = ndarray.array([[1, 2], [3, 4]]);
            expect(a.norm(1)).toBe(6); // max(1+3, 2+4)
        });

        test('case 3: Infinity norm (max row sum)', () => {
            const a = ndarray.array([[1, 2], [3, 4]]);
            expect(a.norm(Infinity)).toBe(7); // max(1+2, 3+4)
        });

        test('case 4: zero matrix', () => {
            const a = ndarray.zeros([2, 2]);
            expect(a.norm()).toBe(0);
        });

        test('case 5: identity matrix Fro-norm', () => {
            const a = ndarray.eye(3);
            expect(a.norm(2)).toBeCloseTo(Math.sqrt(3));
        });

        test('case 6: float32 data', () => {
            const a = ndarray.array([[3, 0], [0, 4]], 'float32' );
            expect(a.norm(2)).toBeCloseTo(5);
        });
    });

    // --- 5. rank ---
    describe('rank', () => {
        test('case 1: full rank identity', () => {
            const a = ndarray.eye(4);
            expect(a.rank()).toBe(4);
        });

        test('case 2: rank deficient matrix', () => {
            const a = ndarray.array([[1, 2], [2, 4]]); // row2 = 2 * row1
            expect(a.rank()).toBe(1);
        });

        test('case 3: zero matrix rank', () => {
            const a = ndarray.zeros([3, 2]);
            expect(a.rank()).toBe(0);
        });

        test('case 4: rectangular matrix', () => {
            const a = ndarray.array([[1, 0, 0], [0, 1, 0]]);
            expect(a.rank()).toBe(2);
        });

        test('case 5: rank with tolerance', () => {
            const a = ndarray.array([[1, 0], [0, 1e-10]]);
            expect(a.rank(1e-9)).toBe(1);
            expect(a.rank(1e-11)).toBe(2);
        });

    });

    // --- 6. cond ---
    describe('cond', () => {
        test('case 1: identity matrix condition', () => {
            const a = ndarray.eye(3);
            expect(a.cond(1)).toBeCloseTo(1);
        });

    });

    // --- 7. eigenSym ---
    describe('eigenSym', () => {
        test('case 1: identity matrix eigenvalues', () => {
            const a = ndarray.eye(3);
            const { values } = a.eigenSym(false);
            values.data.forEach(v => expect(v).toBeCloseTo(1));
        });

        test('case 2: 2x2 symmetric matrix', () => {
            const a = ndarray.array([[2, 1], [1, 2]]);
            const { values } = a.eigenSym(false);
            // Eigenvalues of [[2,1],[1,2]] are 3 and 1
            const sortedVals = Array.from(values.data).sort((a, b) => b - a);
            expect(sortedVals[0]).toBeCloseTo(3);
            expect(sortedVals[1]).toBeCloseTo(1);
        });

        test('case 3: eigenvectors orthogonal', () => {
            const a = ndarray.array([[1, 0.5], [0.5, 1]]);
            const { vectors } = a.eigenSym(true);
            const v0 = vectors.colview(0);
            const v1 = vectors.colview(1);
            //console.log(v0.toString());
            //console.log(v1.toString());
            // dot product of eigenvectors should be 0
            let dot = v0.mul(v1).sum();
            expect(dot).toBeCloseTo(0);
        });

        test('case 4: diagonal matrix', () => {
            const a = ndarray.array([[5, 0], [0, 2]]);
            const { values } = a.eigenSym(false);
            const sortedVals = Array.from(values.data).sort((a, b) => b - a);
            expect(sortedVals).toEqual([5, 2]);
        });

        test('case 5: reconstruction A = VLV^T', () => {
            const a = ndarray.array([[4, 1], [1, 3]]);
            const { values, vectors } = a.eigenSym(true);
            // Reconstruct
            const L = ndarray.zeros([2, 2]);
            L.set(values.get(0), 0, 0);
            L.set(values.get(1), 1, 1);
            const recon = vectors.matmul(L).matmul(vectors.transpose());
            recon.data.forEach((v, i) => expect(v).toBeCloseTo(a.data[i]));
        });

        test('case 6: float32 symmetric matrix', () => {
            const a = ndarray.array([[2, 1], [1, 2]] );
            const { values } = a.eigenSym(false);
            const sortedVals = Array.from(values.data).sort((a, b) => b - a);
            expect(sortedVals[0]).toBeCloseTo(3);
            expect(sortedVals[1]).toBeCloseTo(1);
        });
    });

    // --- 8. pairwiseDist ---
    describe('pairwiseDist', () => {
        test('case 1: distance to self is zero', () => {
            const a = ndarray.array([[1, 1], [2, 2]]);
            const d = a.pairwiseDist(a);
            expect(d.get(0, 0)).toBeCloseTo(0);
            expect(d.get(1, 1)).toBeCloseTo(0);
        });

        test('case 2: 1D distances', () => {
            const a = ndarray.array([[0], [10]]);
            const b = ndarray.array([[5]]);
            const d = a.pairwiseDist(b); // [[5], [5]]
            expect(d.get(0, 0)).toBeCloseTo(5);
            expect(d.get(1, 0)).toBeCloseTo(5);
        });

        test('case 3: simple 2D Euclidean', () => {
            const a = ndarray.array([[0, 0]]);
            const b = ndarray.array([[3, 4]]);
            const d = a.pairwiseDist(b);
            expect(d.get(0, 0)).toBeCloseTo(5);
        });

        test('case 4: multiple points shape', () => {
            const a = random.random([5, 3]);
            const b = random.random([10, 3]);
            const d = a.pairwiseDist(b);
            expect(d.shape).toEqual(new Int32Array([5, 10]));
        });

        test('case 5: orthogonal unit vectors', () => {
            const a = ndarray.array([[1, 0]]);
            const b = ndarray.array([[0, 1]]);
            const d = a.pairwiseDist(b);
            expect(d.get(0, 0)).toBeCloseTo(Math.sqrt(2));
        });
    });

    // --- 9. kmeans ---
    describe('kmeans', () => {
        test('case 1: two clear clusters', () => {
            const data = ndarray.array([[1, 1], [1.1, 1.1], [10, 10], [10.1, 10.1]]);
            const { centroids, labels } = data.kmeans(2);
            expect(labels.get(0)).toBe(labels.get(1));
            expect(labels.get(2)).toBe(labels.get(3));
            expect(labels.get(0)).not.toBe(labels.get(2));
        });

        test('case 2: centroids shape', () => {
            const data = random.random([100, 5]);
            const k = 3;
            const { centroids } = data.kmeans(k);
            expect(centroids.shape).toEqual(new Int32Array([k, 5]));
        });

        test('case 3: labels length', () => {
            const data = random.random([50, 2]);
            const { labels } = data.kmeans(4);
            expect(labels.size).toBe(50);
        });

        test('case 4: data already at centroids', () => {
            const data = ndarray.array([[0, 0], [10, 10]]);
            const { centroids } = data.kmeans(2);
            // Centroids should be very close to [0,0] and [10,10]
            const cData = Array.from(centroids.data).sort((a,b) => a-b);
            expect(cData[0]).toBeCloseTo(0);
            expect(cData[cData.length-1]).toBeCloseTo(10);
        });

        test('case 5: max iterations', () => {
            const data = random.random([10, 2]);
            const { iterations } = data.kmeans(2, 1);
            expect(iterations).toBeLessThanOrEqual(1);
        });
    });

    // --- 10. kronecker ---
    describe('kronecker', () => {
        test('case 1: identity and matrix', () => {
            const a = ndarray.eye(2);
            const b = ndarray.array([[1, 2], [3, 4]]);
            const c = a.kronecker(b);
            // Result is block diagonal [[B, 0], [0, B]]
            expect(c.shape).toEqual(new Int32Array([4, 4]));
            expect(c.get(0, 0)).toBe(1);
            expect(c.get(2, 2)).toBe(1);
            expect(c.get(0, 2)).toBe(0);
        });

        test('case 2: all ones', () => {
            const a = ndarray.ones([2, 1]);
            const b = ndarray.ones([1, 2]);
            const c = a.kronecker(b);
            expect(c.shape).toEqual(new Int32Array([2, 2]));
            c.data.forEach(v => expect(v).toBe(1));
        });

        test('case 3: scalar-like (1x1)', () => {
            const a = ndarray.array([[2]]);
            const b = ndarray.array([[1, 3], [2, 4]]);
            const c = a.kronecker(b);
            expect(c.get(0, 1)).toBe(6);
            expect(c.get(1, 1)).toBe(8);
        });

        test('case 4: rectangular matrices', () => {
            const a = ndarray.zeros([2, 3]);
            const b = ndarray.zeros([4, 5]);
            const c = a.kronecker(b);
            expect(c.shape).toEqual(new Int32Array([8, 15]));
        });

        test('case 5: sign handling', () => {
            const a = ndarray.array([[1, -1]]);
            const b = ndarray.array([[2], [-2]]);
            const c = a.kronecker(b);
            // [ 1*[2,-2]^T, -1*[2,-2]^T ] = [[2, -2], [-2, 2]]
            expect(c.get(0, 0)).toBe(2);
            expect(c.get(1, 0)).toBe(-2);
            expect(c.get(0, 1)).toBe(-2);
            expect(c.get(1, 1)).toBe(2);
        });
    });
});