/**
 * File: ndwasm_blas.test.js
 * Responsibility: Test suite for BLAS (Basic Linear Algebra Subprograms) operators.
 */
const ndarray = require('../dist/ndarray.cjs');
const { NDArray, NDWasm, blas, WasmRuntime } = ndarray;


// --- blas ---
describe('blas', () => {
    test('case 1: dotProduct', () => {
        const a = ndarray.array([1, 2, 3, 4]);
        const b = ndarray.array([1, 2, 3, 4]);
        const c = a.dot(b);
        expect(c.data).toEqual(new Float64Array([30]));
    });

   test('case 1: crossProduct', () => {
        const a = ndarray.array([1, 1, 0]);
        const b = ndarray.array([1, 0, 1]);
        const c = a.cross(b);
        expect(c.data).toEqual(new Float64Array([1, -1, -1]));
    });

});

// --- 1. jsMatMul (GEMM) ---
describe('jsMatMul', () => {
    test('case 1: identity multiplication', () => {
        const a = ndarray.array([[1, 2], [3, 4]]);
        const b = ndarray.eye(2);
        const c = a.jsMatMul(b);
        expect(c.data).toEqual(new Float64Array([1, 2, 3, 4]));
    });

    test('case 2: non-square matrices (2x3 * 3x2)', () => {
        const a = ndarray.array([[1, 2, 3], [4, 5, 6]]);
        const b = ndarray.array([[7, 8], [9, 10], [11, 12]]);
        const c = a.jsMatMul(b);
        // [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        expect(c.shape).toEqual(new Int32Array([2, 2]));
        expect(c.get(0, 0)).toBe(58);
        expect(c.get(1, 1)).toBe(154);
    });

    test('case 3: error on dimension mismatch', () => {
        const a = ndarray.zeros([2, 3]);
        const b = ndarray.zeros([2, 2]);
        expect(() => a.jsMatMul(b)).toThrow(/inner dimensions must match/);
    });

    test('case 4: floating point precision', () => {
        const a = ndarray.array([[0.1, 0.2]]);
        const b = ndarray.array([[0.3], [0.4]]);
        const c = a.jsMatMul(b); // [[0.1*0.3 + 0.2*0.4]] = [[0.11]]
        expect(c.get(0, 0)).toBeCloseTo(0.11);
    });

    test('case 5: large zeros matrix', () => {
        const a = ndarray.zeros([10, 10]);
        const b = ndarray.zeros([10, 10]);
        const c = a.jsMatMul(b);
        c.data.forEach(v => expect(v).toBe(0));
    });
});

describe('jsMatVecMul', () => {
    test('case 1: matrix * vector projection', () => {
        const a = ndarray.array([[1, 2], [3, 4]]);
        const x = ndarray.array([1, 0]);
        const y = a.jsMatVecMul(x);
        expect(y.data).toEqual(new Float64Array([1, 3]));
    });

    test('case 2: identity matVecMul', () => {
        const a = ndarray.eye(3);
        const x = ndarray.array([10, 20, 30]);
        const y = a.jsMatVecMul(x);
        expect(y.data).toEqual(x.data);
    });

    test('case 3: shape [3, 2] * [2]', () => {
        const a = ndarray.zeros([3, 2]);
        const x = ndarray.zeros([2]);
        const y = a.jsMatVecMul(x);
        expect(y.shape).toEqual(new Int32Array([3]));
    });

    test('case 4: dimension mismatch', () => {
        const a = ndarray.zeros([2, 2]);
        const x = ndarray.zeros([3]);
        expect(() => a.jsMatVecMul(x)).toThrow();
    });

    test('case 5: scaling vector', () => {
        const a = ndarray.eye(2).mul(5);
        const x = ndarray.array([1, 2]);
        const y = a.jsMatVecMul(x);
        expect(y.data).toEqual(new Float64Array([5, 10]));
    });

    test('case 6: float32 data', () => {
        const a = ndarray.array([[1, 2], [3, 4]], 'float32' );
        const x = ndarray.array([1, 0], 'float32' );
        const y = a.jsMatVecMul(x);
        expect(y.dtype).toBe('float32');
        expect(y.get(0)).toBeCloseTo(1);
    });
});

describe('NDWasmBlas (WASM)', () => {
    
    beforeAll(async () => {
        const runtime = new WasmRuntime();
        await runtime.init({
            execUrl: 'dist/wasm_exec.js',
            wasmUrl: 'dist/ndarray_plugin.wasm'
        });
        NDWasm.bind(runtime);
    }, 30000);

    // --- 1. matMul (GEMM) ---
    describe('matMul', () => {
        test('case 1: identity multiplication', () => {
            const a = ndarray.array([[1, 2], [3, 4]]);
            const b = ndarray.eye(2);
            const c = a.matMul(b);
            expect(c.data).toEqual(new Float64Array([1, 2, 3, 4]));
        });

        test('case 2: non-square matrices (2x3 * 3x2)', () => {
            const a = ndarray.array([[1, 2, 3], [4, 5, 6]]);
            const b = ndarray.array([[7, 8], [9, 10], [11, 12]]);
            const c = a.matMul(b);
            // [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
            expect(c.shape).toEqual(new Int32Array([2, 2]));
            expect(c.get(0, 0)).toBe(58);
            expect(c.get(1, 1)).toBe(154);
        });

        test('case 3: error on dimension mismatch', () => {
            const a = ndarray.zeros([2, 3]);
            const b = ndarray.zeros([2, 2]);
            expect(() => a.matMul(b)).toThrow(/inner dimensions must match/);
        });

        test('case 4: floating point precision', () => {
            const a = ndarray.array([[0.1, 0.2]]);
            const b = ndarray.array([[0.3], [0.4]]);
            const c = a.matMul(b); // [[0.1*0.3 + 0.2*0.4]] = [[0.11]]
            expect(c.get(0, 0)).toBeCloseTo(0.11);
        });

        test('case 5: large zeros matrix', () => {
            const a = ndarray.zeros([10, 10]);
            const b = ndarray.zeros([10, 10]);
            const c = a.matMul(b);
            c.data.forEach(v => expect(v).toBe(0));
        });
    });

    // --- 2. matPow ---
    describe('matPow', () => {
        test('case 1: power 2 (A * A)', () => {
            const a = ndarray.array([[1, 2], [3, 4]]);
            const c = blas.matPow(a, 2);
            expect(c.data).toEqual(new Float64Array([7, 10, 15, 22]));
        });

        test('case 2: power 0 (should be Identity)', () => {
            const a = ndarray.array([[1, 2], [3, 4]]);
            const c = blas.matPow(a, 0);
            expect(c.data).toEqual(new Float64Array([1, 0, 0, 1]));
        });

        test('case 3: power 1 (should be Self)', () => {
            const a = ndarray.array([[5, 6], [7, 8]]);
            const c = blas.matPow(a, 1);
            expect(c.data).toEqual(a.data);
        });

        test('case 4: identity power', () => {
            const a = ndarray.eye(3);
            const c = blas.matPow(a, 5);
            expect(c.data).toEqual(a.data);
        });

        test('case 5: error on non-square', () => {
            const a = ndarray.zeros([2, 3]);
            expect(() => blas.matPow(a, 2)).toThrow(/must be square/);
        });

        test('case 6: compare with mat mul', () => {
            const a = ndarray.array([[1, 2, 8], [3, 4, 6], [1, 0, 9]]);
            let k=5;
            let b=a;
            for(let i=0;i<k-1;++i){
                b=b.matMul(a);
            }            
            const c = blas.matPow(a, k);
            expect(b.data).toEqual(c.data);
        });
    });

    // --- 3. matMulBatch ---
    describe('matMulBatch', () => {
        test('case 1: simple 3D batch multiplication', () => {
            const a = new NDArray(new Float64Array([1,0,0,1, 2,0,0,2]), {shape: [2, 2, 2]}); // Two Identity-like
            const b = new NDArray(new Float64Array([1,2,3,4, 5,6,7,8]), {shape: [2, 2, 2]});
            const c = a.matMulBatch(b);
            expect(c.shape).toEqual(new Int32Array([2, 2, 2]));
            // Batch 0: I * [1,2,3,4] = [1,2,3,4]
            // Batch 1: 2I * [5,6,7,8] = [10,12,14,16]
            expect(c.get(1, 0, 0)).toBe(10);
        });

        test('case 2: dimension mismatch in batch', () => {
            const a = ndarray.zeros([2, 2, 3]);
            const b = ndarray.zeros([2, 2, 2]);
            expect(() => a.matMulBatch(b)).toThrow();
        });

        test('case 3: batch size mismatch', () => {
            const a = ndarray.zeros([3, 2, 2]);
            const b = ndarray.zeros([2, 2, 2]);
            expect(() => a.matMulBatch(b)).toThrow();
        });

        test('case 4: check shape [2, 1, 3] * [2, 3, 1]', () => {
            const a = ndarray.zeros([2, 1, 3]);
            const b = ndarray.zeros([2, 3, 1]);
            const c = a.matMulBatch(b);
            expect(c.shape).toEqual(new Int32Array([2, 1, 1]));
        });

        test('case 5: error on non-3D input', () => {
            const a = ndarray.zeros([2, 2]);
            const b = ndarray.zeros([2, 2]);
            expect(() => blas.matMulBatch(a, b)).toThrow();
        });
    });

    // --- 4. syrk (Symmetric Rank-K Update) ---
    describe('syrk', () => {
        test('case 1: A * A^T shape', () => {
            const a = ndarray.zeros([5, 2]);
            const c = a.syrk();
            expect(c.shape).toEqual(new Int32Array([5, 5]));
        });

        test('case 2: simple symmetric result', () => {
            const a = ndarray.array([[1, 2]]); // 1x2
            const c = a.syrk(); // 1x1 result: [1*1 + 2*2] = [5]
            expect(c.get(0, 0)).toBe(5);
        });

        test('case 3: identity input', () => {
            const a = ndarray.eye(3);
            const c = a.syrk();
            expect(c.data).toEqual(a.data);
        });

        test('case 4: 2x2 symmetry check', () => {
            const a = ndarray.array([[1, 0], [1, 1]]);
            const c = a.syrk(); // [[1, 1], [1, 2]]
            expect(c.get(0, 1)).toBe(1);
            expect(c.get(1, 0)).toBe(1);
        });

        test('case 5: large zeros', () => {
            const a = ndarray.zeros([10, 2]);
            const c = a.syrk();
            c.data.forEach(v => expect(v).toBe(0));
        });

        test('case 6: float32 data', () => {
            const a = ndarray.array([[1, 0], [1, 1]], 'float32' );
            const c = a.syrk();
            expect(c.dtype).toBe('float32');
            expect(c.get(0, 1)).toBeCloseTo(1);
        });
    });

    // --- 5. trsm (Triangular Solver) ---
    describe('trsm', () => {
        test('case 1: solve with identity (L=I, B=B)', () => {
            const a = ndarray.eye(2);
            const b = ndarray.array([[5, 6], [7, 8]]);
            const x = blas.trsm(a, b, true);
            expect(x.data).toEqual(b.data);
        });

        test('case 2: lower triangular simple solve', () => {
            // L = [[2, 0], [4, 2]], B = [[2], [8]]
            // 2*x1 = 2 => x1 = 1
            // 4*x1 + 2*x2 = 8 => 4 + 2*x2 = 8 => x2 = 2
            const a = ndarray.array([[2, 0], [4, 2]]);
            const b = ndarray.array([[2], [8]]);
            const x = blas.trsm(a, b, true);
            expect(x.get(0, 0)).toBeCloseTo(1);
            expect(x.get(1, 0)).toBeCloseTo(2);
        });

        test('case 3: multiple columns in B', () => {
            const a = ndarray.eye(3);
            const b = ndarray.ones([3, 5]);
            const x = blas.trsm(a, b, true);
            expect(x.shape).toEqual(new Int32Array([3, 5]));
            x.data.forEach(v => expect(v).toBe(1));
        });

        test('case 4: dimension mismatch', () => {
            const a = ndarray.eye(3);
            const b = ndarray.ones([2, 3]);
            expect(() => blas.trsm(a, b, true)).toThrow();
        });

        test('case 5: non-square A', () => {
            const a = ndarray.zeros([3, 2]);
            const b = ndarray.zeros([3, 1]);
            expect(() => blas.trsm(a, b, true)).toThrow();
        });

        test('case 6: float32 triangular solve', () => {
            const a = ndarray.array([[2, 0], [4, 2]], 'float32' );
            const b = ndarray.array([[2], [8]], 'float32' );
            const x = blas.trsm(a, b, true);
            expect(x.dtype).toBe('float32');
            expect(x.get(0, 0)).toBeCloseTo(1);
            expect(x.get(1, 0)).toBeCloseTo(2);
        });

        test('case 7: upper triangular solve', () => {
            // U = [[2, 4], [0, 2]], B = [[8], [2]]
            // 2*x2 = 2 => x2 = 1
            // 2*x1 + 4*x2 = 8 => 2*x1 + 4 = 8 => x1 = 2
            const a = ndarray.array([[2, 4], [0, 2]]);
            const b = ndarray.array([[8], [2]]);
            const x = blas.trsm(a, b, false);
            expect(x.get(0, 0)).toBeCloseTo(2);
            expect(x.get(1, 0)).toBeCloseTo(1);
        });
    });

    // --- 6. matVecMul ---
    describe('matVecMul', () => {
        test('case 1: matrix * vector projection', () => {
            const a = ndarray.array([[1, 2], [3, 4]]);
            const x = ndarray.array([1, 0]);
            const y = a.matVecMul(x);
            expect(y.data).toEqual(new Float64Array([1, 3]));
        });

        test('case 2: identity matVecMul', () => {
            const a = ndarray.eye(3);
            const x = ndarray.array([10, 20, 30]);
            const y = a.matVecMul(x);
            expect(y.data).toEqual(x.data);
        });

        test('case 3: shape [3, 2] * [2]', () => {
            const a = ndarray.zeros([3, 2]);
            const x = ndarray.zeros([2]);
            const y = a.matVecMul(x);
            expect(y.shape).toEqual(new Int32Array([3]));
        });

        test('case 4: dimension mismatch', () => {
            const a = ndarray.zeros([2, 2]);
            const x = ndarray.zeros([3]);
            expect(() => a.matVecMul(x)).toThrow();
        });

        test('case 5: scaling vector', () => {
            const a = ndarray.eye(2).mul(5);
            const x = ndarray.array([1, 2]);
            const y = a.matVecMul(x);
            expect(y.data).toEqual(new Float64Array([5, 10]));
        });

        test('case 6: float32 data', () => {
            const a = ndarray.array([[1, 2], [3, 4]], 'float32' );
            const x = ndarray.array([1, 0], 'float32' );
            const y = a.matVecMul(x);
            expect(y.dtype).toBe('float32');
            expect(y.get(0)).toBeCloseTo(1);
        });
    });

    // --- 7. ger (Vector Outer Product) ---
    describe('ger', () => {
        test('case 1: simple outer product', () => {
            const x = ndarray.array([1, 2]);
            const y = ndarray.array([3, 4]);
            const c = x.ger(y);
            // [[1*3, 1*4], [2*3, 2*4]] = [[3, 4], [6, 8]]
            expect(c.data).toEqual(new Float64Array([3, 4, 6, 8]));
        });

        test('case 2: result shape', () => {
            const x = ndarray.zeros([5]);
            const y = ndarray.zeros([3]);
            const c = x.ger(y);
            expect(c.shape).toEqual(new Int32Array([5, 3]));
        });

        test('case 3: zeros vector product', () => {
            const x = ndarray.zeros([2]);
            const y = ndarray.array([1, 2]);
            const c = x.ger(y);
            c.data.forEach(v => expect(v).toBe(0));
        });

        test('case 4: rank-1 property check', () => {
            const x = ndarray.ones([2]);
            const y = ndarray.ones([2]);
            const c = x.ger(y);
            expect(c.rank()).toBe(1);
        });

        test('case 5: negative values', () => {
            const x = ndarray.array([1, -1]);
            const y = ndarray.array([-2, 2]);
            const c = x.ger(y); // [[-2, 2], [2, -2]]
            expect(c.get(1, 0)).toBe(2);
            expect(c.get(1, 1)).toBe(-2);
        });

        test('case 6: float32 vectors', () => {
            const x = ndarray.array([1, 2], 'float32' );
            const y = ndarray.array([3, 4], 'float32' );
            const c = x.ger(y);
            expect(c.dtype).toBe('float32');
            expect(c.get(1, 1)).toBeCloseTo(8);
        });
    });
});