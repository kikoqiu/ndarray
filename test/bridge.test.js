const ndarray = require('../dist/ndarray.cjs');
const { decomp, NDWasm, WasmRuntime } = ndarray;

// This test suite requires the WASM module to be built and present in the dist/ directory.
// We also increase the timeout for this suite as WASM compilation and initialization can be slow.
describe('NDArray Bridge (WASM)', () => {
    
    beforeAll(async () => {
        // This assumes that the test is run from the project root (where jest is typically run)
        const runtime = new WasmRuntime();
        await runtime.init({
            execUrl: 'dist/wasm_exec.js',
            wasmUrl: 'dist/ndarray_plugin.wasm'
        });
        NDWasm.bind(runtime);
    }, 30000); // 30-second timeout for WASM setup

    test('WASM runtime should be loaded', () => {
        expect(NDWasm.runtime).not.toBeNull();
        expect(NDWasm.runtime.isLoaded).toBe(true);
    });

    test('matMul', () => {
        const a = ndarray.array([[1, 2], [3, 4]]);
        const b = ndarray.array([[5, 6], [7, 8]]);
        // c = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //   = [[19, 22], [43, 50]]
        
        const c = a.matMul(b);

        expect(c.shape).toEqual(new Int32Array([2, 2]));
        expect(c.copy().data).toEqual(new Float64Array([19, 22, 43, 50]));
    });
    
    // Add a test for another WASM function to be sure
    test('inv (matrix inverse)', () => {
        // Create a simple invertible matrix
        const a = ndarray.array([[4, 7], [2, 6]]);
        
        // Expected inverse: 1/(24-14) * [[6, -7], [-2, 4]] = 0.1 * [[6, -7], [-2, 4]]
        const expected = new Float64Array([0.6, -0.7, -0.2, 0.4]);

        const a_inv = a.inv();
        
        expect(a_inv.shape).toEqual(new Int32Array([2, 2]));
        // Compare with a tolerance for floating point errors
        a_inv.data.forEach((val, i) => {
            expect(val).toBeCloseTo(expected[i]);
        });
    });

    test('argsort', () => {
        const a = ndarray.array([5, 2, 8, 1, 9]);
        const sorted_indices = a.argsort();

        expect(sorted_indices.dtype).toBe('int32');
        expect(sorted_indices.copy().data).toEqual(new Int32Array([3, 1, 0, 2, 4]));
    });

    test('fft', () => {
        const r = ndarray.array([1, 0, 0, 0]);
        const i = ndarray.zeros([4]);
        const a = ndarray.stack([r, i], -1); // Create complex input
        const res = a.fft(); // Call new API

        expect(res.shape).toEqual(new Int32Array([4, 2])); // Check new shape

        const expected_real = new Float64Array([1, 1, 1, 1]);
        res.slice(null, 0).copy().data.forEach((val, idx) => { // Access real part
            expect(val).toBeCloseTo(expected_real[idx]);
        });
        
        // Imaginary part should be all zeros
        res.slice(null, 1).copy().data.forEach(val => { // Access imag part
            expect(val).toBeCloseTo(0);
        });
    });

    test('svd', () => {
        const a = ndarray.array([[1, 2], [3, 4], [5, 6]]); // 3x2 matrix
        const { u, s, v } = decomp.svd(a);

        // Check shapes
        expect(u.shape).toEqual(new Int32Array([3, 3]));
        expect(s.shape).toEqual(new Int32Array([2])); // min(m, n) = 2
        expect(v.shape).toEqual(new Int32Array([2, 2]));

        // Reconstruct A from U, S, V
        // A_recon = U @ diag(S) @ V
        // In our case, diag(S) is 3x2
        const s_diag = ndarray.zeros([3, 2]);
        s_diag.set(s.get(0), 0, 0);
        s_diag.set(s.get(1), 1, 1);
        
        const a_recon = u.matMul(s_diag).matMul(v);

        // Compare reconstructed matrix with original
        a_recon.copy().data.forEach((val, i) => {
            expect(val).toBeCloseTo(a.copy().data[i]);
        });
    });

});
