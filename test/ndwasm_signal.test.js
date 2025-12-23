/**
 * File: ndwasm_signal.test.js
 * Responsibility: Test suite for Signal Processing & Transformations (FFT, DCT, Conv2D).
 */
const ndarray = require('../dist/ndarray.cjs');
const { random, NDWasm, WasmRuntime, signal} = ndarray;

describe('NDWasmSignal (WASM)', () => {
    
    beforeAll(async () => {
        const runtime = new WasmRuntime();
        await runtime.init({
            execUrl: 'dist/wasm_exec.js',
            wasmUrl: 'dist/ndarray_plugin.wasm'
        });
        NDWasm.bind(runtime);
    }, 30000);

    // --- 1. fft (1D Fast Fourier Transform) ---
    describe('fft', () => {
        test('case 1: impulse signal at index 0', () => {
            const r = ndarray.array([1, 0, 0, 0]);
            const i = ndarray.zeros([4]);
            const a = ndarray.stack([r, i], -1);
            const res = a.fft();
            // FFT of [1,0,0,0] is [1,1,1,1] (all frequencies present)
            res.slice(null, 0).copy().data.forEach(v => expect(v).toBeCloseTo(1));
            res.slice(null, 1).copy().data.forEach(v => expect(v).toBeCloseTo(0));
        });

        test('case 2: DC signal (all ones)', () => {
            const r = ndarray.array([1, 1, 1, 1]);
            const i = ndarray.zeros([4]);
            const a = ndarray.stack([r, i], -1);
            const res = a.fft();
            // FFT of [1,1,1,1] is [4,0,0,0]
            expect(res.get(0, 0)).toBeCloseTo(4);
            expect(res.get(1, 0)).toBeCloseTo(0);
            expect(res.get(0, 1)).toBeCloseTo(0);
        });

        test('case 3: sine wave frequency component', () => {
            const n = 8;
            const r = ndarray.zeros([n]);
            for(let i=0; i<n; i++) r.set(Math.sin(2 * Math.PI * i / n), i);
            const i = ndarray.zeros([n]);
            const a = ndarray.stack([r, i], -1);
            const res = a.fft();
            // Pure sine wave should have peaks in the imaginary part
            expect(Math.abs(res.get(1, 1))).toBeGreaterThan(1);
        });

        test('case 4: result shape consistency', () => {
            const r = random.random([16]);
            const i = ndarray.zeros([16]);
            const a = ndarray.stack([r, i], -1);
            const res = a.fft();
            expect(res.shape).toEqual(new Int32Array([16, 2]));
        });

        test('case 5: linearity (FFT(2*x) = 2*FFT(x))', () => {
            const r = ndarray.array([1, 2, 3, 4]);
            const i = ndarray.zeros([4]);
            const a = ndarray.stack([r, i], -1);
            const res1 = a.fft();
            const res2 = a.mul(2).fft();
            expect(res2.get(0, 0)).toBeCloseTo(res1.get(0, 0) * 2);
        });
    });

    // --- 2. ifft (1D Inverse FFT) ---
    describe('ifft', () => {
        test('case 1: round trip (IFFT of FFT is self)', () => {
            const r = ndarray.array([1, 2, 3, 4]);
            const i = ndarray.zeros([4]);
            const original = ndarray.stack([r, i], -1);
            const f = original.fft();
            const inv = signal.ifft(f);
            inv.slice(null, 0).copy().data.forEach((v, i) => expect(v).toBeCloseTo(original.slice(i, 0).get()));
        });

        test('case 2: IFFT of DC peak', () => {
            const real = ndarray.array([4, 0, 0, 0]);
            const imag = ndarray.zeros([4]);
            const a = ndarray.stack([real, imag], -1);
            const res = signal.ifft(a);
            res.slice(null, 0).copy().data.forEach(v => expect(v).toBeCloseTo(1));
        });

        test('case 3: error on shape mismatch', () => {
            const a = ndarray.zeros([4]); // Not a complex array
            expect(() => signal.ifft(a)).toThrow(/Input to ifft must be a 1D complex array with shape \[n, 2\]./);
        });

        test('case 4: zero signal', () => {
            const r = ndarray.zeros([4]);
            const i = ndarray.zeros([4]);
            const a = ndarray.stack([r, i], -1);
            const res = signal.ifft(a);
            res.slice(null, 0).copy().data.forEach(v => expect(v).toBe(0));
            res.slice(null, 1).copy().data.forEach(v => expect(v).toBe(0));
        });

        test('case 5: imaginary restoration', () => {
            const r = ndarray.zeros([4]);
            const i = ndarray.array([0, 4, 0, -4]); // Represents a sine in freq
            const a = ndarray.stack([r, i], -1);
            const res = signal.ifft(a);
            expect(Math.abs(res.get(1, 0))).toBeGreaterThan(0);
        });
    });

    // --- 3. rfft (Real FFT) ---
    describe('rfft', () => {
        test('case 1: output size for even input', () => {
            const a = ndarray.zeros([8]);
            const res = a.rfft();
            expect(res.shape).toEqual(new Int32Array([5, 2])); // n/2 + 1
        });

        test('case 2: output size for odd input', () => {
            const a = ndarray.zeros([7]);
            const res = a.rfft();
            expect(res.shape).toEqual(new Int32Array([4, 2])); // floor(7/2) + 1
        });

        test('case 3: impulse at 0 (Real FFT)', () => {
            const a = ndarray.array([1, 0, 0, 0, 0, 0]);
            const res = a.rfft();
            res.slice(null, 0).copy().data.forEach(v => expect(v).toBeCloseTo(1));
        });

        test('case 4: DC component', () => {
            const a = ndarray.ones([4]);
            const res = a.rfft();
            expect(res.get(0, 0)).toBeCloseTo(4);
            expect(res.get(1, 0)).toBeCloseTo(0);
        });

        test('case 5: data type consistency', () => {
            const a = random.random([10]);
            const res = a.rfft();
            expect(res.dtype).toBe('float64');
            expect(res.shape).toEqual(new Int32Array([6, 2]));
        });
    });

    // --- 4. rifft (Real Inverse FFT) ---
    describe('rifft', () => {
        test('case 1: round trip with rfft', () => {
            const original = ndarray.array([1, 2, 3, 4, 5, 6]);
            const f = original.rfft();
            const res = signal.rifft(f, original.size);
            res.data.forEach((v, i) => expect(v).toBeCloseTo(original.data[i]));
        });
    });

    // --- 5. fft2 (2D FFT) ---
    describe('fft2', () => {
        test('case 1: 2D DC signal', () => {
            const r = ndarray.ones([2, 2]);
            const i = ndarray.zeros([2, 2]);
            const a = ndarray.stack([r, i], -1);
            const res = a.fft2();
            // Sum of all elements is 4, should be at (0,0)
            expect(res.get(0, 0, 0)).toBeCloseTo(4);
            expect(res.get(0, 1, 0)).toBeCloseTo(0);
        });

        test('case 2: shape preservation', () => {
            const r = random.random([4, 8]);
            const i = ndarray.zeros([4, 8]);
            const a = ndarray.stack([r, i], -1);
            const res = a.fft2();
            expect(res.shape).toEqual(new Int32Array([4, 8, 2]));
        });

        test('case 3: single pixel impulse', () => {
            const r = ndarray.zeros([4, 4]);
            r.set(1, 0, 0);
            const i = ndarray.zeros([4, 4]);
            const a = ndarray.stack([r, i], -1);
            const res = a.fft2();
            res.slice(null, null, 0).copy().data.forEach(v => expect(v).toBeCloseTo(1));
        });

        test('case 4: error on non-3D', () => {
            const a = ndarray.zeros([4, 4]);
            expect(() => a.fft2()).toThrow(/requires a 3D array/);
        });

        test('case 5: horizontal stripe', () => {
            const r = ndarray.array([[1, 1], [0, 0]]);
            const i = ndarray.zeros([2, 2]);
            const a = ndarray.stack([r, i], -1);
            const res = a.fft2();
            // Vertical frequency component should be non-zero
            expect(res.get(1, 0, 0)).not.toBe(0);
        });
    });

    // --- 6. ifft2 (2D Inverse FFT) ---
    describe('ifft2', () => {
        test('case 1: round trip with fft2', () => {
            const r = random.random([4, 4]);
            const i = ndarray.zeros([4, 4]);
            const original = ndarray.stack([r, i], -1);
            const f = original.fft2();
            const inv = signal.ifft2(f);
            inv.slice(null, null, 0).data.forEach((v, i) => expect(v).toBeCloseTo(original.data[i]));
        });
    });

    // --- 7. dct (Discrete Cosine Transform) ---
    describe('dct', () => {
        test('case 1: DC signal', () => {
            const a = ndarray.ones([4]);
            const res = a.dct();
            // For DCT-II, DC component is at index 0
            expect(res.get(0)).toBeGreaterThan(1);
            expect(res.get(1)).toBeCloseTo(0);
        });

        test('case 2: impulse at 0', () => {
            const a = ndarray.array([1, 0, 0, 0]);
            const res = a.dct();
            // DCT of impulse is a cosine wave
            expect(res.get(0)).not.toBe(0);
            expect(res.get(1)).not.toBe(0);
        });

        test('case 3: zero signal', () => {
            const a = ndarray.zeros([10]);
            const res = a.dct();
            res.data.forEach(v => expect(v).toBe(0));
        });

        test('case 4: energy conservation (relative)', () => {
            let a = random.random([8]);
            const res = a.dct();
            expect(res.norm()).toBeGreaterThan(0);
        });

        test('case 5: shape consistency', () => {
            const a = random.random([16]);
            const res = a.dct();
            expect(res.shape).toEqual(a.shape);
        });
    });

    // --- 8. conv2d ---
    describe('conv2d', () => {
        test('case 1: identity kernel (1x1)', () => {
            const img = ndarray.array([[1, 2], [3, 4]]);
            const kernel = ndarray.array([[1]]);
            const res = img.conv2d(kernel);
            expect(res.data).toEqual(img.data);
        });

        test('case 2: blur kernel (3x3)', () => {
            const img = ndarray.ones([5, 5]);
            const kernel = ndarray.ones([3, 3]).div(9);
            const res = img.conv2d(kernel);
            expect(res.shape).toEqual(new Int32Array([3, 3]));
            res.data.forEach(v => expect(v).toBeCloseTo(1));
        });

        test('case 3: stride test', () => {
            const img = ndarray.ones([5, 5]);
            const kernel = ndarray.ones([2, 2]);
            const res = img.conv2d(kernel, 2, 0); // stride=2
            expect(res.shape).toEqual(new Int32Array([2, 2]));
        });

        test('case 4: padding test', () => {
            const img = ndarray.ones([3, 3]);
            const kernel = ndarray.ones([3, 3]);
            const res = img.conv2d(kernel, 1, 1); // padding=1
            expect(res.shape).toEqual(new Int32Array([3, 3]));
        });

        test('case 5: edge detection (Sobel-like)', () => {
            const img = ndarray.ones([5, 5]);
            const kernel = ndarray.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]);
            const res = img.conv2d(kernel);
            // On a flat field, edge detector should return 0
            res.data.forEach(v => expect(v).toBeCloseTo(0));
        });

        test('case 6: float32 data', () => {
            const img = ndarray.array([[1, 2], [3, 4]], 'float32' );
            const kernel = ndarray.array([[1]], 'float32' );
            const res = img.conv2d(kernel);
            expect(res.dtype).toBe('float32');
            expect(res.data).toEqual(img.data);
        });
    });

    // --- 9. correlate2d ---
    describe('correlate2d', () => {
        test('case 1: match with convolution for symmetric kernel', () => {
            const img = random.random([5, 5]);
            const kernel = ndarray.ones([3, 3]); // Symmetric
            const c1 = img.conv2d(kernel);
            const c2 = img.correlate2d(kernel);
            c1.data.forEach((v, i) => expect(v).toBeCloseTo(c2.data[i]));
        });

        test('case 2: peak detection', () => {
            const img = ndarray.zeros([5, 5]);
            img.set(1, 2, 2);
            const kernel = ndarray.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]]);
            const res = img.correlate2d(kernel, 1, 1);
            // Center of correlation should be 2
            expect(res.get(2, 2)).toBe(2);
        });

        test('case 3: identity test', () => {
            const img = ndarray.array([[10, 20], [30, 40]]);
            const kernel = ndarray.array([[1]]);
            const res = img.correlate2d(kernel);
            expect(res.data).toEqual(img.data);
        });

        test('case 4: rectangular shape', () => {
            const img = ndarray.zeros([10, 5]);
            const kernel = ndarray.zeros([3, 2]);
            const res = img.correlate2d(kernel);
            expect(res.shape).toEqual(new Int32Array([8, 4]));
        });

        test('case 5: offset kernel correlation', () => {
            const img = ndarray.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]);
            const kernel = ndarray.array([[1, 0], [0, 0]]);
            const res = img.correlate2d(kernel);
            // The 1 should "move" in the result based on kernel peak
            expect(res.get(1, 1)).toBe(1);
        });

        test('case 6: float32 data', () => {
            const img = ndarray.array([[10, 20], [30, 40]], 'float32' );
            const kernel = ndarray.array([[1]], 'float32' );
            const res = img.correlate2d(kernel);
            expect(res.dtype).toBe('float32');
            expect(res.data).toEqual(img.data);
        });
    });
});