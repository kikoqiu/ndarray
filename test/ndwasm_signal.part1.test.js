
const ndarray = require('../dist/ndarray.cjs');
const { random, NDWasm, WasmRuntime, signal } = ndarray;

// Helper to create a complex array from a real part, assuming imag part is zero
const toComplex = (realNdArray) => {
    const imagNdArray = ndarray.zeros(realNdArray.shape, realNdArray.dtype);
    return ndarray.stack([realNdArray, imagNdArray], -1);
};

describe('NDWasmSignal (WASM) - Supplementary Tests', () => {

    beforeAll(async () => {
        if (!NDWasm.isLoaded) {
            const runtime = new WasmRuntime();
            await runtime.init({
                execUrl: 'dist/wasm_exec.js',
                wasmUrl: 'dist/ndarray_plugin.wasm'
            });
            NDWasm.bind(runtime);
        }
    }, 30000);

    // --- 1. fft / ifft supplementary tests ---
    describe('fft / ifft', () => {
        const cases = [7, 8, 15, 16, 100]; // Test odd, even, non-power-of-2 sizes
        cases.forEach(n => {
            test(`round trip for n=${n}`, () => {
                const original = toComplex(random.random([n]));
                const f = original.fft();
                const inv = signal.ifft(f);
                original.data.forEach((v, i) => expect(v).toBeCloseTo(inv.data[i]));
            });
        });

        test('Parseval\'s theorem', () => {
            const n = 16;
            const a = toComplex(random.random([n]));
            const res = a.fft();
            const energyInTime = a.pow(2).sum();
            const energyInFreq = res.pow(2).sum() / n;
            expect(energyInTime).toBeCloseTo(energyInFreq);
        });

        test('fft of pure imaginary signal', () => {
            const r = ndarray.zeros([4]);
            const i = ndarray.array([1, 0, 0, 0]);
            const a = ndarray.stack([r, i], -1);
            const res = a.fft();
            // FFT of [i, 0, 0, 0] is [i, i, i, i]
            res.slice(null, 0).copy().data.forEach(v => expect(v).toBeCloseTo(0)); // real part is 0
            res.slice(null, 1).copy().data.forEach(v => expect(v).toBeCloseTo(1)); // imag part is 1
        });
        
        test('fft of single element array', () => {
            const a = toComplex(ndarray.array([5]));
            const res = a.fft();
            expect(res.get(0, 0)).toBeCloseTo(5);
            expect(res.get(0, 1)).toBeCloseTo(0);
        });

        test('fft of cosine wave', () => {
            const n = 16;
            const k = 2; // Frequency
            const r = ndarray.zeros([n]);
            for(let i=0; i<n; i++) r.set(Math.cos(2 * Math.PI * k * i / n), i);
            const a = toComplex(r);
            const res = a.fft();
            // Expect two peaks in the real part at k and N-k
            expect(res.get(k, 0)).toBeGreaterThan(n/3);
            expect(res.get(n-k, 0)).toBeGreaterThan(n/3);
            expect(res.get(k, 1)).toBeCloseTo(0);
        });

        test('ifft of shifted impulse', () => {
            const n = 8;
            const r = ndarray.zeros([n]);
            const i = ndarray.zeros([n]);
            r.set(n, 1); // Impulse at index 1
            const freq = ndarray.stack([r, i], -1);
            const res = signal.ifft(freq);
            // IFFT of a shifted impulse is a complex exponential (cisoid)
            // Real part is cosine, imag part is sine
            for(let j=0; j<n; j++) {
                expect(res.get(j, 0)).toBeCloseTo(Math.cos(2 * Math.PI * j / n)); // real part
                expect(res.get(j, 1)).toBeCloseTo(Math.sin(2 * Math.PI * j / n)); // imag part
            }
        });
    });

    // --- 2. rfft / rifft supplementary tests ---
    describe('rfft / rifft', () => {
        const cases = [7, 8, 15, 16];
        cases.forEach(n => {
            test(`round trip for n=${n}`, () => {
                const original = random.random([n]);
                const f = original.rfft();
                const inv = signal.rifft(f, n);
                original.data.forEach((v, i) => expect(v).toBeCloseTo(inv.data[i]));
            });
        });

        test('rfft of cosine wave', () => {
            const n = 16;
            const k = 3;
            const a = ndarray.zeros([n]);
            for(let i=0; i<n; i++) a.set(Math.cos(2 * Math.PI * k * i / n), i);
            const res = a.rfft();
            // Expect one peak in the real part at index k
            expect(res.get(k, 0)).toBeGreaterThan(n/3);
            expect(res.get(k, 1)).toBeCloseTo(0);
        });

        test('rfft of sine wave', () => {
            const n = 16;
            const k = 4;
            const a = ndarray.zeros([n]);
            for(let i=0; i<n; i++) a.set(Math.sin(2 * Math.PI * k * i / n), i);
            const res = a.rfft();
            // Expect one peak in the imag part at index k
            expect(res.get(k, 0)).toBeCloseTo(0);
            expect(res.get(k, 1)).toBeLessThan(-n/3); // Note: it's negative
        });

        test('rfft of single element array', () => {
            const a = ndarray.array([10]);
            const res = a.rfft();
            expect(res.shape).toEqual(new Int32Array([1, 2]));
            expect(res.get(0, 0)).toBeCloseTo(10);
            expect(res.get(0, 1)).toBeCloseTo(0);
        });

        test('rfft of two element array', () => {
            const a = ndarray.array([10, 20]);
            const res = a.rfft(); // Should produce [[30,0], [-10,0]]
            expect(res.get(0,0)).toBeCloseTo(30);
            expect(res.get(1,0)).toBeCloseTo(-10);
            expect(res.get(0,1)).toBeCloseTo(0);
            expect(res.get(1,1)).toBeCloseTo(0);
        });
    });

    // --- 3. fft2 / ifft2 supplementary tests ---
    describe('fft2 / ifft2', () => {
        const cases = [[3, 5], [8, 7], [4, 4]];
        cases.forEach(([rows, cols]) => {
            test(`round trip for shape ${rows}x${cols}`, () => {
                const original = toComplex(random.random([rows, cols]));
                const f = original.fft2();
                const inv = signal.ifft2(f);
                original.data.forEach((v, i) => expect(v).toBeCloseTo(inv.data[i]));
            });
        });

        test('2D impulse at non-zero location', () => {
            const r = ndarray.zeros([4, 4]);
            r.set(1, 1, 2); // Impulse at (1, 2)
            const a = toComplex(r);
            const res = a.fft2();
            // Magnitude should be constant (1), phase varies
            for(let i=0; i<4; i++) {
                for (let j=0; j<4; j++) {
                    const mag = Math.sqrt(res.get(i,j,0)**2 + res.get(i,j,1)**2);
                    expect(mag).toBeCloseTo(1);
                }
            }
        });

        test('Parseval\'s theorem in 2D', () => {
            const a = toComplex(random.random([4, 5]));
            const res = a.fft2();
            const energyInTime = a.pow(2).sum();
            const energyInFreq = res.pow(2).sum() / (4 * 5);
            expect(energyInTime).toBeCloseTo(energyInFreq);
        });

        test('fft2 on single row matrix', () => {
            const a = toComplex(random.random([1, 8]));
            const res = a.fft2();
            expect(res.shape).toEqual(new Int32Array([1, 8, 2]));
        });

        test('fft2 on single col matrix', () => {
            const a = toComplex(random.random([8, 1]));
            const res = a.fft2();
            expect(res.shape).toEqual(new Int32Array([8, 1, 2]));
        });
    });

    // --- 4. dct supplementary tests ---
    describe('dct', () => {
        const cases = [7, 15, 100];
        cases.forEach(n => {
            test(`linearity for n=${n}`, () => {
                const x = random.random([n]);
                const y = random.random([n]);
                const a = 2.5, b = -3.0;
                const res1 = x.mul(a).add(y.mul(b)).dct();
                const res2 = x.dct().mul(a).add(y.dct().mul(b));
                res1.data.forEach((v, i) => expect(v).toBeCloseTo(res2.data[i]));
            });
        });

        test('dct of constant signal', () => {
            const a = ndarray.ones([8]).mul(5); // array of 5s
            const res = a.dct();
            // Only DC component should be non-zero
            expect(res.get(0)).not.toBeCloseTo(0);
            for(let i=1; i<8; i++) {
                expect(res.get(i)).toBeCloseTo(0);
            }
        });
        
        test('dct of single element array', () => {
            const a = ndarray.array([10]);
            const res = a.dct();
            expect(res.get(0)).toBeCloseTo(10); // DCT of a single element is itself
        });
    });

    // --- 5. conv2d / correlate2d supplementary tests ---
    describe('conv2d / correlate2d', () => {
        test('conv2d with non-square image and kernel', () => {
            const img = ndarray.ones([5, 10]);
            const kernel = ndarray.ones([2, 3]);
            const res = img.conv2d(kernel);
            expect(res.shape).toEqual(new Int32Array([4, 8]));
            // Sum of kernel is 6
            res.data.forEach(v => expect(v).toBeCloseTo(6));
        });

        test('conv2d with stride > 1', () => {
            const img = ndarray.arange(16).reshape([4, 4]);
            const kernel = ndarray.ones([2, 2]);
            const res = img.conv2d(kernel, 2, 0); // Stride 2
            // Expected: [[0+1+4+5, 2+3+6+7], [8+9+12+13, 10+11+14+15]]
            // = [[10, 18], [42, 50]]
            expect(res.data).toEqual(new Float64Array([10, 18, 42, 50]));
        });

        test('conv2d with kernel larger than image', () => {
            const img = ndarray.ones([2, 2]);
            const kernel = ndarray.ones([3, 3]);
            const res = img.conv2d(kernel, 1, 0);
            expect(res.shape).toEqual(new Int32Array([0, 0]));
        });

        test('conv2d with 1xN kernel', () => {
            const img = ndarray.ones([3, 3]);
            const kernel = ndarray.ones([1, 2]);
            const res = img.conv2d(kernel);
            expect(res.shape).toEqual(new Int32Array([3, 2]));
            res.data.forEach(v => expect(v).toBeCloseTo(2));
        });

        test('correlate2d vs conv2d with flipped kernel', () => {
            const img = random.random([5, 5]);
            const kernel = random.random([3, 3]);
            const kernelFlipped = ndarray.zeros(kernel.shape);
            for(let i=0; i<3; i++) {
                for(let j=0; j<3; j++) {
                    kernelFlipped.set(kernel.get(2-i, 2-j), i, j);
                }
            }
            const resCorr = img.correlate2d(kernel, 1, 0);
            const resConv = img.conv2d(kernelFlipped, 1, 0);
            resCorr.data.forEach((v, i) => expect(v).toBeCloseTo(resConv.data[i]));
        });
    });
});
