/**
 * File: ndwasm_image.test.js
 * Responsibility: Test suite for WASM-based image encoding and decoding.
 */

// 1. Import modules from the distribution bundle
const ndarray = require('../dist/ndarray.cjs');
const { NDArray, NDWasm, image, WasmRuntime } = ndarray;

describe('image (WASM)', () => {

    // 2. Initialize the WASM runtime before any tests run.
    beforeAll(async () => {
        const runtime = new WasmRuntime();
        // Point to the build artifacts in the dist directory
        await runtime.init({
            execUrl: 'dist/wasm_exec.js',
            wasmUrl: 'dist/ndarray_plugin.wasm'
        });
        NDWasm.bind(runtime);
    }, 30000); // 30-second timeout for WASM initialization

    // --- 3. Test the helper functions ---
    describe('encodePng', () => {
        it('should encode a 4-channel uint8c array to a valid PNG', () => {
            const arr = ndarray.zeros([10, 10, 4], 'uint8c');
            const pngBytes = image.encodePng(arr);
            
            expect(pngBytes).toBeInstanceOf(Uint8Array);
            expect(pngBytes.length).toBeGreaterThan(10);
            expect(Array.from(pngBytes.slice(0, 8))).toEqual([137, 80, 78, 71, 13, 10, 26, 10]);
        });

        it('should encode a 2D grayscale uint8 array to PNG', () => {
            const arr = ndarray.zeros([8, 12], 'uint8');
            const pngBytes = image.encodePng(arr);

            expect(pngBytes).toBeInstanceOf(Uint8Array);
            expect(pngBytes.length).toBeGreaterThan(10);
            expect(Array.from(pngBytes.slice(0, 8))).toEqual([137, 80, 78, 71, 13, 10, 26, 10]);
        });
    });

    describe('encodeJpeg', () => {
        it('should encode a 3-channel float32 array to a valid JPEG', () => {
            const arr = ndarray.full([5, 5, 3], 0.5, 'float32');
            const jpegBytes = image.encodeJpeg(arr, { quality: 80 });

            expect(jpegBytes).toBeInstanceOf(Uint8Array);
            expect(jpegBytes.length).toBeGreaterThan(10);
            // Check for JPEG magic bytes: FF D8 (start) and FF D9 (end)
            expect(jpegBytes[0]).toBe(255);
            expect(jpegBytes[1]).toBe(216);
            expect(jpegBytes[jpegBytes.length - 2]).toBe(255);
            expect(jpegBytes[jpegBytes.length - 1]).toBe(217);
        });
    });    


    // --- 5. Test the 'decode' function (and its interplay with encode) ---
    describe('decode', () => {
        it('should decode a PNG and recover the original array data', () => {
            const originalData = new Uint8ClampedArray([
                255, 0, 0, 255,    0, 255, 0, 255,
                0, 0, 255, 255,    128, 128, 128, 255,
            ]);
            const originalArr = new NDArray(originalData, { shape: [2, 2, 4], dtype: 'uint8c' });

            const pngBytes = image.encodePng(originalArr);
            expect(pngBytes).not.toBeNull();

            const decodedArr = image.decode(pngBytes);
            
            expect(decodedArr).toBeInstanceOf(NDArray);
            expect(decodedArr.shape).toEqual(new Int32Array([2, 2, 4]));
            expect(decodedArr.dtype).toBe('uint8c');
            expect(decodedArr.data).toEqual(originalData);
        });

        it('should decode a JPEG (with lossy compression)', () => {
            const arr = ndarray.zeros([16, 16, 3], 'float64');
            const height = arr.shape[0];
            const width = arr.shape[1];
            
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    arr.set(x / 15.0, y, x, 0); // Red gradient
                    arr.set(y / 15.0, y, x, 1); // Green gradient
                    arr.set(0.5,      y, x, 2); // Blue constant
                }
            }

            const jpegBytes = image.encodeJpeg(arr, { quality: 95 });
            expect(jpegBytes).not.toBeNull();

            const decodedArr = image.decode(jpegBytes);
            expect(decodedArr).toBeInstanceOf(NDArray);
            expect(decodedArr.shape).toEqual(new Int32Array([16, 16, 4]));
            
            const originalPixel = [Math.round(8/15 * 255), Math.round(4/15 * 255), Math.round(0.5 * 255)]; // y=4, x=8
            
            expect(decodedArr.get(4, 8, 0)).toBeCloseTo(originalPixel[0], -1);
            expect(decodedArr.get(4, 8, 1)).toBeCloseTo(originalPixel[1], -1);
        });
    });

    // --- 6. Test the utility function ---
    describe('convertUint8ArrrayToDataurl', () => {
        it('should convert a byte array to a valid data URL', () => {
            const bytes = new Uint8Array([72, 101, 108, 108, 111]); 
            const dataUrl = image.convertUint8ArrrayToDataurl(bytes, 'text/plain');

            expect(dataUrl).toBe('data:text/plain;base64,SGVsbG8=');
        });
    });
});