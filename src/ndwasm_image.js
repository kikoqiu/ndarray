import { NDArray, DTYPE_MAP } from './ndarray_core.js';
import NDWasm from './ndwasm.js';

/**
 * @namespace NDWasmImage
 * @description Provides WebAssembly-powered functions for image processing.
 * This module allows for efficient conversion between image binary data and NDArrays.
 */
export const NDWasmImage = {
    /**
     * Decodes a binary image into an NDArray.
     * Supports common formats like PNG, JPEG, GIF, and WebP.
     * The resulting NDArray will have a shape of [height, width, 4] and a 'uint8c' dtype,
     * representing RGBA channels. This provides a consistent starting point for image manipulation.
     *
     * @memberof NDWasmImage
     * @param {Uint8Array} imageBytes - The raw binary data of the image file.
     * @returns {NDArray|null} A 3D NDArray representing the image, or null if decoding fails.
     * @example
     * const imageBlob = await fetch('./my-image.png').then(res => res.blob());
     * const imageBytes = new Uint8Array(await imageBlob.arrayBuffer());
     * const imageArray = NDWasmImage.decode(imageBytes);
     * // imageArray.shape is [height, width, 4]
     */
    decode(imageBytes) {
        if (!NDWasm.runtime || !NDWasm.runtime.isLoaded) {
            throw new Error("NDWasm runtime is not initialized. Call NDWasm.init() first.");
        }
        if (!(imageBytes instanceof Uint8Array)) {
            throw new Error("Input must be a Uint8Array.");
        }

        const exports = NDWasm.runtime.exports;
        let imagePtr = 0;
        let resultPtr = 0;
        let pixelDataPtr = 0;

        try {
            // 1. Allocate memory for the input image and copy it to WASM.
            imagePtr = exports.malloc(imageBytes.length);
            if (!imagePtr) throw new Error("WASM malloc failed for image bytes.");
            new Uint8Array(exports.mem.buffer, imagePtr, imageBytes.length).set(imageBytes);

            // 2. Call the WASM decoding function.
            resultPtr = exports.decode_image(imagePtr, imageBytes.length);
            if (!resultPtr) {
                console.error("Failed to decode image in WASM. The image format may be invalid or unsupported.");
                return null;
            }

            // 3. Read the result structure [pixel_ptr, pixel_size, width, height] from WASM memory.
            const resultView = new DataView(exports.mem.buffer, resultPtr, 16);
            pixelDataPtr = resultView.getUint32(0, true);
            const pixelDataSize = resultView.getUint32(4, true);
            const width = resultView.getUint32(8, true);
            const height = resultView.getUint32(12, true);

            if (!pixelDataPtr) {
                console.error("WASM decode_image returned a null pixel data pointer.");
                return null;
            }

            // 4. Create a copy of the pixel data from WASM memory.
            const pixelData = new Uint8ClampedArray(exports.mem.buffer, pixelDataPtr, pixelDataSize).slice();

            // 5. Create the final NDArray.
            return new NDArray(pixelData, { shape: [height, width, 4], dtype: 'uint8c' });

        } finally {
            // 6. Free all allocated WASM memory.
            if (imagePtr) exports.free(imagePtr, imageBytes.length);
            if (resultPtr) exports.free(resultPtr, 16);
            if (pixelDataPtr) {
                const resultView = new DataView(exports.mem.buffer, resultPtr, 16);
                const pixelDataSize = resultView.getUint32(4, true);
                exports.free(pixelDataPtr, pixelDataSize);
            }
        }
    },

    /**
     * Encodes an NDArray into a binary image format (PNG or JPEG).
     *
     * @memberof NDWasmImage
     * @param {NDArray} ndarray - The input array.
     *   Supported dtypes: 'uint8', 'uint8c', 'float32', 'float64'. Float values should be in the range [0, 1].
     *   Supported shapes: [h, w] (grayscale), [h, w, 1] (grayscale), [h, w, 3] (RGB), or [h, w, 4] (RGBA).
     * @param {object} [options={}] - Encoding options.
     * @param {string} [options.format='png'] - The target format: 'png' or 'jpeg'. It is recommended to use the `encodePng` or `encodeJpeg` helpers instead.
     * @param {number} [options.quality=90] - The quality for JPEG encoding (1-100). Ignored for PNG.
     * @returns {Uint8Array|null} A Uint8Array containing the binary data of the encoded image, or null on failure.
     * @see {@link NDWasmImage.encodePng}
     * @see {@link NDWasmImage.encodeJpeg}
     * @example
     * // Encode a 3-channel float array to a high-quality JPEG using the main function
     * const floatArr = NDArray.random([100, 150, 3]);
     * const jpegBytes = NDWasmImage.encode(floatArr, { format: 'jpeg', quality: 95 });
     */
    encode(ndarray, { format = 'png', quality = 90 } = {}) {
        if (!NDWasm.runtime || !NDWasm.runtime.isLoaded) {
            throw new Error("NDWasm runtime is not initialized. Call NDWasm.init() first.");
        }
        if (!(ndarray instanceof NDArray)) {
            throw new Error("Input must be an NDArray.");
        }

        // --- 1. Validate and Pre-process NDArray ---
        
        // Validate dimensions and channels
        let channels;
        if (ndarray.ndim === 2) {
            channels = 1;
        } else if (ndarray.ndim === 3) {
            channels = ndarray.shape[2];
            if (channels !== 1 && channels !== 3 && channels !== 4) {
                throw new Error(`Unsupported channel count for 3D array: ${channels}. Must be 1, 3, or 4.`);
            }
        } else {
            throw new Error(`Unsupported array dimensions: ${ndarray.ndim}. Must be 2 or 3.`);
        }

        let arrayToEncode = ndarray;
        
        // Handle float dtypes by scaling to uint8
        if (ndarray.dtype === 'float32' || ndarray.dtype === 'float64') {
            const scaledData = new Uint8Array(ndarray.size);
            // This is a simple but slow way to iterate. A faster, vectorized `mul` and `astype` would be an improvement.
            let i = 0;
            ndarray.iterate(value => {
                scaledData[i++] = Math.max(0, Math.min(255, value * 255));
            });
            arrayToEncode = new NDArray(scaledData, { shape: ndarray.shape, dtype: 'uint8' });
        }
        
        // Ensure data is contiguous C-style for easy passing to WASM
        let sourceData = arrayToEncode.data;
        if (!arrayToEncode.isContiguous) {
            const flatData = new (DTYPE_MAP[arrayToEncode.dtype])(arrayToEncode.size);
            let i = 0;
            arrayToEncode.iterate(v => flatData[i++] = v);
            sourceData = flatData;
        }

        // --- 2. Interact with WASM ---
        
        const exports = NDWasm.runtime.exports;
        let pixelDataPtr = 0;
        let formatPtr = 0;
        let resultPtr = 0;
        let encodedDataPtr = 0;
        
        const [height, width] = arrayToEncode.shape;
        const formatBytes = new TextEncoder().encode(format);

        try {
            // Allocate and copy pixel data to WASM.
            pixelDataPtr = exports.malloc(sourceData.byteLength);
            if (!pixelDataPtr) throw new Error("WASM malloc failed for pixel data.");
            new Uint8Array(exports.mem.buffer, pixelDataPtr, sourceData.byteLength).set(sourceData);

            // Allocate and copy format string to WASM.
            formatPtr = exports.malloc(formatBytes.length);
            if (!formatPtr) throw new Error("WASM malloc failed for format string.");
            new Uint8Array(exports.mem.buffer, formatPtr, formatBytes.length).set(formatBytes);

            // Call the WASM encoding function.
            resultPtr = exports.encode_image(pixelDataPtr, width, height, channels, quality, formatPtr, formatBytes.length);
            if (!resultPtr) {
                console.error(`Failed to encode image to ${format} in WASM.`);
                return null;
            }

            // Read the result structure [encoded_ptr, encoded_size].
            const resultView = new DataView(exports.mem.buffer, resultPtr, 8);
            encodedDataPtr = resultView.getUint32(0, true);
            const encodedDataSize = resultView.getUint32(4, true);

            if (!encodedDataPtr) {
                console.error("WASM encode_image returned a null data pointer.");
                return null;
            }

            // Create a copy of the encoded image data.
            const encodedData = new Uint8Array(exports.mem.buffer, encodedDataPtr, encodedDataSize).slice();
            
            return encodedData;

        } finally {
            // Free all allocated WASM memory.
            if (pixelDataPtr) exports.free(pixelDataPtr, sourceData.byteLength);
            if (formatPtr) exports.free(formatPtr, formatBytes.length);
            if (resultPtr) exports.free(resultPtr, 8);
            if (encodedDataPtr) {
                const resultView = new DataView(exports.mem.buffer, resultPtr, 8);
                const encodedDataSize = resultView.getUint32(4, true);
                exports.free(encodedDataPtr, encodedDataSize);
            }
        }
    },

    /**
     * Encodes an NDArray into a PNG image.
     * This is a helper function that calls `encode` with `format: 'png'`.
     * @memberof NDWasmImage
     * @param {NDArray} ndarray - The input array. See `encode` for supported shapes and dtypes.
     * @returns {Uint8Array|null} A Uint8Array containing the binary data of the PNG image.
     * @example
     * const grayArr = ndarray.zeros([16, 16]);
     * const pngBytes = NDWasmImage.encodePng(grayArr);
     */
    encodePng(ndarray) {
        return this.encode(ndarray, { format: 'png' });
    },

    /**
     * Encodes an NDArray into a JPEG image.
     * This is a helper function that calls `encode` with `format: 'jpeg'`.
     * @memberof NDWasmImage
     * @param {NDArray} ndarray - The input array. See `encode` for supported shapes and dtypes.
     * @param {object} [options={}] - Encoding options.
     * @param {number} [options.quality=90] - The quality for JPEG encoding (1-100).
     * @returns {Uint8Array|null} A Uint8Array containing the binary data of the JPEG image.
     * @example
     * const floatArr = NDArray.random([20, 20, 3]);
     * const jpegBytes = NDWasmImage.encodeJpeg(floatArr, { quality: 85 });
     */
    encodeJpeg(ndarray, options = {}) {
        return this.encode(ndarray, { ...options, format: 'jpeg' });
    },
    
    /**
     * Converts a Uint8Array of binary data into a Base64 Data URL.
     * This is a utility function that runs purely in JavaScript.
     *
     * @memberof NDWasmImage
     * @param {Uint8Array} uint8array - The byte array to convert.
     * @param {string} [mimeType='image/png'] - The MIME type for the Data URL (e.g., 'image/jpeg').
     * @returns {string} The complete Data URL string.
     * @example
     * const pngBytes = NDWasmImage.encodePng(myNdarray);
     * const dataUrl = NDWasmImage.convertUint8ArrrayToDataurl(pngBytes, 'image/png');
     * // <img src={dataUrl} />
     */
    convertUint8ArrrayToDataurl(uint8array, mimeType = 'image/png') {
        // In a browser environment
        if (typeof window !== 'undefined' && typeof window.btoa === 'function') {
            let binary = '';
            const len = uint8array.byteLength;
            for (let i = 0; i < len; i++) {
                binary += String.fromCharCode(uint8array[i]);
            }
            const base64 = window.btoa(binary);
            return `data:${mimeType};base64,${base64}`;
        }
        // In a Node.js environment
        if (typeof Buffer !== 'undefined') {
            const base64 = Buffer.from(uint8array).toString('base64');
            return `data:${mimeType};base64,${base64}`;
        }
        
        throw new Error("Unsupported environment: btoa or Buffer not available.");
    }
};

export default NDWasmImage;