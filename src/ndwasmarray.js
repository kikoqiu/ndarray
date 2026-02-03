/**
 * File: ndwasmarray.js
 * Responsibility: High-performance WASM-resident NDArray.
 * Handles explicit memory management within the WASM heap. 
 * Users must call .dispose() manually to free memory.
 */

import { NDArray } from './ndarray_core.js';
import { NDWasm } from './ndwasm.js';

/**
 * NDWasmArray
 */
export class NDWasmArray {
    /**
     * @param {WasmBuffer} buffer - The WASM memory bridge (contains .ptr and .view).
     * @param {Int32Array|Array} shape - Dimensions of the array.
     * @param {string} dtype - Data type (e.g., 'float64').
     */
    constructor(buffer, shape, dtype) {
        this.buffer = buffer;
        this.shape = shape instanceof Int32Array ? shape : Int32Array.from(shape);
        this.dtype = dtype;
        this.ndim = this.shape.length;
        this.size = this.ndim === 0 ? 1 : this.shape.reduce((a, b) => a * b, 1);
    }

    /**
     * Static factory: Creates an uninitialized WASM-resident array.
     * @param {*} shape - Shape of the array.
     * @param {*} dtype - Data type.
     * @returns {NDWasmArray}
     */
    static newArray(shape, dtype = 'float64') {
        const size = shape.reduce((a, b) => a * b, 1);
        const buffer = NDWasm.runtime.createBuffer(size, dtype);
        return new NDWasmArray(buffer, shape, dtype);
    }

    /**
     * Static factory: Creates a WASM-resident array.
     * 1. If source is an NDArray, it calls .push() to move it to WASM.
     * 2. If source is a JS Array, it allocates WASM memory and fills it directly 
     *    via recursive traversal to avoid intermediate flattening.
     * 3. If source is a single number, it creates a 1-element WASM array.
     * Must dispose() manually to free WASM memory.
     * @param {NDArray|Array|number} source - Source data.
     * @param {string} [dtype='float64'] - Data type.
     * @returns {NDWasmArray}
     */
    static fromArray(source, dtype = 'float64') {
        // Handle existing NDArray instance
        if (source instanceof NDArray) {
            return source.push();
        }

        if (!NDWasm.runtime?.isLoaded) {
            throw new Error("WasmRuntime not initialized. Call NDWasm.bind(runtime) first.");
        }

        // Handle standard JS Arrays
        if (Array.isArray(source)) {
            // 1. Infer shape and total size
            const shape = [];
            let curr = source;
            while (Array.isArray(curr)) {
                shape.push(curr.length);
                curr = curr[0];
            }
            const size = shape.length === 0 ? 0 : shape.reduce((a, b) => a * b, 1);

            // 2. Pre-allocate memory in WASM heap
            const buffer = NDWasm.runtime.createBuffer(size, dtype);

            // 3. Populate WASM memory directly via recursive traversal
            let offset = 0;
            const fill = (arr) => {
                for (let i = 0; i < arr.length; i++) {
                    if (Array.isArray(arr[i])) {
                        fill(arr[i]);
                    } else {
                        buffer.view[offset++] = arr[i];
                    }
                }
            };
            fill(source);

            return new NDWasmArray(buffer, shape, dtype);
        }

        // Handle single numeric value
        if (typeof source === 'number') {
            const buffer = NDWasm.runtime.createBuffer(1, dtype);
            buffer.view[0] = source;
            return new NDWasmArray(buffer, [1], dtype);
        }

        throw new Error("Source must be an Array or an NDArray.");
    }

    /**
     * Pulls data from WASM to a JS-managed NDArray.
     * @param {boolean} [dispose=true] - Release WASM memory after pulling.
     */
    pull(dispose = true) {
        if (!this.buffer) throw new Error("WASM memory already disposed.");
        
        const data = this.buffer.pull(); 
        const result = new NDArray(data, { shape: this.shape, dtype: this.dtype });

        if (dispose) this.dispose();
        return result;
    }

    /**
     * Manually releases WASM heap memory.
     */
    dispose() {
        if (this.buffer) {
            this.buffer.dispose();
            this.buffer = null;
        }
    }

    /**
     * Matrix Multiplication: C = this * right
     * @param {NDWasmArray} right
     * @param {NDWasmArray} result - Pre-allocated result array.
     * @returns {NDWasmArray} the result array.
     */
    matMul(right, result) {
        if(!(right instanceof NDWasmArray)) {
            throw new Error("Right operand must be an NDWasmArray.");
        }
        
        if (this.shape[1] !== right.shape[0]) {
            throw new Error(`Inner dimensions mismatch: ${this.shape[1]} != ${right.shape[0]}`);
        }

        if(!result || !(result instanceof NDWasmArray) || result.dtype !== this.dtype) {
            throw new Error(`Invalid result array. Expected NDWasmArray with dtype ${this.dtype}.`);
        }else if(result.shape[0] !== this.shape[0] || result.shape[1] !== right.shape[1] ||result.ndim !== 2) {
            throw new Error(`Result array has incorrect shape. Expected [${this.shape[0]}, ${right.shape[1]}], got [${result.shape[0]}, ${result.shape[1]}].`);
        }      


        const m = this.shape[0];
        const n = this.shape[1];
        const k = right.shape[1];
        const suffix = NDWasm.runtime._getSuffix(this.dtype);

        const status = NDWasm.runtime.exports[`MatMul${suffix}`](
            this.buffer.ptr, 
            right.buffer.ptr, 
            result.buffer.ptr, 
            m, n, k
        );

        if (status !==undefined && status !== 0) throw new Error(`WASM MatMul failed with status: ${status}`);

        return result;        
    }

    /**
     * Batched Matrix Multiplication: C[i] = this[i] * other[i]
     * @param {NDWasmArray} other
     * @param {NDWasmArray} result - Pre-allocated result array.
     * @returns {NDWasmArray}
     */
    matMulBatch(right, result) {
        if(!(right instanceof NDWasmArray)) {
            throw new Error("Right operand must be an NDWasmArray.");
        }
        if (this.ndim !== 3 || right.ndim !== 3 || this.shape[0] !== right.shape[0]) {
            throw new Error(`dimensions mismatch. ${this.ndim}D and ${right.ndim}D arrays with batch size ${this.shape[0]} and ${right.shape[0]}.`);
        }

        if(!result || !(result instanceof NDWasmArray) || result.dtype !== this.dtype){
            throw new Error(`Invalid result array. Expected NDWasmArray with dtype ${this.dtype}.`);
        }else if(result.shape[0] !== this.shape[0] || result.shape[1] !== this.shape[1] || result.shape[2] !== right.shape[2] || result.ndim !== 3) {
            throw new Error(`Invalid result array. Expected shape [${this.shape[0]}, ${this.shape[1]}, ${right.shape[2]}], got [${result.shape[0]}, ${result.shape[1]}, ${result.shape[2]}].`);
        }

        const batch = this.shape[0];
        const m = this.shape[1];
        const n = this.shape[2];
        const k = right.shape[2];
        const suffix = NDWasm.runtime._getSuffix(this.dtype);

        const status = NDWasm.runtime.exports[`MatMulBatch${suffix}`](
            this.buffer.ptr,
            right.buffer.ptr,
            result.buffer.ptr,
            batch, m, n, k
        );

        if (status !==undefined && status !== 0) throw new Error(`WASM MatMulBatch failed with status: ${status}`);

        return result;
    }
}

