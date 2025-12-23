/**
 * ndwasm.js
 * Responsibility: Loading WASM, memory buffer management, NDArray bridge extensions.
 */
import { NDArray, DTYPE_MAP } from './ndarray_core.js';



/**
 * WasmBuffer
 */
class WasmBuffer {
    /**
     * @param {Object} exports - The exports object from Go WASM.
     * @param {number} size - Number of elements.
     * @param {string} dtype - Data type (float64, float32, etc.).
     */
    constructor(exports, size, dtype) {
        this.exports = exports;
        this.size = size;
        this.dtype = dtype;
        
        const Constructor = DTYPE_MAP[dtype];
        if (!Constructor) throw new Error(`Unsupported dtype: ${dtype}`);
        
        this.byteLength = size * Constructor.BYTES_PER_ELEMENT;
        
        // 1. Allocate memory in the WASM heap.
        if(this.byteLength === 0){
            this.ptr = 0;
        }else{
            this.ptr = this.exports.malloc(this.byteLength);
            if (!this.ptr) throw new Error(`WASM malloc failed to allocate ${this.byteLength} bytes`);
        }
        this.refresh();
    }

    /**
     * Refreshes the view into WASM memory.
     */
    refresh(){
        const Constructor = DTYPE_MAP[this.dtype];
        // Create a direct view into WASM memory (sharing the same ArrayBuffer).
        this.view = new Constructor(this.exports.mem.buffer, this.ptr, this.size);
        return this;
    }

    /** Synchronizes JS data into the WASM buffer. */
    push(typedArray) {
        this.refresh();
        this.view.set(typedArray);
    }

    /** Pulls data from the WASM buffer back to JS (returns a copy).
     * @returns {NDArray}
     */
    pull() {
        this.refresh();
        return this.view.slice(); 
    }

    /** Disposes of the temporary buffer. */
    dispose() {
        if (this.ptr) {
            this.exports.free(this.ptr);
            this.ptr = null;
            this.view = null;
        }
    }
}

/**
 * WasmRuntime
 */
class WasmRuntime {
    constructor() {
        this.instance = null;
        this.exports = null;
        this.isLoaded = false;
    }

    /**
     * Initializes the Go WASM environment.
     * @param {object} [options] - Optional configuration.
     * @param {string} [options.wasmUrl='./ndarray_plugin.wasm'] - Path or URL to the wasm file.
     * @param {string} [options.execUrl='./wasm_exec.js'] - Path to the wasm_exec.js file (Node.js only).
     * @param {number} [options.initialMemoryPages] - Initial memory size in 64KiB pages.
     * @param {number} [options.maximumMemoryPages] - Maximum memory size in 64KiB pages.
     * @param {string} [options.baseDir='.']
     */
    async init(options = {}) {
        options = {
            initialMemoryPages : 1024*1024*64 / (64*1024),
            maximumMemoryPages : 1024*1024*512 / (64*1024),
            ...options};
        const wasmUrl = options.wasmUrl || `${options.baseDir || '.'}/ndarray_plugin.wasm`;
        const execUrl = options.execUrl || `${options.baseDir || '.'}/wasm_exec.js`;

        // Create the WebAssembly.Memory object
        const wasmMemory = new WebAssembly.Memory({
            initial: options.initialMemoryPages,
            maximum: options.maximumMemoryPages
        });


        // Environment-specific loading
        const isNode = typeof window === 'undefined';

        if (isNode) {
            // Node.js environment
            const fs = require('fs');
            const path = require('path');
            
            // The wasm_exec.js from Go requires these globals.
            global.fs = fs;
            global.util = require('util');
            
            // Load and execute the wasm_exec.js script to define the Go global.
            require(path.resolve(process.cwd(), execUrl));

            const wasmBytes = fs.readFileSync(path.resolve(process.cwd(), wasmUrl));
            const go = new Go();
            // Add the memory object to the import object
            go.env['mem'] = wasmMemory;

            const { instance } = await WebAssembly.instantiate(wasmBytes, go.importObject);
            this.instance = instance;
            go.run(this.instance);
        } else {
            // Browser environment
            try{
                await importScripts(execUrl);
            }catch(e){
                console.log(e);
            }
            if (typeof Go === 'undefined') {
                console.error("Go's wasm_exec.js must be loaded with a <script> tag before initializing WasmRuntime in a browser.");
                throw new Error("Missing Go global object.");
            }
            
            const go = new Go();
            // Add the memory object to the import object
            go.env['mem'] = wasmMemory;
            const result = await WebAssembly.instantiateStreaming(fetch(wasmUrl), go.importObject);
            this.instance = result.instance;
            // The go.run promise resolves when the WASM exits, which in a reactor model is never. We don't await it.
            go.run(this.instance);
        }

        this.exports = this.instance.exports;
        this.isLoaded = true;
    }

    /**
     * Helper method: Gets the suffix for Go export function names based on type.
     * @private
     */
    _getSuffix(dtype) {
        if (dtype === 'float64') return '_F64';
        if (dtype === 'float32') return '_F32';
        throw new Error(`Go-Wasm compute does not support dtype: ${dtype}. Use float64 or float32.`);
    }

    /**
     * Quickly allocates a buffer.
     * @returns {WasmBuffer}
     */
    createBuffer(size, dtype) {
        return new WasmBuffer(this.exports, size, dtype);
    }
}


/**
 * Static factory: Creates a new NDArray directly from WASM computation results.
 * @param {WasmBuffer} bridge - The WasmBuffer instance containing the result from WASM.
 * @param {Array<number>} shape - The shape of the new NDArray.
 * @param {string} [dtype] - The data type of the new NDArray. If not provided, uses bridge.dtype.
 * @returns {NDArray} The new NDArray.
 */
export function fromWasm(bridge, shape, dtype) {
    const data = bridge.pull(); // Data is a flat TypedArray pulled from WASM.
    // Call the constructor directly as we already have the flat data array and the correct shape.
    return new NDArray(data, { shape, dtype: dtype || bridge.dtype });
};


// --- Main WASM Bridge Object ---
import { NDWasmBlas } from './ndwasm_blas.js';
import { NDWasmDecomp } from './ndwasm_decomp.js';
import { NDWasmSignal } from './ndwasm_signal.js';
import { NDWasmAnalysis } from './ndwasm_analysis.js';

// --- Main WASM Bridge Object ---
/**
 * @namespace NDWasm
 */
const NDWasm = {
    runtime: null,

    /** Binds a loaded WASM runtime to the bridge.
     * @param {*} runtime 
     */
    bind(runtime) {
        this.runtime = runtime;
    },
    /**
     * Init the NDWasm
     * @param {string} [baseDir='.'] 
     */
    async init(baseDir = '.'){
        const runtime = new WasmRuntime();
        await runtime.init({baseDir});
        NDWasm.bind(runtime);
    },

    /** Internal helper: executes a computation in WASM and manages memory. */
    _compute(inputs, outShape, outDtype, computeFn) {
        if (!this.runtime || !this.runtime.isLoaded) throw new Error("WasmRuntime not initialized");
        
        // 1. Convert all input NDArrays to WASM buffer bridges.
        const bridges = inputs.map(arr => arr.toWasm(this.runtime));
        // 2. Prepare the output buffer bridge.
        const size = outShape.reduce((a, b) => a * b, 1);
        const outBridge = this.runtime.createBuffer(size, outDtype);

        try {
            // 3. Execute the Go function, passing pointers.
            const status = computeFn(...bridges.map(b => b.ptr), outBridge.ptr);
            if (status !== undefined && status !== 0) {
                throw new Error(`WASM compute failed with status code: ${status}`);
            }
            // 4. Create a new JS NDArray by copying the data back from the output bridge.
            return fromWasm(outBridge, outShape, outDtype);
        } finally {
            // 5. Free all temporary WASM memory.
            bridges.forEach(b => b.dispose());
            outBridge.dispose();
        }
    },
};


export const blas=NDWasmBlas
export const decomp=NDWasmDecomp;
export const signal=NDWasmSignal;
export const analysis=NDWasmAnalysis;

export { NDWasm, WasmRuntime, WasmBuffer };
export default NDWasm;
