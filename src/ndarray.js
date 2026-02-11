/**
 * Main entry point for the ndarray library.
 * This file imports all components, orchestrates the help system, and exports the public API.
 */

// 1. Import library components
import { NDArray, DTYPE_MAP } from './ndarray_core.js';
import * as ndwasmarray from './ndwasmarray.js';
import * as ndwasm from './ndwasm.js';

import { NDProb } from "./ndarray_prob.js";
import { NDWasmDecomp } from "./ndwasm_decomp.js";
import { NDWasmAnalysis } from "./ndwasm_analysis.js";
import { NDWasmBlas } from "./ndwasm_blas.js";
import { NDWasmSignal } from "./ndwasm_signal.js";
import { NDWasmImage } from "./ndwasm_image.js";
import { NDWasmOptimize } from './ndwasm_optimize.js';

import * as factory from "./ndarray_factory.js"
import * as helpers from "./ndarray_helpers.js"


// 2. Import help system and documentation data
import docMap from './docs.json';
import { help } from './help.js';

// 3. Register all documented objects with the help system
function registerAll() {
    // Define the top-level objects that are exported and documented
    const rootObjects = { 
        NDArray, 
        ...ndwasmarray,
        ...ndwasm,
        NDProb,
        NDWasmDecomp,
        NDWasmAnalysis,
        NDWasmBlas,
        NDWasmSignal,
        NDWasmImage,
        NDWasmOptimize,
        ...factory,
        ...helpers
    };

    for (const name in docMap) {
        const parts = name.split('.');
        const rootName = parts[0];
        
        if (!rootObjects[rootName]) continue;

        let obj = rootObjects[rootName];
        
        // Traverse the object path (e.g., 'prototype', 'reshape')
        for (let i = 1; i < parts.length; i++) {
            obj = obj[parts[i]];
            if (!obj) break;
        }

        // If we found a live object, register it in the helpmap.
        if (obj) {
            help.helpmap.set(obj, name);
        }
    }
}
// Run registration logic once on startup.
registerAll();




export {
    NDProb,
    NDWasmDecomp,
    NDWasmAnalysis,
    NDWasmBlas,
    NDWasmSignal,
    NDWasmImage,
    NDWasmOptimize
};
export { help };

export {
    NDProb as random,
    NDWasmDecomp as decomp,
    NDWasmAnalysis as analysis,
    NDWasmBlas as blas,
    NDWasmSignal as signal,
    NDWasmImage as image,
    NDWasmOptimize as optimize
};

export { Jit } from "./ndarray_jit.js";

export * from "./ndwasm.js";
export * from "./ndarray_factory.js";
export * from "./ndarray_helpers.js";
export * from "./ndwasmarray.js";
export { NDArray, DTYPE_MAP };

export function init(baseDir='.'){
    return ndwasm.NDWasm.init(baseDir);
}

// Make the NDArray class the default export for convenience.
export default NDArray;
