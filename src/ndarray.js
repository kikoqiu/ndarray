/**
 * Main entry point for the ndarray library.
 * This file imports all components, orchestrates the help system, and exports the public API.
 */

// 1. Import library components
import { NDArray, DTYPE_MAP } from './ndarray_core.js';
import { NDWasmArray } from './ndwasmarray.js';
import * as ndwasm from './ndwasm.js';

import { NDProb } from "./ndarray_prob.js";
import { NDWasmDecomp } from "./ndwasm_decomp.js";
import { NDWasmAnalysis } from "./ndwasm_analysis.js";
import { NDWasmBlas } from "./ndwasm_blas.js";
import { NDWasmSignal } from "./ndwasm_signal.js";
import { NDWasmImage } from "./ndwasm_image.js";
import { NDWasmOptimize } from './ndwasm_optimize.js';

import * as factory from "./ndarray_factory.js"

// 2. Import help system and documentation data
import docMap from './docs.json';
import { help } from './help.js';
export { help };

// 3. Register all documented objects with the help system
function registerAll() {
    // Define the top-level objects that are exported and documented
    const rootObjects = { 
        NDArray, 
        NDWasmArray, 
        ...ndwasm,
        NDProb,
        NDWasmDecomp,
        NDWasmAnalysis,
        NDWasmBlas,
        NDWasmSignal,
        NDWasmImage,
        NDWasmOptimize,
        ...factory,
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

export const random = NDProb;
export const image = NDWasmImage;
export const optimize = NDWasmOptimize;
export const decomp = NDWasmDecomp;
export const analysis = NDWasmAnalysis;
export const blas = NDWasmBlas;
export const signal = NDWasmSignal;

export { Jit } from "./ndarray_jit.js";

export * from "./ndwasm.js";
export * from "./ndarray_factory.js";
export { NDArray, DTYPE_MAP, NDWasmArray };

export function init(baseDir='.'){
    return ndwasm.NDWasm.init(baseDir);
}

// Make the NDArray class the default export for convenience.
export default NDArray;