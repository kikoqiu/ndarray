export var Jit = { debug: false };

/**
 * Generates a JIT-compiled unary kernel.
 * Output is always contiguous, so ptrOut simply increments by 1.
 * @private
 */
export function _createUnaryKernel(cacheKey, shape, sIn, fnOrStr) {
    let kernel = UNARY_KERNEL_CACHE.get(cacheKey);
    if (kernel) {
        return kernel;
    }
    
    const ndim = shape.length;
    const opBody = prepareUnaryOp(fnOrStr);

    let fnSource;

    // Handle 0-dimensional arrays (scalars)
    if (ndim === 0) {
        fnSource = `
            return function(dataIn, dataOut, offIn, offOut) {
                "use strict";
                dataOut[offOut] = ${opBody.replace('ptrIn', 'offIn')};
            }
        `;
    }else{

        // --- Build Nested Loops ---
        
        // 1. Innermost loop: Apply operation and step the input pointer.
        // Since the output of copy/map is a fresh contiguous array, 
        // ptrOut always increments by exactly 1.
        let code = `
            dataOut[ptrOut++] = ${opBody};
            ptrIn += ${sIn[ndim - 1]};
        `;
        code = `for (let i${ndim - 1} = 0; i${ndim - 1} < ${shape[ndim - 1]}; i${ndim - 1}++) { ${code} }`;

        // 2. Outer loops with "Gap" adjustment for the input pointer.
        for (let d = ndim - 2; d >= 0; d--) {
            // Adjustment: Move to next dimension start - distance covered by inner loops
            const adjIn = sIn[d] - (shape[d + 1] * sIn[d + 1]);
            code += ` ptrIn += ${adjIn};`;
            code = `for (let i${d} = 0; i${d} < ${shape[d]}; i${d}++) { ${code} }`;
        }

        // 3. Assemble final function
        fnSource = `
            return function(dataIn, dataOut, offIn, offOut) {
                "use strict";
                let ptrIn = offIn, ptrOut = offOut;
                ${code}
            };
        `;
    }
    if(Jit.debug) {
        console.log(cacheKey, "Unary Kernel Source:\n", fnSource);
    }
    kernel = new Function(fnSource)();
    UNARY_KERNEL_CACHE.set(cacheKey, kernel);
    return kernel;
}


/**
 * Generates a kernel with unrolled nested loops and cumulative pointer increments.
 * @private
 */
export function _createBinKernel(cacheKey, shape, sA, sB, sOut, opStr, isComparison) {
    let kernel = BIN_KERNEL_CACHE.get(cacheKey);
    if (kernel) {
        return kernel;
    }
    
    const ndim = shape.length;
    let body = extractOpBody(opStr);
    
    // Inline the data access into the operation body
    body = body.replace(/\ba|x\b/g, 'dataA[ptrA]').replace(/\bb|y\b/g, 'dataB[ptrB]');
    if (isComparison) body = `(${body}) ? 1 : 0`;

    let fnSource;
    // Handle 0-dimensional arrays (scalars)
    if (ndim === 0) {
        fnSource = `
            return function(dataA, dataB, dataOut, offA, offB, offOut) {
                "use strict";
                let ptrA = offA, ptrB = offB, ptrOut = offOut;
                dataOut[ptrOut] = ${body};
            }
        `;
    }else{

        // --- Build the Nested Loops String ---
        
        // 1. Innermost Loop logic: Apply operation and increment pointers by the last dimension's stride.
        let code = `
            dataOut[ptrOut] = ${body};
            ptrA += ${sA[ndim - 1]};
            ptrB += ${sB[ndim - 1]};
            ptrOut += ${sOut[ndim - 1]};
        `;
        code = `for (let i${ndim - 1} = 0; i${ndim - 1} < ${shape[ndim - 1]}; i${ndim - 1}++) { ${code} }`;

        // 2. Outer Loops logic: Wrap inner loops and apply pointer adjustments (gaps).
        // The adjustment ensures that after an inner loop finishes, the pointer moves 
        // to the correct start of the next index in the parent dimension.
        for (let d = ndim - 2; d >= 0; d--) {
            // Gap = Stride of current dimension - distance already moved by the inner dimension
            const adjA = sA[d] - (shape[d + 1] * sA[d + 1]);
            const adjB = sB[d] - (shape[d + 1] * sB[d + 1]);
            const adjOut = sOut[d] - (shape[d + 1] * sOut[d + 1]);
            
            code += ` ptrA += ${adjA}; ptrB += ${adjB}; ptrOut += ${adjOut};`;
            code = `for (let i${d} = 0; i${d} < ${shape[d]}; i${d}++) { ${code} }`;
        }

        // 3. Construct final function
        fnSource = `
            return function(dataA, dataB, dataOut, offA, offB, offOut) {
                "use strict";
                let ptrA = offA, ptrB = offB, ptrOut = offOut;
                ${code}
            };
        `;
    }

    if(Jit.debug) {
        console.log(cacheKey, "Bin Kernel Source:\n", fnSource);
    }
    kernel = new Function(fnSource)();
    BIN_KERNEL_CACHE.set(cacheKey, kernel);
    return kernel;
}



/**
     * Generates a JIT kernel using nested loops and pointer gaps.
     * @private
     */
export function _createReduceKernel(cacheKey, shape, strides, iterAxes, reduceAxes, reducer, finalFn) {
    let kernel = REDUCE_KERNEL_CACHE.get(cacheKey);
    if (kernel) {
        return kernel;
    }
    const nRed = reduceAxes.length;
    const nIter = iterAxes.length;

    const redExpr = prepareReduceExpr(reducer, 'reducer');
    const finalExpr = finalFn ? prepareReduceExpr(finalFn, 'finalizer') : 'acc';

    // --- 1. Build Reduction Space (Inner Loops) ---
    // The innermost operation: accumulate and step by the last reduction axis stride
    let redCode = `
        acc = ${redExpr};
        pIn += ${strides[reduceAxes[nRed - 1]]};
    `;
    
    // Nest reduction loops if multiple axes are being collapsed
    for (let d = nRed - 1; d >= 0; d--) {
        const ax = reduceAxes[d];
        // Gap to jump to the next index of the parent reduction dimension
        const gap = (d === 0) ? 0 : (strides[reduceAxes[d - 1]] - shape[ax] * strides[ax]);
        redCode = `
            for (let r${d} = 0; r${d} < ${shape[ax]}; r${d}++) {
                ${redCode}
            }
            pIn += ${gap};
        `;
    }

    // --- 2. Build Iteration Space (Outer Loops) ---
    // The "body" of the iteration produces one output element per reduction block
    let fullCode = `
        let acc = initVal;
        ${redCode}
        dataOut[pOut++] = ${finalExpr};
    `;

    if (nIter > 0) {
        // Calculate the total memory distance covered by the reduction block
        // Displacement = (shape of outermost reduction axis) * (stride of outermost reduction axis)
        const innerBlockDisplacement = shape[reduceAxes[0]] * strides[reduceAxes[0]];
        
        for (let d = nIter - 1; d >= 0; d--) {
            const ax = iterAxes[d];
            const stride = strides[ax];
            // The displacement to account for is either the reduction block (if innermost iter)
            // or the inner iteration loop displacement.
            const movedByChild = (d === nIter - 1) ? innerBlockDisplacement : (shape[iterAxes[d+1]] * strides[iterAxes[d+1]]);
            const gap = stride - movedByChild;

            // CRITICAL: pIn += gap must be INSIDE the loop to adjust ptr for next i{d}
            fullCode = `
                for (let i${d} = 0; i${d} < ${shape[ax]}; i${d}++) {
                    ${fullCode}
                    pIn += ${gap}; 
                }
            `;
        }
    }

    // --- 3. Assemble Final Kernel ---
    const fnSource = `
        return function(dataIn, dataOut, offIn, initVal, count) {
            "use strict";
            let pIn = offIn;
            let pOut = 0;
            ${fullCode}
        };
    `;

    if(Jit.debug) {
        console.log(cacheKey, "reduce Kernel Source:\n", fnSource);
    }
    kernel = new Function(fnSource)();
    REDUCE_KERNEL_CACHE.set(cacheKey, kernel);
    return kernel;
}

/**
 * Global cache for set JIT kernels to avoid redundant compilations.
 * @private
 */
const SET_KERNEL_CACHE = new Map();

/**
 * Generates a JIT-compiled kernel for the set operation.
 * Optimized for V8 by using incremental pointer arithmetic and static nesting.
 * 
 * @param {string} cacheKey - Unique key for the specific array structure.
 * @param {number} ndim - Number of dimensions of the target array.
 * @param {Array} targetShape - The logical shape of the selection.
 * @param {Int32Array} tStrides - Strides of the target NDArray.
 * @param {Int32Array} sStrides - Pre-computed broadcasting strides for the source.
 * @param {Uint8Array} hasPSet - Flags indicating if a dimension uses fancy indexing or scalar.
 * @param {Array<boolean>} isDimReduced - Flags indicating if a dimension is collapsed by scalar indexing.
 * @private
 */
export function _createSetKernel(cacheKey, ndim, targetShape, tStrides, sStrides, hasPSet, isDimReduced) {
    let kernel = SET_KERNEL_CACHE.get(cacheKey);
    if (kernel) return kernel;

    let targetDimIdx = 0;

    /**
     * Recursively builds the nested loop string.
     * @param {number} d - Current dimension index.
     * @returns {string} - The generated code block for this dimension.
     */
    function buildLevel(d) {
        if (d === ndim) {
            // Innermost level: Perform the actual data assignment
            return `dataT[pT${d}] = dataS[pS${d}];`;
        }

        const tStride = tStrides[d];
        const sStride = sStrides[d];
        const pT_prev = `pT${d}`;
        const pS_prev = `pS${d}`;
        const pT_next = `pT${d + 1}`;
        const pS_next = `pS${d + 1}`;

        if (isDimReduced[d]) {
            // SCALAR INDEX: Offset the pointer once and move to the next dimension level.
            return `
            const ${pT_next} = ${pT_prev} + ps[${d}][0] * ${tStride};
            const ${pS_next} = ${pS_prev};
            ${buildLevel(d + 1)}`;
        } else {
            const len = targetShape[targetDimIdx++];
            if (!hasPSet[d]) {
                // SLICE INDEX (:): Use incremental pointer arithmetic in the loop header for V8 speed.
                return `
                for (let i${d} = 0, ${pT_next} = ${pT_prev}, ${pS_next} = ${pS_prev}; i${d} < ${len}; i${d}++, ${pT_next} += ${tStride}, ${pS_next} += ${sStride}) {
                    ${buildLevel(d + 1)}
                }`;
            } else {
                // FANCY INDEXING: Map the pointer using the pick-set array at each iteration.
                return `
                for (let i${d} = 0; i${d} < ${len}; i${d}++) {
                    const ${pT_next} = ${pT_prev} + ps[${d}][i${d}] * ${tStride};
                    const ${pS_next} = ${pS_prev} + i${d} * ${sStride};
                    ${buildLevel(d + 1)}
                }`;
            }
        }
    }

    const fnSource = `
        return function(dataT, dataS, offT, offS, ps) {
            "use strict";
            const pT0 = offT;
            const pS0 = offS;
            ${buildLevel(0)}
        };
    `;

    if(Jit.debug) {
        console.log(cacheKey, "Set Kernel Source:\n", fnSource);
    }

    kernel = new Function(fnSource)();
    SET_KERNEL_CACHE.set(cacheKey, kernel);
    return kernel;
}

/**
 * @private
 */
const BIN_KERNEL_CACHE = new Map();

/**
 * Extracts the expression body from a function.
 * Supports both: (a, b) => a + b  AND  function(a, b) { return a + b; }
 * @private
 */
function extractOpBody(fnStr) {
    // Try arrow function first
    let match = fnStr.match(/=>\s*([\s\S]+)/);
    if (match) return match[1].trim().replace(/;$/, '');
    // Try standard function return
    match = fnStr.match(/\{[\s\S]*return\s+([\s\S]+?);?\s*\}/);
    if (match) return match[1].trim();
    return fnStr; // Fallback
}




/**
 * Global cache for unary JIT kernels (copy/map).
 * @private
 */
const UNARY_KERNEL_CACHE = new Map();

/**
 * Extracts function body or processes template strings into executable JS code.
 * @private
 */
function prepareUnaryOp(fnOrStr) {
    if (typeof fnOrStr === 'string') {
        // If it's a template like 'Math.sin(${val})', replace with pointer access
        if (fnOrStr.includes('${val}')) {
            return fnOrStr.replace(/\$\{val\}/g, 'dataIn[ptrIn]');
        }
        return fnOrStr.replace(/\bval\b/g, 'dataIn[ptrIn]');
    }
    // If it's a function, stringify and replace the parameter with pointer access
    const fnStr = fnOrStr.toString();
    const body = extractOpBody(fnStr); // Reuses the helper
    // Matches common parameter names like x, val, or item
    return body.replace(/\b(x|a|val|item)\b/g, 'dataIn[ptrIn]');
}







/**
 * @private
 */
const REDUCE_KERNEL_CACHE = new Map();

/**
 * Normalizes the reducer and finalizer into inline expressions.
 * @private
 */
function prepareReduceExpr(fnOrStr, type = 'reducer') {
    const s = fnOrStr.toString();
    const body = extractOpBody(s);
    if (type === 'reducer') {
        // Map arguments: (acc, val) -> acc + dataIn[pIn]
        return body.replace(/\bacc|a\b/g, 'acc').replace(/\b(b|val|v|item)\b/g, 'dataIn[pIn]');
    }
    // Map arguments: (acc, count) -> acc / count
    return body.replace(/\bacc|a\b/g, 'acc').replace(/\b(n|count|len)\b/g, 'count');
}





/**
 * Global cache for pick JIT kernels.
 * @private
 */
const PICK_KERNEL_CACHE = new Map();

/**
 * Generates a JIT-compiled kernel for the pick operation.
 * Optimized for V8 with incremental pointer arithmetic and contiguous output writes.
 * 
 * @param {string} cacheKey - Unique key for the specific array structure.
 * @param {number} ndim - Number of dimensions of the source array.
 * @param {Int32Array} sStrides - Strides of the source NDArray.
 * @param {Uint8Array} isFullSlice - Flags indicating if a dimension is a full ":" slice.
 * @param {Array<boolean>} isDimReduced - Flags indicating if a dimension is collapsed by scalar indexing.
 * @param {Int32Array} odometerShape - The lengths of the pick-sets for each dimension.
 * @private
 */
export function _createPickKernel(cacheKey, ndim, sStrides, isFullSlice, isDimReduced, odometerShape) {
    let kernel = PICK_KERNEL_CACHE.get(cacheKey);
    if (kernel) return kernel;

    /**
     * Recursively builds the nested loop string.
     * @param {number} d - Current source dimension index.
     */
    function buildLevel(d) {
        if (d === ndim) {
            // Innermost level: Read from calculated source pointer and write to contiguous output
            return `dataOut[pOut++] = dataIn[pIn${d}];`;
        }

        const sStride = sStrides[d];
        const pIn_prev = `pIn${d}`;
        const pIn_next = `pIn${d + 1}`;
        const len = odometerShape[d];

        if (isDimReduced[d]) {
            // SCALAR INDEX: Offset the source pointer once and move to next dimension.
            return `
            const ${pIn_next} = ${pIn_prev} + ps[${d}][0] * ${sStride};
            ${buildLevel(d + 1)}`;
        } else if (isFullSlice[d]) {
            // FULL SLICE (:): Optimization - use incremental pointer arithmetic (no pick-set lookup).
            return `
            for (let i${d} = 0, ${pIn_next} = ${pIn_prev}; i${d} < ${len}; i${d}++, ${pIn_next} += ${sStride}) {
                ${buildLevel(d + 1)}
            }`;
        } else {
            // FANCY INDEXING: Lookup source index from the pick-set (ps) array.
            return `
            for (let i${d} = 0; i${d} < ${len}; i${d}++) {
                const ${pIn_next} = ${pIn_prev} + ps[${d}][i${d}] * ${sStride};
                ${buildLevel(d + 1)}
            }`;
        }
    }

    const fnSource = `
        return function(dataIn, dataOut, offIn, ps) {
            "use strict";
            let pOut = 0;
            const pIn0 = offIn;
            ${buildLevel(0)}
        };
    `;

    if(Jit.debug) { 
        console.log(cacheKey, "Pick Kernel Source:\n", fnSource);
    }

    kernel = new Function(fnSource)();
    PICK_KERNEL_CACHE.set(cacheKey, kernel);
    return kernel;
}