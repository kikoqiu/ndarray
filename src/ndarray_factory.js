// --- factory functions ---
import { NDArray, DTYPE_MAP } from './ndarray_core.js';

/**
 * Creates an NDArray from a regular array or TypedArray.
 * @param {Array|TypedArray} source - The source data.
 * @param {string} [dtype='float64'] - The desired data type.
 * @return {NDArray}
 */
export function array(source, dtype = 'float64') {
    const Ctor = DTYPE_MAP[dtype] || Float64Array;
    const isTyped = (v) => ArrayBuffer.isView(v) && !(v instanceof DataView);

    // 1. Shape Inference & Strict Validation
    const getShape = (v) => {
        if (isTyped(v)) return [v.length];
        if (!Array.isArray(v)) return []; // Scalar
        if (v.length === 0) throw new Error("Input array cannot be empty.");

        const sub = getShape(v[0]);
        const subStr = sub.join(',');
        
        for (let i = 1; i < v.length; i++) {
            if (getShape(v[i]).join(',') !== subStr) {
                throw new Error(`Jagged array detected at index ${i}.`);
            }
        }
        return [v.length, ...sub];
    };

    const shape = getShape(source);
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Ctor(size);

    // 2. Fill Data
    // Special handling for scalar (0-dim) to avoid logic complexity in loop
    if (shape.length === 0) {
        data[0] = source;
    } else {
        let offset = 0;
        const lastDim = shape.length - 1; // The index of the innermost dimension

        const fill = (v, dim) => {
            if (dim === lastDim) {
                // Optimization: We are at the bottom level.
                // v is guaranteed to be an Array-like (JS Array or TypedArray)
                // data.set is the fastest way to copy a block of memory.
                data.set(v, offset);
                offset += v.length;
            } else {
                // We are not at the bottom. v is guaranteed to be an Array.
                // Recurse deeper.
                for (let i = 0; i < v.length; i++) {
                    fill(v[i], dim + 1);
                }
            }
        };
        fill(source, 0);
    }

    return new NDArray(data, { shape, dtype });
}

/**
 * Creates an NDArray from a regular array or TypedArray.
 * @param {Array|TypedArray} source - The source data.
 * @return {NDArray}
 */
export function float64(source){
    return array(source,"float64");
}
/**
 * Creates an NDArray from a regular array or TypedArray.
 * @param {Array|TypedArray} source - The source data.
 * @return {NDArray}
 */
export function float32(source){
    return array(source,"float32");
}
/**
 * Creates an NDArray from a regular array or TypedArray.
 * @param {Array|TypedArray} source - The source data.
 * @return {NDArray}
 */
export function uint32(source){
    return array(source,"uint32");
}
/**
 * Creates an NDArray from a regular array or TypedArray.
 * @param {Array|TypedArray} source - The source data.
 * @return {NDArray}
 */
export function int32(source){
    return array(source,"int32");
}
/**
 * Creates an NDArray from a regular array or TypedArray.
 * @param {Array|TypedArray} source - The source data.
 * @return {NDArray}
 */
export function int16(source){
    return array(source,"int16");
}
/**
 * Creates an NDArray from a regular array or TypedArray.
 * @param {Array|TypedArray} source - The source data.
 * @return {NDArray}
 */
export function uint16(source){
    return array(source,"uint16");
}
/**
 * Creates an NDArray from a regular array or TypedArray.
 * @param {Array|TypedArray} source - The source data.
 * @return {NDArray}
 */
export function int8(source){
    return array(source,"int8");
}
/**
 * Creates an NDArray from a regular array or TypedArray.
 * @param {Array|TypedArray} source - The source data.
 * @return {NDArray}
 */
export function uint8(source){
    return array(source,"uint8");
}
/**
 * Creates an NDArray from a regular array or TypedArray.
 * @param {Array|TypedArray} source - The source data.
 * @return {NDArray}
 */
export function uint8c(source){
    return array(source,"uint8c");
}

/**
 * Creates a new NDArray of the given shape, filled with zeros.
 * @param {Array<number>} shape - The shape of the new array.
 * @param {string} [dtype='float64'] - The data type of the new array.
 * @returns {NDArray}
 */
export function zeros(shape, dtype = 'float64') {
    const size = shape.reduce((a, b) => a * b, 1);
    const Constructor = DTYPE_MAP[dtype];
    return new NDArray(new Constructor(size), { shape, dtype });
}

/**
 * Creates a new NDArray of the given shape, filled with ones.
 * @param {Array<number>} shape - The shape of the new array.
 * @param {string} [dtype='float64'] - The data type of the new array.
 * @returns {NDArray}
 */
export function ones(shape, dtype = 'float64') {
    const arr = zeros(shape, dtype);
    arr.data.fill(1);
    return arr;
}

/**
 * Creates a new NDArray of the given shape, filled with a specified value.
 * @param {Array<number>} shape - The shape of the new array.
 * @param {*} value - The value to fill the array with.
 * @param {string} [dtype='float64'] - The data type of the new array.
 * @returns {NDArray}
 */
export function full(shape, value, dtype = 'float64') {
    const arr = zeros(shape, dtype);
    arr.data.fill(value);
    return arr;
}

/**
 * Like Python's arange: [start, stop)
 * @param {number} start
 * @param {number | null} stop
 * @param {number | null} step
 * @param {string} dtype
 */
export function arange(start, stop = null, step = 1, dtype = 'float64') {
    if (stop === null) {
        stop = start;
        start = 0;
    }
    const size = Math.ceil((stop - start) / step);
    const Constructor = DTYPE_MAP[dtype];
    const data = new Constructor(size);
    for (let i = 0; i < size; i++) {
        data[i] = start + i * step;
    }
    return new NDArray(data, { shape: [size], dtype });
}

/**
 * Linearly spaced points.
 * @param {number} start
 * @param {number | null} stop
 * @param {number} num
 * @param {string} dtype
 * @returns {NDArray}
 */
export function linspace(start, stop, num = 50, dtype = 'float64') {
    const step = (stop - start) / (num - 1);
    const Constructor = DTYPE_MAP[dtype];
    const data = new Constructor(num);
    for (let i = 0; i < num; i++) {
        data[i] = start + i * step;
    }
    return new NDArray(data, { shape: [num], dtype });
}


/**
 * where(condition, x, y)
 * Returns a new array with elements chosen from x or y depending on condition.
 * Supports NumPy-style broadcasting across all three arguments, including 0-sized dimensions.
 * 
 * @param {NDArray|Array|number} condition - Where True, yield x, otherwise yield y.
 * @param {NDArray|Array|number} x - Values from which to choose if condition is True.
 * @param {NDArray|Array|number} y - Values from which to choose if condition is False.
 * @returns {NDArray} A new contiguous NDArray.
 */
export function where(condition, x, y) {
    // 1. Standardize inputs to NDArray
    const toNDArray = (val, defaultDtype = 'float64') => {
        return (val instanceof NDArray) ? val : array(val, defaultDtype);
    };

    const cond = toNDArray(condition, 'int8');

    const dtypes = [(x instanceof NDArray)?x.dtype:null , (y instanceof NDArray)?y.dtype:null];
    let dtype = null;
    for(let i of dtypes){
        if(dtype === null){
            dtype = i;
        }else if(dtype !== i && i !== null){
            throw new Error(`dtype is different ${dtype} ${i}`);
        }
    }
    if(dtype===null){
        dtype="float64";
    }
    const xArr = toNDArray(x, dtype);
    const yArr = toNDArray(y, dtype);

    // 2. Determine output shape using strict NumPy broadcasting rules
    const outNdim = Math.max(cond.ndim, xArr.ndim, yArr.ndim);
    const outShape = new Int32Array(outNdim);

    for (let i = 1; i <= outNdim; i++) {
        // Dimension sizes from the right (defaulting to 1 for missing dimensions)
        const dC = (cond.ndim - i >= 0) ? cond.shape[cond.ndim - i] : 1;
        const dX = (xArr.ndim - i >= 0) ? xArr.shape[xArr.ndim - i] : 1;
        const dY = (yArr.ndim - i >= 0) ? yArr.shape[yArr.ndim - i] : 1;

        // Broadcasting logic: 
        // A dimension is compatible if all values are equal, or if some are 1.
        let maxDim = 1;
        const dims = [dC, dX, dY];
        for (let j = 0; j < 3; j++) {
            const d = dims[j];
            if (d !== 1) {
                if (maxDim !== 1 && maxDim !== d) {
                    throw new Error(`Incompatible shapes for where: condition [${cond.shape}], x [${xArr.shape}], y [${yArr.shape}]`);
                }
                maxDim = d;
            }
        }
        outShape[outNdim - i] = maxDim;
    }

    // 3. Prepare traversal metadata
    // reduce returns 1 for outNdim == 0 (scalars) and 0 if any dim is 0.
    const outSize = outShape.reduce((a, b) => a * b, 1);
    
    const sC = new Int32Array(outNdim);
    const sX = new Int32Array(outNdim);
    const sY = new Int32Array(outNdim);
    const wrapC = new Float64Array(outNdim);
    const wrapX = new Float64Array(outNdim);
    const wrapY = new Float64Array(outNdim);

    for (let i = 0; i < outNdim; i++) {
        const dim = outShape[i];
        const axisFromRight = outNdim - i;

        const getStride = (arr, axisRight) => {
            const axis = arr.ndim - axisRight;
            // Broadcast (stride 0) if dimension is missing (axis < 0) or size is 1
            if (axis < 0 || arr.shape[axis] === 1) return 0;
            return arr.strides[axis];
        };

        sC[i] = getStride(cond, axisFromRight);
        sX[i] = getStride(xArr, axisFromRight);
        sY[i] = getStride(yArr, axisFromRight);

        // wrapBack is the distance to jump from the last element back to the first in a dimension
        wrapC[i] = (dim - 1) * sC[i];
        wrapX[i] = (dim - 1) * sX[i];
        wrapY[i] = (dim - 1) * sY[i];
    }

    // 4. Execution Engine
    const Constructor = DTYPE_MAP[xArr.dtype] || Float64Array;
    const newData = new Constructor(outSize);
    const currentIdx = new Int32Array(outNdim);
    
    let pC = cond.offset;
    let pX = xArr.offset;
    let pY = yArr.offset;

    const dataC = cond.data;
    const dataX = xArr.data;
    const dataY = yArr.data;

    // Loop runs outSize times. If outSize is 0, it doesn't run at all.
    // If outSize is 1 (0D scalar), it runs exactly once.
    for (let i = 0; i < outSize; i++) {
        // Core ternary selection
        newData[i] = dataC[pC] ? dataX[pX] : dataY[pY];

        // Odometer increment with manual pointer updates
        for (let d = outNdim - 1; d >= 0; d--) {
            if (++currentIdx[d] < outShape[d]) {
                pC += sC[d];
                pX += sX[d];
                pY += sY[d];
                break;
            } else {
                currentIdx[d] = 0;
                pC -= wrapC[d];
                pX -= wrapX[d];
                pY -= wrapY[d];
            }
        }
    }

    return new NDArray(newData, { 
        shape: outShape, 
        dtype: xArr.dtype 
    });
};



/**
 * Joins a sequence of arrays along an existing axis.
 * @param {Array<NDArray>} arrays The arrays must have the same shape, except in the dimension corresponding to `axis`.
 * @param {number} [axis=0] The axis along which the arrays will be joined.
 * @returns {NDArray} A new NDArray.
 */
export function concat(arrays, axis = 0) {
    if (!arrays || arrays.length === 0) {
        throw new Error('Array list cannot be empty.');
    }

    if (arrays.length === 1) {
        return arrays[0].copy(); // .copy() is available in this file.
    }

    const firstArr = arrays[0];
    const { dtype, ndim } = firstArr;

    // --- Validation and Metadata ---
    if (ndim === 0) {
        if (!arrays.every(arr => arr.ndim === 0)) {
            throw new Error('All arrays must be scalars to concatenate with a scalar.');
        }
        const newData = new (DTYPE_MAP[dtype])(arrays.length);
        for (let i = 0; i < arrays.length; i++) {
            newData[i] = arrays[i].data[arrays[i].offset];
        }
        return new NDArray(newData, { shape: [arrays.length], dtype });
    }
    
    const finalAxis = axis < 0 ? ndim + axis : axis;
    if (finalAxis < 0 || finalAxis >= ndim) {
        throw new Error(`Axis ${axis} is out of bounds for array of dimension ${ndim}.`);
    }

    let newShape = [...firstArr.shape];
    let concatDimSize = 0;

    for (const arr of arrays) {
        if (arr.ndim !== ndim) throw new Error('All arrays must have same number of dimensions.');
        if (arr.dtype !== dtype) throw new Error('All arrays must have same dtype.');
        for (let j = 0; j < ndim; j++) {
            if (j !== finalAxis && arr.shape[j] !== newShape[j]) {
                throw new Error(`Dimension mismatch on axis ${j}: expected ${newShape[j]} but got ${arr.shape[j]}.`);
            }
        }
        concatDimSize += arr.shape[finalAxis];
    }
    newShape[finalAxis] = concatDimSize;
    // --- End Validation ---

    const result = zeros(newShape, dtype);

    let destAxisOffset = 0;
    for (const arr of arrays) {
        if (arr.size === 0) continue;

        const dimSize = arr.shape[finalAxis];
        
        const sliceSpec = Array(ndim).fill(null);
        sliceSpec[finalAxis] = [destAxisOffset, destAxisOffset + dimSize];

        const view = result.slice(...sliceSpec);
        
        view.set(arr);
        
        destAxisOffset += dimSize;
    }

    return result;
}


/**
 * Joins a sequence of arrays along a new axis.
 * The `stack` function creates a new dimension, whereas `concat` joins along an existing one.
 * All input arrays must have the same shape and dtype.
 *
 * @param {Array<NDArray>} arrays - The list of arrays to stack.
 * @param {number} [axis=0] - The axis in the result array along which the input arrays are stacked.
 * @returns {NDArray} A new NDArray.
 */
export function stack(arrays, axis = 0) {
    if (!arrays || arrays.length === 0) {
        throw new Error('Array list cannot be empty.');
    }

    const firstArr = arrays[0];
    const { shape: oldShape, ndim: oldNdim, dtype } = firstArr;

    // The result of stack will have one more dimension than the input arrays
    const targetNdim = oldNdim + 1;
    
    // Normalize negative axis (e.g., -1 refers to the last possible axis)
    let finalAxis = axis < 0 ? targetNdim + axis : axis;
    if (finalAxis < 0 || finalAxis >= targetNdim) {
        throw new Error(`Axis ${axis} is out of bounds for stack into ${targetNdim} dimensions.`);
    }

    // Step 1: Validate shapes and expand dimensions
    const expandedArrays = arrays.map((arr, index) => {
        // Ensure dimensionality matches
        if (arr.ndim !== oldNdim) {
            throw new Error(`Array at index ${index} has ${arr.ndim} dimensions, expected ${oldNdim}.`);
        }
        
        // Ensure dtypes match
        if (arr.dtype !== dtype) {
            throw new Error(`Dtype mismatch at index ${index}. Expected ${dtype}.`);
        }

        // Ensure shapes match exactly
        for (let i = 0; i < oldNdim; i++) {
            if (arr.shape[i] !== oldShape[i]) {
                throw new Error(
                    `Shape mismatch at index ${index}: expected [${oldShape}] but got [${arr.shape}].`
                );
            }
        }

        /**
         * Step 2: Reshape the array to add a singleton dimension at the target axis.
         * Example: If shape is [3, 4] and axis is 1, new shape becomes [3, 1, 4].
         */
        const newShape = [...arr.shape];
        newShape.splice(finalAxis, 0, 1);

        // return a new view with the expanded shape
        return arr.reshape(...newShape);
    });

    /**
     * Step 3: Concatenate the expanded arrays along the newly created axis.
     * Since every array now has a size of 1 at 'finalAxis', 
     * concat will join them to a size of 'arrays.length' at that axis.
     */
    return concat(expandedArrays, finalAxis);
};










/**
 * Creates a 2D identity matrix.
 * @param {number} n - Number of rows.
 * @param {number} [m] - Number of columns. Defaults to n if not provided.
 * @param {string} [dtype='float64'] - Data type of the array.
 * @returns {NDArray} A 2D NDArray with ones on the main diagonal.
 */
export function eye(n, m, dtype = 'float64') {
    const rows = n;
    const cols = m === undefined ? n : m;
    const shape = [rows, cols];
    
    // 1. Initialize an all-zero NDArray.
    // This handles data allocation and dtype mapping via the existing zeros method.
    const res = zeros(shape, dtype);
    
    const offset = 0;
    // 2. Fill the main diagonal with 1.
    // For a contiguous 2D array, the element at (i, i) is at index: i * cols + i.
    const minDim = Math.min(rows, cols);
    for (let i = 0; i < minDim; i++) {
        res.data[offset + i * cols + i] = 1;
    }
    
    return res;
}