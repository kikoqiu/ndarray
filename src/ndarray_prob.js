import { NDArray, DTYPE_MAP } from './ndarray_core.js';

/**
 * NDArray Probability & Random Module
 * Uses Cryptographically Strong Pseudo-Random Number Generator (CSPRNG).
 * All functions return a new NDArray instance.
 * @namespace NDProb
 */
export const NDProb = {
    /**
     * Internal helper to get a Float64 between [0, 1) using crypto.
     * Generates high-quality uniform random floats.
     * @private
     */
    _cryptoUniform01(size) {
        const uint32 = new Uint32Array(size);
        // Browser & Node.js compatible crypto access
        const cryptoObj = globalThis.crypto;
        
        // getRandomValues has a limit (usually 65536 bytes), so we chunk if necessary
        const quota = 65536 / 4;
        for (let i = 0; i < size; i += quota) {
            const chunk = uint32.subarray(i, Math.min(i + quota, size));
            cryptoObj.getRandomValues(chunk);
        }

        const floats = new Float64Array(size);
        for (let i = 0; i < size; i++) {
            // Divide by 2^32 to get [0, 1)
            floats[i] = uint32[i] / 4294967296.0;
        }
        return floats;
    },

    /**
     * Uniform distribution over [low, high).
     * @memberof NDProb
     * @function
     * @param {Array} shape - Dimensions of the output array.
     * @param {number} [low=0] - Lower boundary.
     * @param {number} [high=1] - Upper boundary.
     * @param {string} [dtype='float64'] - Data type.
     * @returns {NDArray}
     */
    random(shape, low = 0, high = 1, dtype = 'float64') {
        const size = shape.reduce((a, b) => a * b, 1);
        const data = this._cryptoUniform01(size);
        const range = high - low;

        if (range !== 1 || low !== 0) {
            for (let i = 0; i < size; i++) data[i] = data[i] * range + low;
        }

        // Cast to specified dtype if necessary
        const finalData = dtype === 'float64' ? data : this._cast(data, dtype);
        return new NDArray(finalData, { shape, dtype });
    },

    /**
     * Normal (Gaussian) distribution using Box-Muller transform.
     * @param {Array} shape - Dimensions of the output array.
     * @param {number} [mean=0] - Mean of the distribution.
     * @param {number} [std=1] - Standard deviation.
     * @param {string} [dtype='float64'] - Data type.
     * @memberof NDProb
     * @returns {NDArray}
     */
    normal(shape, mean = 0, std = 1, dtype = 'float64') {
        const size = shape.reduce((a, b) => a * b, 1);
        // Box-Muller requires two uniform variables to produce two normal variables
        const u1 = this._cryptoUniform01(Math.ceil(size / 2) * 2);
        const u2 = this._cryptoUniform01(u1.length);
        const data = new Float64Array(size);

        for (let i = 0; i < size; i += 2) {
            const r = Math.sqrt(-2.0 * Math.log(Math.max(u1[i], 1e-10)));
            const theta = 2.0 * Math.PI * u2[i];
            
            data[i] = r * Math.cos(theta) * std + mean;
            if (i + 1 < size) {
                data[i + 1] = r * Math.sin(theta) * std + mean;
            }
        }

        const finalData = dtype === 'float64' ? data : this._cast(data, dtype);
        return new NDArray(finalData, { shape, dtype });
    },

    /**
     * Bernoulli distribution (0 or 1 with probability p).
     * @param {Array} shape
     * @param {number} [p=0.5] - Probability of success (1).
     * @param {string} [dtype='int32']
     * @memberof NDProb
     * @returns {NDArray}
     */
    bernoulli(shape, p = 0.5, dtype = 'int32') {
        const size = shape.reduce((a, b) => a * b, 1);
        const u = this._cryptoUniform01(size);
        const data = new DTYPE_MAP[dtype](size);

        for (let i = 0; i < size; i++) {
            data[i] = u[i] < p ? 1 : 0;
        }
        return new NDArray(data, { shape, dtype });
    },

    /**
     * Exponential distribution: f(x; λ) = λe^(-λx).
     * Inverse transform sampling: x = -ln(1-u) / λ.
     * @param {Array} shape
     * @param {number} [lambda=1.0] - Rate parameter.
     * @memberof NDProb
     * @returns {NDArray}
     */
    exponential(shape, lambda = 1.0, dtype = 'float64') {
        const size = shape.reduce((a, b) => a * b, 1);
        const u = this._cryptoUniform01(size);
        const data = new Float64Array(size);

        for (let i = 0; i < size; i++) {
            // Using Math.max to avoid log(0)
            data[i] = -Math.log(Math.max(1.0 - u[i], 1e-10)) / lambda;
        }

        const finalData = dtype === 'float64' ? data : this._cast(data, dtype);
        return new NDArray(finalData, { shape, dtype });
    },

    /**
     * Poisson distribution using Knuth's algorithm.
     * Note: For very large lambda, this becomes slow; but for most use cases it's fine.
     * @param {Array} shape
     * @param {number} [lambda=1.0] - Mean of the distribution.
     * @memberof NDProb
     * @returns {NDArray}
     */
    poisson(shape, lambda = 1.0, dtype = 'int32') {
        const size = shape.reduce((a, b) => a * b, 1);
        const data = new DTYPE_MAP[dtype](size);
        const L = Math.exp(-lambda);

        for (let i = 0; i < size; i++) {
            let k = 0;
            let p = 1.0;
            do {
                k++;
                // We fetch one by one here as Poisson count per element is variable
                p *= this._cryptoUniform01(1)[0];
            } while (p > L);
            data[i] = k - 1;
        }
        return new NDArray(data, { shape, dtype });
    },

    /** 
     * @private
     * @internal
     */
    _cast(data, dtype) {
        const out = new DTYPE_MAP[dtype](data.length);
        out.set(data);
        return out;
    }
};

