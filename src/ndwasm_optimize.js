import { NDArray } from './ndarray_core.js';
import { NDWasm, fromWasm } from './ndwasm.js';
import { array, concat } from './ndarray_factory.js';

/**
 * Maps status codes from gonum/optimize to human-readable messages.
 * @private
 * @constant
 */
const OPTIMIZE_STATUS_MAP = [
"NotTerminated",
"Success",
"FunctionThreshold",
"FunctionConvergence",
"GradientThreshold",
"StepConvergence",
"FunctionNegativeInfinity",
"MethodConverge",
"Failure,optimize: termination ended in failure",
"IterationLimit,optimize: maximum number of major iterations reached",
"RuntimeLimit,optimize: maximum runtime reached",
"FunctionEvaluationLimit,optimize: maximum number of function evaluations reached",
"GradientEvaluationLimit,optimize: maximum number of gradient evaluations reached",
"HessianEvaluationLimit,optimize: maximum number of Hessian evaluations reached",
];


/**
 * Namespace for Optimization functions using Go WASM.
 * @namespace NDWasmOptimize
 */
export const NDWasmOptimize = {
    /**
     * Provides Optimization capabilities by wrapping Go WASM functions.
     * minimize cᵀ * x
     * s.t      G * x <= h
     * 	        A * x = b
     *          lower <= x <= upper
     * @param {NDArray} c - Coefficient vector for the objective function (1D NDArray of float64).
     * @param {NDArray | null} G - Coefficient matrix for inequality constraints (2D NDArray of float64).
     * @param {NDArray | null} h - Right-hand side vector for inequality constraints (1D NDArray of float64).
     * @param {NDArray | null} A - Coefficient matrix for equality constraints (2D NDArray of float64).
     * @param {NDArray | null} b - Right-hand side vector for equality constraints (1D NDArray of float64).
     * @param {Array} bounds - Optional variable bounds as an array of [lower, upper] pairs. Use null for unbounded. [0, null] for all for default.
     * @returns {{x: NDArray, fun: number, status: number, message: string}} - The optimization result.
     * @throws {Error} If WASM runtime is not loaded or inputs are invalid.
     */
    linprog(c, G, h, A, b, bounds) {
        if (!NDWasm.runtime?.isLoaded) throw new Error("WasmRuntime not loaded.");
        
        // 1. Validate Dimensions
        if (c.ndim !== 1) throw new Error("c must be 1D.");
        if ((G && G.ndim !== 2) || (h && h.ndim !== 1)) throw new Error("G must be 2D and h must be 1D.");
        if ((A && A.ndim !== 2) || (b && b.ndim !== 1)) throw new Error("A must be 2D and b must be 1D.");
        
        if (G && h && G.shape[0] !== h.shape[0]) throw new Error(`Dimension mismatch: G rows (${G.shape[0]}) must match h length (${h.shape[0]}).`);
        if (G && G.shape[1] !== c.shape[0]) throw new Error(`Dimension mismatch: G cols (${G.shape[1]}) must match c length (${c.shape[0]}).`);
        if (A && b && A.shape[0] !== b.shape[0]) throw new Error(`Dimension mismatch: A rows (${A.shape[0]}) must match b length (${b.shape[0]}).`);
        if (A && A.shape[1] !== c.shape[0]) throw new Error(`Dimension mismatch: A cols (${A.shape[1]}) must match c length (${c.shape[0]}).`);

        // Check for lack of constraints is no longer strictly necessary if bounds are handled natively, 
        // but keeping it depends on specific solver requirements. The Go solver seems to handle no constraints fine.

        let cWasm, GWasm, hWasm, AWasm, bWasm, boundsWasm, xResultWasm, objValWasm, statusWasm;

        try {
            // 2. Prepare Data
            const nVars = c.shape[0];
            cWasm = c.toWasm(NDWasm.runtime);
            GWasm = G ? G.toWasm(NDWasm.runtime) : null;
            hWasm = h ? h.toWasm(NDWasm.runtime) : null;
            AWasm = A ? A.toWasm(NDWasm.runtime) : null;
            bWasm = b ? b.toWasm(NDWasm.runtime) : null;

            // 3. Process Bounds (Interleaved format: [Low0, High0, Low1, High1...])
            const boundsData = Array(nVars * 2);
            for (let i = 0; i < nVars; i++) {
                let [lower, upper] = (bounds && bounds[i])? bounds[i] : [0, Infinity];// Default Lower: 0, Default Upper: +Inf

                // Handle Lower
                if (lower === null) lower = -Infinity; // Treat explicit null in pair as -Inf
                // Handle Upper
                if (upper === null) upper = Infinity; // Treat explicit null in pair as +Inf

                boundsData[i * 2] = lower;
                boundsData[i * 2 + 1] = upper;
            }

            // Create WASM buffer for bounds and populate it
            boundsWasm = array(boundsData).toWasm(NDWasm.runtime);

            // 4. Output Buffers
            xResultWasm = NDWasm.runtime.createBuffer(c.size, c.dtype);
            objValWasm = NDWasm.runtime.createBuffer(1, 'float64');
            statusWasm = NDWasm.runtime.createBuffer(1, 'int32');

            // 5. Call WASM Function
            // Signature: cPtr, cLen, gPtr, gRows, hPtr, aPtr, aRows, bPtr, boundsPtr, xPtr, objPtr, statusPtr
            NDWasm.runtime.exports.LinProg_F64(
                cWasm.ptr, cWasm.size,
                GWasm?.ptr ?? 0, G ? G.shape[0] : 0,
                hWasm?.ptr ?? 0,
                AWasm?.ptr ?? 0, A ? A.shape[0] : 0,
                bWasm?.ptr ?? 0,
                boundsWasm.ptr,
                xResultWasm.ptr, objValWasm.ptr, statusWasm.ptr
            );

            // 6. Parse Results
            const x = fromWasm(xResultWasm, c.shape);
            const fun = objValWasm.refresh().view[0];
            const status = statusWasm.refresh().view[0];
            const message = { 0: "Optimal", 1: "Infeasible", 2: "Unbounded", [-1]: "Error" }[status] || "Unknown";
            
            return { x, fun, status, message };

        } finally {
            // 7. Cleanup
            [cWasm, GWasm, hWasm, AWasm, bWasm, boundsWasm, xResultWasm, objValWasm, statusWasm].forEach(b => b?.dispose());
        }
    },

    /**
     * Fits a simple linear regression model: Y = alpha + beta*X.
     * @param {NDArray} x - The independent variable (1D NDArray of float64).
     * @param {NDArray} y - The dependent variable (1D NDArray of float64).
     * @returns {{alpha: number, beta: number}} - An object containing the intercept (alpha) and slope (beta) of the fitted line.
     * @throws {Error} If WASM runtime is not loaded or inputs are invalid.
     */
    linearRegression(x, y) {
        if (!NDWasm.runtime?.isLoaded) throw new Error("WasmRuntime not loaded.");
        if (x.ndim !== 1 || y.ndim !== 1 || x.size !== y.size) throw new Error("Inputs must be 1D arrays of the same length.");
        
        let xWasm, yWasm, alphaWasm, betaWasm;
        try {
            xWasm = x.toWasm(NDWasm.runtime);
            yWasm = y.toWasm(NDWasm.runtime);
            alphaWasm = NDWasm.runtime.createBuffer(1, 'float64');
            betaWasm = NDWasm.runtime.createBuffer(1, 'float64');
            
            NDWasm.runtime.exports.LinearRegression_F64(xWasm.ptr, yWasm.ptr, x.size, alphaWasm.ptr, betaWasm.ptr);
            
            const alpha = alphaWasm.refresh().view[0];
            const beta = betaWasm.refresh().view[0];
            return { alpha, beta };
        } finally {
            [xWasm, yWasm, alphaWasm, betaWasm].forEach(b => b?.dispose());
        }
    },
    /**
     * Fits a polynomial of the specified degree to a set of 2D points using least squares.
     * Model: Y = c0 + c1*X + c2*X^2 + ... + cd*X^d
     * @param {NDArray} x - The independent variable (1D NDArray of float64).
     * @param {NDArray} y - The dependent variable (1D NDArray of float64).
     * @param {number} degree - The degree of the polynomial to fit (non-negative integer).
     * @returns {Float64Array} - An array of coefficients in ascending order of degree [c0, c1, ..., cd].
     * @throws {Error} If WASM runtime is not loaded, inputs are invalid, or degree is negative.
     */
    polyfit(x, y, degree) {
        if (!NDWasm.runtime?.isLoaded) throw new Error("WasmRuntime not loaded.");
        if (x.ndim !== 1 || y.ndim !== 1 || x.size !== y.size) throw new Error("Inputs must be 1D arrays of the same length.");
        if (!Number.isInteger(degree) || degree < 0) throw new Error("Degree must be a non-negative integer.");
        
        let xWasm, yWasm, coeffsWasm;
        try {
            xWasm = x.toWasm(NDWasm.runtime);
            yWasm = y.toWasm(NDWasm.runtime);
            // Coefficient array size is degree + 1 (for c0 to cd)
            coeffsWasm = NDWasm.runtime.createBuffer(degree + 1, 'float64');
            
            // Call the exported Go WASM function
            NDWasm.runtime.exports.Polyfit_F64(xWasm.ptr, yWasm.ptr, x.size, degree, coeffsWasm.ptr);
            
            const coeffs = fromWasm(coeffsWasm, [degree + 1], 'float64');
            return coeffs;
        } finally {
            [xWasm, yWasm, coeffsWasm].forEach(b => b?.dispose());
        }
    },
    /**
     * Finds the minimum of a scalar function of one or more variables using an L-BFGS optimizer.
     * @param {Function} func - The objective function to be minimized. It must take a 1D `Float64Array` `x` (current point) and return a single number (the function value at `x`).
     * @param {NDArray} x0 - The initial guess for the optimization (1D NDArray of float64).
     * @param {Object} [options] - Optional parameters.
     * @param {Function} [options.grad] - The gradient of the objective function. Must take `x` (a 1D `Float64Array`) and write the result into the second argument `grad_out` (a 1D `Float64Array`). This function should *not* return a value.
     * @returns {{x: NDArray, success: boolean, message: string, ...stats}} The optimization result.
     */
    minimize(func, x0, options = {}) {
        if (!NDWasm.runtime?.isLoaded) throw new Error("WasmRuntime not loaded.");
        if (typeof func !== 'function') throw new Error("Objective 'func' must be a JavaScript function.");
        if (x0.ndim !== 1) throw new Error("Initial guess 'x0' must be a 1D NDArray.");

        const { grad } = options;
        let x0Wasm, resultWasm, statsWasm;

        try {
            // Workaround for js.Value bug: pass functions via global scope
            globalThis.ndarray_minimize_func = function(xPtr, size) {
                const xArr = new Float64Array(NDWasm.runtime.exports.mem.buffer, xPtr, size);
                return func(xArr);
            };
            globalThis.ndarray_minimize_grad = !grad?null:function(xPtr, gradPtr, size) {
                const xArr = new Float64Array(NDWasm.runtime.exports.mem.buffer, xPtr, size);
                const gradArr = new Float64Array(NDWasm.runtime.exports.mem.buffer, gradPtr, size);
                grad(xArr, gradArr);
            };

            x0Wasm = x0.toWasm(NDWasm.runtime);
            resultWasm = NDWasm.runtime.createBuffer(x0.size, 'float64');
            statsWasm = NDWasm.runtime.createBuffer(6, 'float64');

            NDWasm.runtime.exports.Minimize_F64(
                x0Wasm.ptr, x0.size,
                resultWasm.ptr,
                statsWasm.ptr
            );

            const resultArr = fromWasm(resultWasm, [x0.size], 'float64');
            const stats = fromWasm(statsWasm, [6], 'float64').toArray();
            
            const status = stats[0];
            const message = OPTIMIZE_STATUS_MAP[Math.abs(status)] || "Unknown status";

            return {
                x: resultArr,
                success: status > 0, // Success if status is Optimal
                status,
                message,
                fun: stats[1],
                niter: stats[2],
                nfev: stats[3],
                ngev: stats[4],
                runtime: stats[5],
            };

        } finally {
            // Clean up global scope
            delete globalThis.ndarray_minimize_func;
            delete globalThis.ndarray_minimize_grad;
            [x0Wasm, resultWasm, statsWasm].forEach(b => b?.dispose());
        }
    }
};