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
    },




    /**
     * @param {Function} odefun - (t, y, f_out, m_out)
     * @param {Array|NDArray} tspan -[t0, tfinal]
     * @param {Array|NDArray} y0 - Initial state
     * @param {Object} options - {absTol, relTol, initialStep, hasM, jac}
     * @param {number} options.jac: (t, y, f, index_out, val_out) => returns nnz (number of non-zeros)
     *     - index_out: NDArray of int32 with shape [max_nnz, 2]
     *     - val_out: NDArray of float64 with shape [max_nnz]
     * @param {boolean} options.hasM - Whether the ODE function uses the mass matrix M (i.e. M*dy/dt = f(t,y))
     * @param {number} options.absTol - Absolute tolerance for ODE solver
     * @param {number} options.relTol - Relative tolerance for ODE solver
     * @param {number} options.maxStep - Maximum number of steps for ODE solver
     * @param {number} options.maxTime - Maximum time for ODE solver in milliseconds
     */
    ode15s(odefun, tspan, y0, options = {}){
        return this.odeSolve(odefun, tspan, y0, {...options, method: 'ode15s'});
    },
    
    /**
     * @param {Function} odefun - (t, y, f_out, m_out)
     * @param {Array|NDArray} tspan -[t0, tfinal]
     * @param {Array|NDArray} y0 - Initial state
     * @param {Object} options - {absTol, relTol, initialStep, hasM, jac}
     * @param {number} options.jac: (t, y, f, index_out, val_out) => returns nnz (number of non-zeros)
     *     - index_out: NDArray of int32 with shape [max_nnz, 2]
     *     - val_out: NDArray of float64 with shape [max_nnz]
     * @param {boolean} options.hasM - Whether the ODE function uses the mass matrix M (i.e. M*dy/dt = f(t,y))
     * @param {number} options.absTol - Absolute tolerance for ODE solver
     * @param {number} options.relTol - Relative tolerance for ODE solver
     * @param {number} options.maxStep - Maximum number of steps for ODE solver
     * @param {number} options.maxTime - Maximum time for ODE solver in milliseconds
     */
    ode45(odefun, tspan, y0, options = {}) {
        return this.odeSolve(odefun, tspan, y0, {...options, method: 'ode45'});
    },



    /**
     * @param {Function} odefun - (t, y, f_out, m_out)
     * @param {Array|NDArray} tspan -[t0, tfinal]
     * @param {Array|NDArray} y0 - Initial state
     * @param {Object} options - {absTol, relTol, initialStep, method: 'ode45'|'ode15s', hasM, jac}
     * @param {number} options.jac: (t, y, f, index_out, val_out) => returns nnz (number of non-zeros)
     *     - index_out: NDArray of int32 with shape [max_nnz, 2]
     *     - val_out: NDArray of float64 with shape [max_nnz]
     * @param {boolean} options.hasM - Whether the ODE function uses the mass matrix M (i.e. M*dy/dt = f(t,y))
     * @param {number} options.absTol - Absolute tolerance for ODE solver
     * @param {number} options.relTol - Relative tolerance for ODE solver
     * @param {number} options.maxStep - Maximum number of steps for ODE solver
     * @param {number} options.maxTime - Maximum time for ODE solver in milliseconds
     */
    odeSolve(odefun, tspan, y0, options = {}) {
        if (!NDWasm.runtime?.isLoaded) throw new Error("WASM not loaded");

        y0 = (y0 instanceof NDArray) ? y0 : array(y0);
        tspan = (tspan instanceof NDArray) ? tspan : array(tspan);

        const dim = y0.size;
        const method = options.method === 'ode15s' ? 1 : 0;
        const hasM = options.hasM ? 1 : 0;
        const hasJac = typeof options.jac === 'function' ? 1 : 0;

        let y0Wasm, tspanWasm, configWasm, statsWasm;

        try {
            {
                const size = dim;
                const yArr = new NDArray(null, {shape: [size], dtype: 'float64'});
                const fArr = new NDArray(null, {shape: [size], dtype: 'float64'});
                const mArr = hasM ? new NDArray(null, {shape: [size], dtype: 'float64'}) : null;
                // 1. Setup Global Callbacks (Pointer-based for zero-copy performance)
                globalThis.ndarray_ode_func = (t, yPtr, fPtr, mPtr, size) => {
                    let data=new Float64Array(NDWasm.runtime.exports.mem.buffer,0);
                    yArr.data = data.subarray(yPtr/8, yPtr/8+size);
                    fArr.data = data.subarray(fPtr/8, fPtr/8+size);
                    let mArr = null;
                    if(hasM) {
                        mArr = data.subarray(mPtr/8, mPtr/8+size);
                    }
                    try{
                        odefun(t, yArr, fArr, mArr);
                    }catch(e){
                        console.error(`Error in odefun at t=${t}:`, e);
                        return false; // Indicate failure to ODE function
                    }
                    return true; // Indicate success
                };
            }
            {
                const size = dim;
                const maxNnz = size * size;
                const yArr = new NDArray(null, {shape: [size], dtype: 'float64'});
                const fArr = new NDArray(null, {shape: [size], dtype: 'float64'});
                
                // Int32Array for indices (row, col) with shape[maxNnz, 2]
                // Note: maxNnz * 2 total elements
                const indexArr = new NDArray(null, {shape: [maxNnz, 2], dtype: 'int32'});
                
                // Float64Array for non-zero values with shape [maxNnz]
                const valArr = new NDArray(null, {shape: [maxNnz], dtype: 'float64'});

                globalThis.ndarray_ode_jac = hasJac ? (t, yPtr, fPtr, indexPtr, valPtr, size) => {
                    let data=new Float64Array(NDWasm.runtime.exports.mem.buffer,0);
                    yArr.data = data.subarray(yPtr/8, yPtr/8+size);
                    fArr.data = data.subarray(fPtr/8, fPtr/8+size);
                    indexArr.data = new Int32Array(NDWasm.runtime.exports.mem.buffer, indexPtr, maxNnz * 2);
                    valArr.data = data.subarray(valPtr/8, valPtr/8+maxNnz);

                    // The jacobian function writes to indexArr and valArr and MUST return the integer length of non-zeros (nnz)
                    return options.jac(t, yArr, fArr, indexArr, valArr);
                } : null;
            }

            // 2. Pre-allocate Buffers
            y0Wasm = y0.toWasm(NDWasm.runtime);
            tspanWasm = tspan.toWasm(NDWasm.runtime);

            configWasm = NDWasm.runtime.createBuffer(4, 'float64');
            const cfgArr = new Float64Array(NDWasm.runtime.exports.mem.buffer, configWasm.ptr, 4);
            cfgArr[0] = options.absTol || 1e-6;
            cfgArr[1] = options.relTol || 1e-3;
            cfgArr[2] = options.initialStep || 0;
            cfgArr[3] = method;
            cfgArr[4] = options.maxStep || 2000000;
            cfgArr[5] = options.maxTime || 1200000; // in milliseconds


            // Stats buffer:[status, steps, runtime, tPtr, yPtr, dyPtr, length]
            statsWasm = NDWasm.runtime.createBuffer(7, 'float64');

            // 3. Execution
            NDWasm.runtime.exports.Ode_Solve_F64(
                y0Wasm.ptr, dim,
                tspanWasm.ptr,
                configWasm.ptr,
                statsWasm.ptr,
                hasM
            );

            // 4. Result Extraction (Zero-Copy extraction, followed by detachment)
            const stats = new Float64Array(NDWasm.runtime.exports.mem.buffer, statsWasm.ptr, 7);
            if (stats[0] !== 1) {
                options.status=stats[0];
                console.log(`Ode_Solve_F64 failed with status: ${stats[0]}`);
                return null;
            }
            const nPoints = stats[6];
            const tPtr = stats[3];
            const yPtr = stats[4];
            const dyPtr = stats[5];

            const T = new NDArray(new Float64Array(NDWasm.runtime.exports.mem.buffer, tPtr, nPoints), {shape:[nPoints], dtype: 'float64'}).copy(); 
            const Y = new NDArray(new Float64Array(NDWasm.runtime.exports.mem.buffer, yPtr, nPoints * dim), {shape: [nPoints, dim], dtype: 'float64'}).copy();
            const Dy = new NDArray(new Float64Array(NDWasm.runtime.exports.mem.buffer, dyPtr, nPoints * dim), {shape: [nPoints, dim], dtype: 'float64'}).copy();

            return {
                t: T,
                y: Y,
                dy: Dy,
                steps: stats[1],
                runtime: stats[2]
            };

        } finally {
            // 5. Cleanup
            delete globalThis.ndarray_ode_func;
            delete globalThis.ndarray_ode_jac;
            [y0Wasm, tspanWasm, configWasm, statsWasm].forEach(b => b?.dispose());
        }
    },


    /**
     * Solve 1D Parabolic and Elliptic PDEs using the method of lines with an underlying ODE15s integrator.
     * 
     * @param {number} m - Symmetry parameter: 0 (slab), 1 (cylinder), 2 (sphere)
     * @param {Function} pdefun - (x, t, u, dudx, c_out, f_out, s_out)
     * @param {Function} icfun - (x) => returns initial conditions as Array or NDArray
     * @param {Function} bcfun - (xl, ul, xr, ur, t, pl_out, ql_out, pr_out, qr_out)
     * @param {Array|NDArray} xmesh - Spatial grid points [x0, ..., xN]
     * @param {Array|NDArray} tspan - Time output points [t0, ..., tM]
     * @param {Object} options - {absTol: 1e-5, relTol: 1e-4}
     * @param {number} options.absTol - Absolute tolerance for ODE solver
     * @param {number} options.relTol - Relative tolerance for ODE solver
     * @param {number} options.maxStep - Maximum number of steps for ODE solver
     * @param {number} options.maxTime - Maximum time for ODE solver in milliseconds
     * @param {boolean} options.estimateError - Whether to return an error estimate
     * @param {boolean} options.estimatedError - The returned error estimate (if options.estimateError is true) will be a 2D tensor of shape [length(xmesh), dim] containing the estimated local truncation error at each point in space.
     * @returns {NDArray|null} 3D Tensor of shape [length(tspan), length(xmesh), dim]
     */
    pdepe(m, pdefun, icfun, bcfun, xmesh, tspan, options = {}) {
        if (!NDWasm.runtime?.isLoaded) throw new Error("WASM not loaded");

        const xArr = (xmesh instanceof NDArray) ? xmesh : array(xmesh);
        const tArr = (tspan instanceof NDArray) ? tspan : array(tspan);

        // 1. Infer spatial nodes and equation dimension (D) via a local probe
        const xLen = xArr.size;
        const tLen = tArr.size;
        const testU0 = icfun(xArr.get(0));
        const dim = testU0.size || testU0.length;

        let xWasm, tWasm, configWasm, statsWasm, errorWasm;

        try {
            let lastErrorLogged = null;

            // 2. Map Global JS Pointers (Zero-Copy direct writes)
            globalThis.ndarray_ic_fun = (x, uPtr, dim) => {
                const mem = NDWasm.runtime.exports.mem.buffer;
                const uArr = new Float64Array(mem, uPtr, dim);
                let res;
                try{
                    res = icfun(x);
                }catch(e){
                    lastErrorLogged = e;
                    console.error("Error in icfun at x =", x, ":", e);
                    return false;
                }
                uArr.set(res.data || res);
                return true;
            };

            const u = new NDArray(null, {shape: [dim], dtype: 'float64'});
            const dudx = new NDArray(null, {shape: [dim], dtype: 'float64'});
            const c = new NDArray(null, {shape: [dim], dtype: 'float64'});
            const f = new NDArray(null, {shape: [dim], dtype: 'float64'});
            const s = new NDArray(null, {shape: [dim], dtype: 'float64'});

            globalThis.ndarray_pde_fun = (x, t, uPtr, dudxPtr, cPtr, fPtr, sPtr, dim) => {
                const mem = NDWasm.runtime.exports.mem.buffer;
                // Create views into the WASM memory for each of the arrays based on the provided pointers and dimension
                let data=new Float64Array(mem,0);
                u.data=data.subarray(uPtr/8, uPtr/8+dim);
                dudx.data=data.subarray(dudxPtr/8, dudxPtr/8+dim);
                c.data=data.subarray(cPtr/8, cPtr/8+dim);
                f.data=data.subarray(fPtr/8, fPtr/8+dim);
                s.data=data.subarray(sPtr/8, sPtr/8+dim);
                try{
                    pdefun(x, t, u, dudx, c, f, s);
                }catch(e){
                    lastErrorLogged = e;
                    console.error(`Error in pdefun at x=${x}, t=${t}:`, e);
                    return false;
                }
                return true;
            };

            const ul = new NDArray(null, {shape: [dim], dtype: 'float64'});
            const ur = new NDArray(null, {shape: [dim], dtype: 'float64'});
            const pl = new NDArray(null, {shape: [dim], dtype: 'float64'});
            const ql = new NDArray(null, {shape: [dim], dtype: 'float64'});
            const pr = new NDArray(null, {shape: [dim], dtype: 'float64'});
            const qr = new NDArray(null, {shape: [dim], dtype: 'float64'});

            globalThis.ndarray_bc_fun = (xl, ulPtr, xr, urPtr, t, plPtr, qlPtr, prPtr, qrPtr, dim) => {
                const mem = NDWasm.runtime.exports.mem.buffer;
                let data=new Float64Array(mem,0);
                ul.data=data.subarray(ulPtr/8, ulPtr/8+dim);
                ur.data=data.subarray(urPtr/8, urPtr/8+dim);
                pl.data=data.subarray(plPtr/8, plPtr/8+dim);
                ql.data=data.subarray(qlPtr/8, qlPtr/8+dim);
                pr.data=data.subarray(prPtr/8, prPtr/8+dim);
                qr.data=data.subarray(qrPtr/8, qrPtr/8+dim);
                try{
                    bcfun(xl, ul, xr, ur, t, pl, ql, pr, qr);
                }catch(e){
                    lastErrorLogged = e;
                    console.error(`Error in bcfun at xl=${xl}, xr=${xr}, t=${t}:`, e);
                    return false;
                }
                return true;
            };

            const mem = NDWasm.runtime.exports.mem.buffer;
            // 3. Pre-allocate Configs
            xWasm = xArr.toWasm(NDWasm.runtime);
            tWasm = tArr.toWasm(NDWasm.runtime);

            configWasm = NDWasm.runtime.createBuffer(4, 'float64');
            const cfgArr = new Float64Array(mem, configWasm.ptr, 5);
            cfgArr[0] = options.absTol || 1e-5;
            cfgArr[1] = options.relTol || 1e-4;
            cfgArr[2] = options.maxStep || 2000000;
            cfgArr[3] = options.maxTime || 1200000; // in milliseconds
            cfgArr[4] = options.estimateError ? 1 : 0;

            // Stats: [status, runtime, solPtr]
            statsWasm = NDWasm.runtime.createBuffer(3, 'float64');
            errorWasm = options.estimateError ? NDWasm.runtime.createBuffer(xLen * dim, 'float64') : null;

            // 4. Execution
            NDWasm.runtime.exports.Pdepe_Solve_F64(
                m,
                xWasm.ptr, xLen,
                tWasm.ptr, tLen,
                dim,
                configWasm.ptr,
                statsWasm.ptr,
                errorWasm?.ptr ?? 0
            );

            if(lastErrorLogged){
                throw lastErrorLogged; // Surface the last logged error from user callbacks after execution
            }

            // 5. Post-Processing
            const stats = new Float64Array(NDWasm.runtime.exports.mem.buffer, statsWasm.ptr, 3);
            if (stats[0] !== 1) {
                options.status=stats[0];
                console.log(`Ode_Solve_F64 failed with status: ${stats[0]}`);
                return null;
            }

            const solPtr = stats[2];
            const runtimeMs = stats[1];

            // Map and Copy out the 3D Tensor: [tLen, xLen, dim]
            const solElements = tLen * xLen * dim;
            const solTensor = new NDArray(
                new Float64Array(NDWasm.runtime.exports.mem.buffer, solPtr, solElements), 
                { shape: [tLen, xLen, dim], dtype: 'float64' }
            ).copy();
            
            if(errorWasm){
                const errors = errorWasm.pull().reshape([xLen, dim]);
                options.estimatedError=errors;
            }

            options.runtime_ms=runtimeMs;
            

            return solTensor;

        } finally {
            // 6. Memory Cleanup
            delete globalThis.ndarray_ic_fun;
            delete globalThis.ndarray_pde_fun;
            delete globalThis.ndarray_bc_fun;
            [xWasm, tWasm, configWasm, statsWasm,errorWasm].forEach(b => b?.dispose());
        }
    }
};