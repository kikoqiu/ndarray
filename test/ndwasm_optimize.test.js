/**
 * File: ndwasm_optimize.test.js
 * Responsibility: Test suite for Optimization functions (linprog, minimize, etc.)
 */
const ndarray = require('../dist/ndarray.cjs');
const { NDWasm, WasmRuntime, NDWasmOptimize, array, zeros } = ndarray;

describe('NDWasmOptimize (WASM)', () => {
    
    beforeAll(async () => {
        const runtime = new WasmRuntime();
        await runtime.init({
            execUrl: 'dist/wasm_exec.js',
            wasmUrl: 'dist/ndarray_plugin.wasm'
        });
        NDWasm.bind(runtime);
    }, 30000);

    // --- 1. linearRegression ---
    describe('linearRegression', () => {
        test('case 1: simple positive correlation', () => {
            const x = array([1, 2, 3, 4, 5]);
            const y = array([2, 4, 5, 4, 5]); // y approx = x + 1
            const { alpha, beta } = NDWasmOptimize.linearRegression(x, y);
            expect(alpha).toBeCloseTo(2.2); // Intercept
            expect(beta).toBeCloseTo(0.6);  // Slope
        });

        test('case 2: no correlation', () => {
            const x = array([1, 2, 3, 4, 5]);
            const y = array([3, 3, 3, 3, 3]);
            const { alpha, beta } = NDWasmOptimize.linearRegression(x, y);
            expect(alpha).toBeCloseTo(3);
            expect(beta).toBeCloseTo(0);
        });

        test('case 3: perfect negative correlation', () => {
            const x = array([1, 2, 3]);
            const y = array([3, 2, 1]); // y = -x + 4
            const { alpha, beta } = NDWasmOptimize.linearRegression(x, y);
            expect(alpha).toBeCloseTo(4);
            expect(beta).toBeCloseTo(-1);
        });

        test('case 4: error on mismatched lengths', () => {
            const x = array([1, 2]);
            const y = array([1, 2, 3]);
            expect(() => NDWasmOptimize.linearRegression(x, y)).toThrow(/same length/);
        });
    });

    // --- 2. linprog (Linear Programming) ---
    describe('linprog', () => {
        // From scipy.optimize.linprog documentation
        test('case 1: simple minimization', () => {
            // min: -x[0] + 4x[1]
            // s.t. -3x[0] + x[1] <= 6
            //       x[0] + 2x[1] <= 4
            // x >= 0
            const c = array([-1, 4]);
            const G = array([[-3, 1], [1, 2]]);
            const h = array([6, 4]);
            //console.log("c:", c.toString());
            //console.log("G:", G.toString());
            //console.log("h:", h.toString());
            // The problem is implicitly in "Ax <= h" form.
            const { x, fun, status, message } = NDWasmOptimize.linprog(c, G, h, null, null);
            
            expect(status).toBe(0); // Optimal
            expect(message).toBe("Optimal");
            expect(fun).toBeCloseTo(-4); // Objective value
            expect(x.get(0)).toBeCloseTo(4); // x[0]
            expect(x.get(1)).toBeCloseTo(0); // x[1]
        });

        test('case 2: simple maximization', () => {
            // max: x[0] + 2x[1] (which is min: -x[0] - 2x[1])
            // s.t. 2x[0] + x[1] <= 20
            //     -4x[0] + 5x[1] <= 10
            //     -x[0] + 2x[1] >= -2  =>  x[0] - 2x[1] <= 2
            //      x[0], x[1] >= 0
            const c = array([-1, -2]);
            const G = array([[2, 1], [-4, 5], [1, -2]]);
            const h = array([20, 10, 2]);

            const { x, fun, status } = NDWasmOptimize.linprog(c, G, h, null, null);
            
            expect(status).toBe(0);
            expect(fun).toBeCloseTo(-20.714286);
            expect(x.get(0)).toBeCloseTo(6.428571428571429);
            expect(x.get(1)).toBeCloseTo(7.142857142857143);
        });

        test('case 3: infeasible problem', () => {
             // min x
             // s.t. x <= -1
             //      x >= 1  => -x <= -1
             const c = array([1]);
             const G = array([[1], [-1]]);
             const h = array([-1, -1]);
             const { status, message } = NDWasmOptimize.linprog(c, G, h, null, null);
             expect(status).toBe(1); // Infeasible
             expect(message).toBe("Infeasible");
        });

        test('case 4: unbounded problem', () => {
            // min -x
            // s.t. x >= 0 (no upper bound)
            const c = array([-1]);
            const G = zeros([0, 1]); // No constraints of type G*x <= h
            const h = zeros([0]);
            const { status, message } = NDWasmOptimize.linprog(c, G, h, null, null);
            expect(status).toBe(2); // Unbounded
            expect(message).toBe("Unbounded");
        });
    });

    // --- 3. minimize (Non-linear optimization) ---
    describe('minimize', () => {
        // Minimize the Rosenbrock function
        test('case 1: rosenbrock function without gradient', () => {
            const rosen = (x) => {
                return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2;
            };
            const x0 = zeros([2]);
            const res = NDWasmOptimize.minimize(rosen, x0);

            expect(res.success).toBe(true);
            expect(res.x.get(0)).toBeCloseTo(1.0);
            expect(res.x.get(1)).toBeCloseTo(1.0);
            expect(res.fun).toBeCloseTo(0.0);
        });

        test('case 2: rosenbrock function with gradient', () => {
            const rosen = (x) => {
                return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2;
            };
            const rosen_grad = (x, grad_out) => {
                grad_out[0] = -2 * (1 - x[0]) - 400 * (x[1] - x[0]**2) * x[0];
                grad_out[1] = 200 * (x[1] - x[0]**2);
            };
            const x0 = zeros([2]);
            const res = NDWasmOptimize.minimize(rosen, x0, { grad: rosen_grad });

            expect(res.success).toBe(true);
            expect(res.x.get(0)).toBeCloseTo(1.0);
            expect(res.x.get(1)).toBeCloseTo(1.0);
            expect(res.fun).toBeCloseTo(0.0);
            expect(res.ngev).toBeGreaterThan(0); // Check that gradient was evaluated
        });

        test('case 3: simple quadratic function f(x) = (x-2)^2', () => {
            const func = (x) => (x[0] - 2)**2;
            const x0 = array([0]);
            const res = NDWasmOptimize.minimize(func, x0);

            expect(res.success).toBe(true);
            expect(res.x.get(0)).toBeCloseTo(2.0);
            expect(res.fun).toBeCloseTo(0.0);
        });
    });







    describe('linprog: standard constraints and bounds merging', () => {

        /**
         * Case 1: Standard Inequality Constraints Only (Gx <= h)
         * Problem: Maximize x + y  => Minimize -x - y
         * Subject to:
         *   2x + y <= 10
         *   x + 2y <= 8
         *   x, y >= 0 (Default bounds)
         * Expected: Intersection of lines. 
         *   2x + y = 10  => y = 10 - 2x
         *   x + 2(10 - 2x) = 8 => x + 20 - 4x = 8 => -3x = -12 => x = 4
         *   y = 2
         *   Result: x=4, y=2, Obj = -(4+2) = -6
         */
        test('case 1: standard inequality constraints (Gx <= h)', () => {
            const c = array([-1, -1]);
            const G = array([
                [2, 1],
                [1, 2]
            ]);
            const h = array([10, 8]);

            // No A, b, or explicit bounds (defaults to >= 0)
            const { x, fun, status } = NDWasmOptimize.linprog(c, G, h, null, null, null);

            expect(status).toBe(0); // Optimal
            expect(x.get(0)).toBeCloseTo(4.0);
            expect(x.get(1)).toBeCloseTo(2.0);
            expect(fun).toBeCloseTo(-6.0);
        });

        /**
         * Case 2: Standard Equality Constraints Only (Ax = b)
         * Problem: Minimize x + y
         * Subject to:
         *   x + 2y = 10
         *   x, y >= 0 (Default)
         * Analysis:
         *   x = 10 - 2y. Since x >= 0, 10 - 2y >= 0 => 2y <= 10 => y <= 5.
         *   Obj = (10 - 2y) + y = 10 - y.
         *   To minimize Obj, maximize y. Max y = 5.
         *   Then x = 0.
         *   Result: x=0, y=5, Obj=5.
         */
        test('case 2: standard equality constraints (Ax = b)', () => {
            const c = array([1, 1]);
            const A = array([[1, 2]]);
            const b = array([10]);

            const { x, fun, status } = NDWasmOptimize.linprog(c, null, null, A, b, null);

            expect(status).toBe(0);
            expect(x.get(0)).toBeCloseTo(0.0);
            expect(x.get(1)).toBeCloseTo(5.0);
            expect(fun).toBeCloseTo(5.0);
        });

        /**
         * Case 3: Explicitly Unbounded (Fixed from previous request)
         * Problem: Minimize x
         * Subject to: x >= -10 (Provided via G matrix)
         * Bounds: [null, null] (Explicitly overrides default x >= 0)
         * Expected: x = -10
         */
        test('case 3: explicitly unbounded variables with G constraint', () => {
            const c = array([1]);
            
            // G constraint: -x <= 10 => x >= -10
            const G = array([[-1]]);
            const h = array([10]);
            
            // Explicitly set bounds to null to remove default non-negativity
            const bounds = [[null, null]];

            const { x, fun, status } = NDWasmOptimize.linprog(c, G, h, null, null, bounds);

            expect(status).toBe(0);
            expect(x.get(0)).toBeCloseTo(-10.0);
            expect(fun).toBeCloseTo(-10.0);
        });

        /**
         * Case 4: Mixed G, A and Bounds (Bounds act as additional constraints)
         * Problem: Minimize x0
         * Subject to:
         *   x0 + x1 = 10  (A)
         *   x0 <= 8       (G)
         *   x0 >= 2       (Bounds)
         *   x1 >= 0       (Default Bound)
         * Analysis:
         *   x0 range [2, 8].
         *   To minimize x0, choose x0 = 2.
         *   Then x1 = 8.
         */
        test('case 4: mixed G, A and bounds', () => {
            const c = array([1, 0]); // Min x0
            
            const A = array([[1, 1]]); // x0 + x1 = 10
            const b = array([10]);

            const G = array([[1, 0]]); // x0 <= 8
            const h = array([8]);

            const bounds = [[2, null], [0, null]]; // x0 >= 2

            const { x, fun, status } = NDWasmOptimize.linprog(c, G, h, A, b, bounds);

            expect(status).toBe(0);
            expect(x.get(0)).toBeCloseTo(2.0);
            expect(x.get(1)).toBeCloseTo(8.0);
            expect(fun).toBeCloseTo(2.0);
        });

        /**
         * Case 5: Bounds Tighter than G Constraints
         * Problem: Maximize x
         * Subject to:
         *   x <= 10 (G)
         *   x <= 5  (Bounds)
         * Expected: x = 5 (Bounds dominate)
         */
        test('case 5: bounds are tighter than G constraints', () => {
            const c = array([-1]); // Max x
            
            const G = array([[1]]); // x <= 10
            const h = array([10]);

            const bounds = [[0, 5]]; // 0 <= x <= 5

            const { x, fun, status } = NDWasmOptimize.linprog(c, G, h, null, null, bounds);

            expect(status).toBe(0);
            expect(x.get(0)).toBeCloseTo(5.0);
            expect(fun).toBeCloseTo(-5.0);
        });

        /**
         * Case 6: G Constraints Tighter than Bounds
         * Problem: Maximize x
         * Subject to:
         *   x <= 3 (G)
         *   x <= 10 (Bounds)
         * Expected: x = 3 (G constraints dominate)
         */
        test('case 6: G constraints are tighter than bounds', () => {
            const c = array([-1]);
            
            const G = array([[1]]); // x <= 3
            const h = array([3]);

            const bounds = [[0, 10]]; 

            const { x, fun, status } = NDWasmOptimize.linprog(c, G, h, null, null, bounds);

            expect(status).toBe(0);
            expect(x.get(0)).toBeCloseTo(3.0);
        });

        /**
         * Case 7: Infeasibility via Equality and Bounds intersection
         * Problem: Minimize x
         * Subject to:
         *   x = 5 (A)
         *   x <= 2 (Bounds)
         * Expected: Infeasible (Status 1)
         */
        test('case 7: infeasible intersection of equality A and bounds', () => {
            const c = array([1]);
            
            const A = array([[1]]); // x = 5
            const b = array([5]);

            const bounds = [[0, 2]]; // x <= 2

            const { status, message } = NDWasmOptimize.linprog(c, null, null, A, b, bounds);

            expect(status).toBe(1); // Infeasible
            expect(message).toBe("Infeasible");
        });

        /**
         * Case 8: Redundant Constraints (Multiple layers)
         * Verify that providing G, A, and Bounds simultaneously for the same variables
         * works correctly without numerical errors.
         * Problem: x = 5
         * Constraints: x <= 10, x >= 0, 2 <= x <= 8
         */
        test('case 8: redundant but consistent constraints', () => {
            const c = array([1]);
            
            const A = array([[1]]); // x = 5
            const b = array([5]);

            const G = array([[1]]); // x <= 10
            const h = array([10]);

            const bounds = [[2, 8]]; // 2 <= x <= 8

            const { x, status } = NDWasmOptimize.linprog(c, G, h, A, b, bounds);

            expect(status).toBe(0);
            expect(x.get(0)).toBeCloseTo(5.0);
        });

        /**
         * Case 9: Multi-variable Equality (Sum constraint) with Bounds
         * Problem: Minimize x0^2? No, linear. Min x0 + x1 + x2
         * Subject to:
         *   x0 + x1 + x2 = 1 (Simplex / Probability constraint)
         *   0.2 <= x_i <= 0.5 for all i
         * Analysis:
         *   Min 0.2 + 0.2 + 0.2 = 0.6 (Sum is not 1).
         *   We need x0+x1+x2=1. 
         *   Valid solution example: x=[0.2, 0.3, 0.5] -> Sum=1. Obj=1.
         *   Actually, since Obj = Sum(x) and Sum(x) = 1 (Constraint), the Obj value is constantly 1.
         *   The solver should just find *any* feasible point.
         */
        test('case 9: multi-variable sum equality with tight bounds', () => {
            const c = array([1, 1, 1]);
            
            const A = array([[1, 1, 1]]); // Sum = 1
            const b = array([1]);

            // Bounds force each component between 0.2 and 0.5
            const bounds = [
                [0.2, 0.5],
                [0.2, 0.5],
                [0.2, 0.5]
            ];

            const { x, fun, status } = NDWasmOptimize.linprog(c, null, null, A, b, bounds);

            expect(status).toBe(0);
            expect(fun).toBeCloseTo(1.0);
            
            // Check individual constraints
            const v0 = x.get(0), v1 = x.get(1), v2 = x.get(2);
            expect(v0).toBeGreaterThanOrEqual(0.199);
            expect(v0).toBeLessThanOrEqual(0.501);
            expect(v1 + v2 + v0).toBeCloseTo(1.0);
        });

        /**
         * Case 10: Specific "Unbounded" detection (if supported by solver)
         * Problem: Minimize -x
         * Subject to: x >= 5 (via Bounds)
         * No upper bound on x.
         * Since we are minimizing -x (Maximize x), and x can go to Infinity,
         * this should return Unbounded status.
         * Note: We provide a dummy A constraint to pass input validation if needed,
         * or G constraint that doesn't stop it.
         */
        test('case 10: unbounded problem detection with bounds', () => {
            const c = array([-1]);
            
            // Dummy constraint that doesn't bound x upwards: x >= 0
            const G = array([[-1]]);
            const h = array([0]);

            // Bound x >= 5. Upper bound is null (Infinity)
            const bounds = [[5, null]];

            const { status, message } = NDWasmOptimize.linprog(c, G, h, null, null, bounds);

            expect(status).toBe(2); // Unbounded
            expect(message).toBe("Unbounded");
        });

    });







    describe('linprog: complex and large-scale scenarios', () => {

        // --- Helper Functions for Data Generation ---


        // 验证结果是否满足 Ax <= b (允许误差)
        const checkFeasibility = (A, x, b, tolerance = 1e-4) => {
            // calculated = A * x
            // A: [m, n], x: [n]
            const m = A.shape[0];
            const n = A.shape[1];
            const Ax = new Float64Array(m);
            
            for(let i=0; i<m; i++) {
                let sum = 0;
                for(let j=0; j<n; j++) {
                    sum += A.get(i, j) * x.get(j);
                }
                Ax[i] = sum;
            }

            for(let i=0; i<m; i++) {
                // Ax <= b => Ax - b <= tolerance
                if (Ax[i] > b.get(i) + tolerance) {
                    return false;
                }
            }
            return true;
        };

        /**
         * Case 1: Random Dense Feasible Problem (50 vars, 30 constraints)
         * Method: "Method of Manufactured Solutions"
         * 1. Generate a random target solution x_sol (positive).
         * 2. Generate random matrix G.
         * 3. Set h = G * x_sol + slack (slack >= 0).
         * This guarantees x_sol is a feasible point, so the problem is Solvable.
         */
        test('case 1: dense random feasible problem (50 vars)', () => {
            const nVars = 50;
            const nCons = 30;
            
            const x_target = ndarray.random.random([nVars], 1, 10); // True feasible point
            const G = ndarray.random.random([nCons, nVars], 1, 10);

            // Calculate h to ensure x_target is feasible: h = G*x + random_slack
            // Manual matrix multiplication simulation for test setup
            let h = G.matVecMul(x_target);
            h=h.add(ndarray.random.random([nCons], 1, 5)); // Add slack
            
            //const c = ndarray.random.random([nVars], -1, 1); // Random objective
            const c = ndarray.ones([nVars]); // Simpler objective: Minimize sum(x)
            const { x, status, message } = NDWasmOptimize.linprog(c, G, h, null, null, [[3,null]]);
            //console.log("G:", G.toString());
            //console.log("h:", h.toString());
            //console.log("c:", c.toString());
            //console.log("x:", x.toString());

            expect(status).toBe(0); // Should work
            expect(message).toBe("Optimal");
            // Verify feasibility
            expect(checkFeasibility(G, x, h)).toBe(true);
        });

        /**
         * Case 2: The "Diet Problem" Scale-up (100 Foods, 20 Nutrients)
         * Minimize Cost s.t. Nutrition >= Requirement
         * Reformulated as: Minimize c*x, s.t. -A*x <= -b, x >= 0
         */
        test('case 2: large scale diet problem (100 vars, 20 constraints)', () => {
            const nFoods = 100;
            const nNutrients = 20;

            const c = ndarray.random.random([nFoods], 0.5, 20.0); // Food costs (positive)
            
            // Nutrient content matrix (Foods contribute positive nutrition)
            const A_nutrients = ndarray.random.random([nNutrients, nFoods], 0, 50); 
            const b_requirements = ndarray.random.random([nNutrients], 100, 500); // Daily reqs

            // Convert A*x >= b  -->  -A*x <= -b
            const G = A_nutrients.mul(-1);
            const h = b_requirements.mul(-1);

            // Bounds: x >= 0 (Default, but explicitly testing null passing works for scale)
            const { x, status } = NDWasmOptimize.linprog(c, G, h, null, null, null);

            expect(status).toBe(0);
            // Check if we met requirements (Ax >= b)
            // Or check Gx <= h (-Ax <= -b)
            expect(checkFeasibility(G, x, h)).toBe(true);
        });

        /**
         * Case 3: Combinatorial Relaxation (Knapsack-like)
         * 200 Items. 0 <= x_i <= 1.
         * Maximize Value (Minimize -Value) subject to Weight <= Capacity.
         * This stresses the 'bounds' array processing with many entries.
         */
        test('case 3: relaxed knapsack (200 variables, box constraints)', () => {
            const nItems = 200;
            const values = ndarray.random.random([nItems], 10, 100);
            const weights = ndarray.random.random([1, nItems], 1, 50); // 1xN matrix
            const capacity = nItems * 10; // Some reasonable capacity
            
            const c = values.mul(-1); // Maximize value
            const G = weights;
            const h = array([capacity]);

            // Create 200 bounds [0, 1]
            const bounds = Array(nItems).fill([0, 1]);

            const { x, status } = NDWasmOptimize.linprog(c, G, h, null, null, bounds);

            expect(status).toBe(0);
            // Check bounds manually
            for(let i=0; i<nItems; i++) {
                expect(x.get(i)).toBeGreaterThanOrEqual(-1e-6);
                expect(x.get(i)).toBeLessThanOrEqual(1.000001);
            }
        });

        /**
         * Case 4: Transportation Problem (Sparse / Network Structure)
         * 10 Suppliers, 10 Customers. Total 100 variables (x_ij).
         * Equality constraints: Sum(x_ij, j) = Supply_i
         * Equality constraints: Sum(x_ij, i) = Demand_j
         * Total Equality Constraints: 20.
         */
        test('case 4: transportation problem (equality constraints)', () => {
            const nSup = 10;
            const nDem = 10;
            const nVars = nSup * nDem;

            const c = ndarray.random.random([nVars], 1, 10); // Transport costs
            
            // Build A matrix for equalities: [nSup + nDem, nVars]
            // Row 0..9: Supply constraints
            // Row 10..19: Demand constraints
            const A_data = new Float64Array((nSup + nDem) * nVars);
            const b_data = new Float64Array(nSup + nDem);
            
            const supply = 100;
            const demand = 100;

            // Supply rows
            for(let i=0; i<nSup; i++) {
                b_data[i] = supply;
                for(let j=0; j<nDem; j++) {
                    // Variable index for x_ij is i*nDem + j
                    A_data[i * nVars + (i * nDem + j)] = 1;
                }
            }
            // Demand rows
            for(let j=0; j<nDem; j++) {
                b_data[nSup + j] = demand;
                for(let i=0; i<nSup; i++) {
                    A_data[(nSup + j) * nVars + (i * nDem + j)] = 1;
                }
            }

            const A = array(A_data).reshape([nSup + nDem, nVars]);
            const b = array(b_data);

            const { x, status } = NDWasmOptimize.linprog(c, null, null, A, b, null);

            expect(status).toBe(0);
            // Check total flow matches
            let totalFlow = 0;
            for(let k=0; k<nVars; k++) totalFlow += x.get(k);
            expect(totalFlow).toBeCloseTo(supply * nSup); // Should equal total supply
        });

        /**
         * Case 5: "Infeasible" High Dimensionality
         * 50 Variables.
         * Constraints: x_i <= 5 AND x_i >= 10.
         * This forces the solver to detect infeasibility quickly across many rows.
         */
        test('case 5: high-dim infeasible problem', () => {
            const nVars = 50;
            const c = ndarray.random.random([nVars]);

            // Constraint set 1: x <= 5  (Identity matrix)
            // Constraint set 2: x >= 10 => -x <= -10 (-Identity matrix)
            // Concatenate them to make G
            
            // Build G manually for clarity
            const G_data = new Float64Array(2 * nVars * nVars); 
            // First nVars rows: Identity
            for(let i=0; i<nVars; i++) G_data[i*nVars + i] = 1;
            // Next nVars rows: -Identity
            for(let i=0; i<nVars; i++) G_data[(nVars+i)*nVars + i] = -1;
            
            const G = array(G_data).reshape([2 * nVars, nVars]);

            // h vector: [5, 5, ..., 5, -10, -10, ..., -10]
            const h_data = new Float64Array(2 * nVars);
            h_data.fill(5, 0, nVars);
            h_data.fill(-10, nVars, 2 * nVars);
            const h = array(h_data);

            const { status, message } = NDWasmOptimize.linprog(c, G, h, null, null, null);

            expect(status).toBe(1); // Infeasible
            expect(message).toBe("Infeasible");
        });

        /**
         * Case 6: Multi-Period Production (Staircase Structure)
         * Model: inventory[t] = inventory[t-1] + prod[t] - demand[t]
         * Variables: P_0...P_9, I_0...I_9 (20 vars)
         * Constraints: 10 Equality constraints linking time steps.
         */
        test('case 6: multi-period production (staircase equalities)', () => {
            const T = 10;
            const nVars = 2 * T; // P_t, I_t
            
            // Min Cost: Production costs increase over time
            const c_data = new Float64Array(nVars);
            for(let t=0; t<T; t++) {
                c_data[t] = 10 + t; // Cost of P_t
                c_data[T + t] = 1;  // Cost of I_t (Holding cost)
            }
            const c = array(c_data);

            // Equalities: I_t - I_{t-1} - P_t = -Demand_t
            // For t=0: I_0 - P_0 = -D_0 (Assuming I_{-1} = 0)
            const A_data = new Float64Array(T * nVars);
            const b_data = new Float64Array(T);
            const Demand = 50;

            for(let t=0; t<T; t++) {
                b_data[t] = -Demand;
                
                // Coeff for I_t (index T+t) is 1
                A_data[t * nVars + (T + t)] = 1;
                // Coeff for P_t (index t) is -1
                A_data[t * nVars + t] = -1;
                
                if (t > 0) {
                    // Coeff for I_{t-1} (index T + t - 1) is -1
                    A_data[t * nVars + (T + t - 1)] = -1;
                }
            }

            const A = array(A_data).reshape([T, nVars]);
            const b = array(b_data);
            
            // Bounds: P, I >= 0 (Default)

            const { x, status } = NDWasmOptimize.linprog(c, null, null, A, b, null);

            expect(status).toBe(0);
            // Check logic: Total production should equal Total Demand (since last inventory cost > 0)
            let totalProd = 0;
            for(let t=0; t<T; t++) totalProd += x.get(t);
            expect(totalProd).toBeCloseTo(Demand * T);
        });

        /**
         * Case 7: Numerical Sensitivity (Ill-Conditioned)
         * Constraints lines are nearly parallel. 
         * x + y <= 10
         * x + 1.00000001y <= 10.00000001
         */
        test('case 7: numerically sensitive / ill-conditioned matrix', () => {
            const c = array([-1, -1]); // Max x+y
            
            // Almost singular matrix
            const G = array([
                [1, 1],
                [1, 1.00000001]
            ]);
            const h = array([10, 10.00000001]);

            const { x, status } = NDWasmOptimize.linprog(c, G, h, null, null, null);

            expect(status).toBe(0);
            // Should handle it without crashing or NaN
            expect(x.get(0)).not.toBeNaN();
            expect(x.get(1)).not.toBeNaN();
        });

        /**
         * Case 8: Large "Unbounded" Check
         * 100 Variables. Minimize sum(x).
         * Constraints: x_i - x_{i-1} >= 0 (Ascending order)
         * But no global lower bound on x_0. So x_0 can go to -Infinity.
         */
        test('case 8: large unbounded chain', () => {
            const nVars = 100;
            const c = array(Array(nVars).fill(1)); // Minimize sum

            // G: x_{i-1} - x_i <= 0  (x_i >= x_{i-1})
            // We need nVars-1 constraints.
            const G_data = new Float64Array((nVars - 1) * nVars);
            for(let i=0; i<nVars-1; i++) {
                // Row i constraint: x_i - x_{i+1} <= 0
                G_data[i * nVars + i] = 1;
                G_data[i * nVars + (i+1)] = -1;
            }
            const G = array(G_data).reshape([nVars - 1, nVars]);
            const h = zeros([nVars - 1]);

            // Explicitly set bounds to null to allow negative infinity
            const bounds = Array(nVars).fill([null, null]);

            const { status, message } = NDWasmOptimize.linprog(c, G, h, null, null, bounds);

            expect(status).toBe(2); // Unbounded
            expect(message).toBe("Unbounded");
        });

        /**
         * Case 9: L1 Regression Formulation (Minimize Absolute Error)
         * Fit y = a*x + b.
         * Data points: (1, 1), (2, 2.1), (3, 2.9), (4, 4.2), (5, 5).
         * Formulation: Min sum(u_i). s.t. u_i >= y_i - (ax_i + b), u_i >= -(y_i - (ax_i + b))
         * Variables: a, b, u_1...u_5 (Total 7 vars).
         * Bounds: u_i >= 0. a, b free.
         */
        test('case 9: L1 regression (auxiliary variables)', () => {
            const X = [1, 2, 3, 4, 5];
            const Y = [1, 2.1, 2.9, 4.2, 5]; // Roughly y=x
            const nPts = 5;
            
            // Vars: [a, b, u0, u1, u2, u3, u4]
            // Objective: 0*a + 0*b + 1*u0 ...
            const c = array([0, 0, 1, 1, 1, 1, 1]);

            // Constraints:
            // 1. y - (ax + b) - u <= 0  => -ax - b - u <= -y
            // 2. -(y - (ax+b)) - u <= 0 =>  ax + b - u <= y
            
            const nCons = 2 * nPts;
            const G_data = new Float64Array(nCons * 7);
            const h_data = new Float64Array(nCons);

            for(let i=0; i<nPts; i++) {
                const row1 = 2 * i;
                const row2 = 2 * i + 1;
                
                // Eq 1: -x*a - 1*b - 1*u_i <= -y
                G_data[row1 * 7 + 0] = -X[i]; // a
                G_data[row1 * 7 + 1] = -1;    // b
                G_data[row1 * 7 + (2 + i)] = -1; // u_i
                h_data[row1] = -Y[i];

                // Eq 2: x*a + 1*b - 1*u_i <= y
                G_data[row2 * 7 + 0] = X[i];  // a
                G_data[row2 * 7 + 1] = 1;     // b
                G_data[row2 * 7 + (2 + i)] = -1; // u_i
                h_data[row2] = Y[i];
            }

            const G = array(G_data).reshape([nCons, 7]);
            const h = array(h_data);

            // Bounds: a, b are free (-Inf, Inf). u_i >= 0 (Default is fine, but let's be explicit about a,b)
            const bounds = [
                [null, null], // a
                [null, null], // b
                // u0-u4 default >= 0 is correct
            ];

            const { x, status } = NDWasmOptimize.linprog(c, G, h, null, null, bounds);

            expect(status).toBe(0);
            expect(x.get(0)).toBeCloseTo(1.0, 1); // Slope approx 1
            expect(x.get(1)).toBeCloseTo(0.0, 1); // Intercept approx 0
        });

        /**
         * Case 10: Mixed Integer-like rounding check
         * 50 vars. 50 Constraints.
         * Inputs are integers, Solution should be close to integers but float.
         * Just ensuring high-density integer input doesn't cause type conversion issues in WASM.
         */
        test('case 10: dense integer input matrix', () => {
            const n = 50;
            // Matrix of all 1s and 2s
            const G_data = new Float64Array(n * n);
            for(let i=0; i<n*n; i++) G_data[i] = (i % 2 === 0) ? 1 : 2;
            const G = array(G_data).reshape([n, n]);
            
            const h = array(new Float64Array(n).fill(100));
            const c = array(new Float64Array(n).fill(-1)); // Maximize sum

            const { x, status } = NDWasmOptimize.linprog(c, G, h, null, null, null);

            expect(status).toBe(0);
            // Result check
            //console.log("Result x:", x.toString());
            // Values should be positive due to constraints
            expect(x.sum()).toBeCloseTo(100);
        });

    });



    describe('polyfit', () => {
        test('case 1: degree 1, simple positive correlation (matches linear regression)', () => {
            const x = array([1, 2, 3, 4, 5]);
            const y = array([2, 4, 5, 4, 5]); // y approx = 2.2 + 0.6x
            const coeffs = NDWasmOptimize.polyfit(x, y, 1);
            expect(coeffs.length).toBe(2);
            expect(coeffs[0]).toBeCloseTo(2.2); // Intercept (c0)
            expect(coeffs[1]).toBeCloseTo(0.6); // Slope (c1)
        });

        test('case 2: degree 1, perfect negative correlation', () => {
            const x = array([1, 2, 3]);
            const y = array([3, 2, 1]); // y = 4 - 1*x
            const coeffs = NDWasmOptimize.polyfit(x, y, 1);
            expect(coeffs[0]).toBeCloseTo(4);
            expect(coeffs[1]).toBeCloseTo(-1);
        });

        test('case 3: degree 2, perfect quadratic fit (y = x^2)', () => {
            const x = array([-2, -1, 0, 1, 2]);
            const y = array([4, 1, 0, 1, 4]); // y = 0 + 0*x + 1*x^2
            const coeffs = NDWasmOptimize.polyfit(x, y, 2);
            expect(coeffs.length).toBe(3);
            expect(coeffs[0]).toBeCloseTo(0);
            expect(coeffs[1]).toBeCloseTo(0);
            expect(coeffs[2]).toBeCloseTo(1);
        });

        test('case 4: degree 2, quadratic fit with all terms (y = 1.5 - 2x + 0.5x^2)', () => {
            const x = array([0, 1, 2, 3, 4]);
            const y = array([1.5, 0, -0.5, 0, 1.5]);
            const coeffs = NDWasmOptimize.polyfit(x, y, 2);
            expect(coeffs[0]).toBeCloseTo(1.5);
            expect(coeffs[1]).toBeCloseTo(-2);
            expect(coeffs[2]).toBeCloseTo(0.5);
        });

        test('case 5: degree 3, perfect cubic fit (y = 5 - 2x^2 + x^3)', () => {
            const x = array([-2, -1, 0, 1, 2]);
            // x=-2 => 5 - 8 - 8 = -11
            // x=-1 => 5 - 2 - 1 = 2
            // x=0  => 5
            // x=1  => 5 - 2 + 1 = 4
            // x=2  => 5 - 8 + 8 = 5
            const y = array([-11, 2, 5, 4, 5]);
            const coeffs = NDWasmOptimize.polyfit(x, y, 3);
            expect(coeffs.length).toBe(4);
            expect(coeffs[0]).toBeCloseTo(5);
            expect(coeffs[1]).toBeCloseTo(0);
            expect(coeffs[2]).toBeCloseTo(-2);
            expect(coeffs[3]).toBeCloseTo(1);
        });

        test('case 6: degree 0, constant fit (matches mean of y)', () => {
            const x = array([1, 2, 3, 4, 5]);
            const y = array([2, 4, 4, 4, 6]); // mean = 20 / 5 = 4
            const coeffs = NDWasmOptimize.polyfit(x, y, 0);
            expect(coeffs.length).toBe(1);
            expect(coeffs[0]).toBeCloseTo(4);
        });

        test('case 7: degree 4, perfect quartic fit (y = x^4)', () => {
            const x = array([-2, -1, 0, 1, 2]);
            const y = array([16, 1, 0, 1, 16]);
            const coeffs = NDWasmOptimize.polyfit(x, y, 4);
            expect(coeffs.length).toBe(5);
            expect(coeffs[0]).toBeCloseTo(0);
            expect(coeffs[1]).toBeCloseTo(0);
            expect(coeffs[2]).toBeCloseTo(0);
            expect(coeffs[3]).toBeCloseTo(0);
            expect(coeffs[4]).toBeCloseTo(1);
        });

        test('case 8: error on mismatched lengths', () => {
            const x = array([1, 2]);
            const y = array([1, 2, 3]);
            expect(() => NDWasmOptimize.polyfit(x, y, 1)).toThrow(/same length/);
        });

        test('case 9: error on negative degree', () => {
            const x = array([1, 2, 3]);
            const y = array([1, 2, 3]);
            expect(() => NDWasmOptimize.polyfit(x, y, -1)).toThrow(/non-negative integer/);
        });

        test('case 10: error on non-integer degree', () => {
            const x = array([1, 2, 3]);
            const y = array([1, 2, 3]);
            expect(() => NDWasmOptimize.polyfit(x, y, 1.5)).toThrow(/non-negative integer/);
        });
    });



    describe('polyfit - randomized high-degree tests', () => {
        // Generate 10 random test cases
        for (let i = 1; i <= 10; i++) {
            test(`case ${10 + i}: random degree 5-15, random data points 10-100`, () => {
                // 1. Randomly choose a degree between 5 and 15
                const degree = Math.floor(Math.random() * 11) + 5;
                
                // 2. Randomly choose number of data points between 10 and 100.
                // We also ensure that numPoints >= degree + 1 so the system is not underdetermined.
                const randomPoints = Math.floor(Math.random() * 91) + 10; 
                const numPoints = Math.max(degree + 1, randomPoints);

                // 3. Generate random true coefficients for the polynomial
                // Used to generate the y values. Kept between -5 and 5 to avoid overflow.
                const trueCoeffs =[];
                for (let j = 0; j <= degree; j++) {
                    trueCoeffs.push((Math.random() - 0.5) * 10);
                }

                // 4. Generate x and y data
                const xData = [];
                const yData =[];
                for (let k = 0; k < numPoints; k++) {
                    // It is crucial to keep x within [-1, 1] for high-degree polynomial fitting.
                    // Otherwise, x^15 can cause severe numerical instability (ill-conditioned Vandermonde matrix).
                    const xVal = (Math.random() - 0.5) * 2; 
                    let yVal = 0;
                    for (let j = 0; j <= degree; j++) {
                        yVal += trueCoeffs[j] * Math.pow(xVal, j);
                    }
                    xData.push(xVal);
                    yData.push(yVal);
                }

                const x = array(xData);
                const y = array(yData);

                // 5. Run polyfit
                const coeffs = NDWasmOptimize.polyfit(x, y, degree);

                // 6. Assertions
                expect(coeffs.length).toBe(degree + 1);
                
                for (let j = 0; j <= degree; j++) {
                    // Due to floating point arithmetic and the condition number of the matrix, 
                    // we use a 3-decimal-place tolerance for high-degree coefficients.
                    expect(coeffs[j]).toBeCloseTo(trueCoeffs[j], 3);
                }
            });
        }
    });
});
