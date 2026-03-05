const ndarray = require('../dist/ndarray.cjs');
const { NDWasm, WasmRuntime, NDWasmOptimize, array, zeros, linspace } = ndarray;


/**
 * Test suite for High-Performance ODE Solvers (ode45 & ode15s)
 * Using ndarray for vector operations and WASM-optimized backends.
 */
describe('NDArray ODE Solver Suite - ode45 & ode15s', () => {
    beforeAll(async () => {
        const runtime = new WasmRuntime();
        await runtime.init({
            execUrl: 'dist/wasm_exec.js',
            wasmUrl: 'dist/ndarray_plugin.wasm'
        });
        NDWasm.bind(runtime);
    }, 30000);


    // Default configuration as requested
    const defaultOptions = { absTol: 1e-6, relTol: 1e-5 };

    // Helper: Access the last value of a specific column in the result matrix
    const getLast = (y_res, col = 0) => y_res.get(-1, col);

    // 1. Simple Linear Integration: y' = 1, y(0) = 0 => y(t) = t
    test('1. Integrates constant derivative y\' = 1', () => {
        const odefun = (t, y, f, m) => f.set(1);
        const tspan = [0, 2];
        const y0 = [0];
        
        const res = ndarray.optimize.ode15s(odefun, tspan, y0);
        
        expect(res.y.get(-1,0)).toBeCloseTo(2, 20);
    });

    // ==========================================
    // 1. Basic Integration Tests
    // ==========================================

    test('1. Exponential Decay (y\' = -y)', () => {
        // Using ndarray vector operation: f = y * -1
        const odefun = (t, y, f) => f.set(y.mul(-1));
        const tspan = [0, 1];
        const y0 = [1.0];
        
        const res45 = ndarray.optimize.ode45(odefun, tspan, y0, defaultOptions);
        const res15s = ndarray.optimize.ode15s(odefun, tspan, y0, defaultOptions);
        
        const expected = Math.exp(-1.0);
        // Verify both explicit (ode45) and implicit (ode15s) solvers
        expect(getLast(res45.y, 0)).toBeCloseTo(expected, 4);
        expect(getLast(res15s.y, 0)).toBeCloseTo(expected, 4);
    });

    test('2. Simple Harmonic Oscillator (y\'\' = -y)', () => {
        // System: y0' = y1, y1' = -y0
        const odefun = (t, y, f) => {
            f.set([y.get(1), -y.get(0)]);
        };
        const tspan = [0, Math.PI];
        const y0 = [1.0, 0.0];
        
        const res = ndarray.optimize.ode45(odefun, tspan, y0, defaultOptions);
        // At t = PI: y[0] should be -1, y[1] should be 0
        expect(getLast(res.y, 0)).toBeCloseTo(-1.0, 3);
        expect(getLast(res.y, 1)).toBeCloseTo(0.0, 3);
    });

    test('3. Logistic Growth (y\' = y(1-y))', () => {
        const odefun = (t, y, f) => {
            const val = y.get(0);
            f.set([val * (1.0 - val)]);
        };
        const res = ndarray.optimize.ode15s(odefun, [0, 2], [0.5], defaultOptions);
        
        const expected = 1.0 / (1.0 + Math.exp(-2.0));
        expect(getLast(res.y, 0)).toBeCloseTo(expected, 4);
    });

    test('4. Reverse Integration (t_start > t_final)', () => {
        const odefun = (t, y, f) => f.set(y.mul(-1));
        // Integrating backwards from t=1 to t=0
        const res = ndarray.optimize.ode45(odefun, [1, 0], [Math.exp(-1.0)], defaultOptions);
        
        expect(getLast(res.y, 0)).toBeCloseTo(1.0, 3);
    });

    // ==========================================
    // 2. Stiff Systems and DAEs
    // ==========================================

    test('5. Robertson Problem (Classic Stiff System)', () => {
        const odefun = (t, y, f) => {
            const y0 = y.get(0), y1 = y.get(1), y2 = y.get(2);
            f.set([
                -0.04 * y0 + 1e4 * y1 * y2,
                0.04 * y0 - 1e4 * y1 * y2 - 3e7 * y1 * y1,
                3e7 * y1 * y1
            ]);
        };
        
        // Robertson problem requires stiff solvers like ode15s and tight tolerances
        const res = ndarray.optimize.ode15s(odefun, [0, 0.4], [1, 0, 0], { 
            absTol: 1e-8, 
            relTol: 1e-8 
        });
        
        expect(res).toBeDefined();
        // Mass conservation check: y0 + y1 + y2 should remain 1.0
        const mass = getLast(res.y, 0) + getLast(res.y, 1) + getLast(res.y, 2);
        expect(mass).toBeCloseTo(1.0, 6);
    });

    test('6. Index-1 DAE with Mass Matrix', () => {
        /**
         * Differential equation: y0' = y1
         * Algebraic equation: 0 = y0 + y1 - 1
         * Mass Matrix M = [1, 0; 0, 0]
         */
        const odefun = (t, y, f, m) => {
            const y0 = y.get(0), y1 = y.get(1);
            f.set([y1, y0 + y1 - 1.0]);
            
            // Set Diagonal of the Mass Matrix
            m.set([1.0, 0.0]);
        };
        
        // Specify hasM: true in options to enable Mass Matrix handling
        const res = ndarray.optimize.ode15s(odefun, [0, 1], [0.5, 0.5], { 
            ...defaultOptions,
            hasM: true 
        });
        
        const finalY0 = getLast(res.y, 0);
        const finalY1 = getLast(res.y, 1);
        // Constraint check: y0 + y1 must be 1.0
        expect(finalY0 + finalY1).toBeCloseTo(1.0, 5);
    });

    // ==========================================
    // 3. Performance & High-Dimensionality
    // ==========================================

    test('7. 1D Heat Equation (100 Dimensions)', () => {
        const N = 100;
        const dx = 1.0 / (N - 1);
        const alpha = 0.01;
        
        // Performance test: Direct access to underlying Float64Array (.data)
        const odefun = (t, y, f) => {
            const yd = y.data;
            const fd = f.data;
            const coef = alpha / (dx * dx);
            
            fd[0] = 0; // Boundary condition
            fd[N - 1] = 0; // Boundary condition
            for (let i = 1; i < N - 1; i++) {
                fd[i] = coef * (yd[i + 1] - 2 * yd[i] + yd[i - 1]);
            }
        };

        const y0 = new Array(N).fill(0).map((_, i) => Math.sin(Math.PI * i * dx));
        
        const start = Date.now();
        const res = ndarray.optimize.ode15s(odefun, [0, 0.5], y0, defaultOptions);
        const duration = Date.now() - start;
        
        expect(res).toBeDefined();
        expect(res.t.get(-1)).toBeCloseTo(0.5, 4);
        console.log(`Heat Eq (N=100) duration: ${duration}ms, steps: ${res.steps}`);
    });

    test('8. Large Scale Linear System (500 Dimensions)', () => {
        const N = 500;
        const odefun = (t, y, f) => {
            const yd = y.data;
            const fd = f.data;
            for (let i = 0; i < N; i++) {
                // Diagonal system: y_i' = -lambda_i * y_i
                fd[i] = -(i + 1) * 0.1 * yd[i];
            }
        };

        const y0 = new Array(N).fill(1.0);
        const res = ndarray.optimize.ode15s(odefun, [0, 1.0], y0, defaultOptions);
        
        expect(res).toBeDefined();
        // Check index 9: lambda = (9+1)*0.1 = 1.0. Result should be exp(-1)
        expect(getLast(res.y, 9)).toBeCloseTo(Math.exp(-1.0), 3);
    });

    // ==========================================
    // 4. Robustness & Error Handling
    // ==========================================

    test('9. Van der Pol Oscillator (mu=1000, Highly Stiff)', () => {
        const mu = 1000.0;
        const odefun = (t, y, f) => {
            const y0 = y.get(0), y1 = y.get(1);
            f.set([
                y1,
                mu * (1 - y0 * y0) * y1 - y0
            ]);
        };
        
        // This problem is very stiff and requires a BDF solver (ode15s)
        const res = ndarray.optimize.ode15s(odefun, [0, 1], [2, 0], {
            absTol: 1e-3,
            relTol: 1e-2
        });
        expect(res).not.toBeNull();
    });

    test('10. Custom Step Control - Initial Step', () => {
        const odefun = (t, y, f) => f.set([1.0]);
        const initialStep = 0.0123;
        
        const res = ndarray.optimize.ode45(odefun, [0, 1], [0], {
            ...defaultOptions,
            initialStep: initialStep
        });
        
        // The first successful time step should match our initial requirement
        expect(res.t.get(1)).toBeCloseTo(initialStep, 8);
    });
});






/**
 * Test suite for High-Performance PDEPE Solver (1D Parabolic and Elliptic PDEs)
 *  performance via WASM Zero-Allocation bindings.
 */
describe('NDArray PDE Solver Suite - pdepe', () => {
    beforeAll(async () => {
        const runtime = new WasmRuntime();
        await runtime.init({
            execUrl: 'dist/wasm_exec.js',
            wasmUrl: 'dist/ndarray_plugin.wasm'
        });
        NDWasm.bind(runtime);
    }, 30000);

    // ==========================================
    // Test Helpers
    // ==========================================

    const findIndex = (arr, target, tol) => arr.iterate((v,i) => Math.abs(v - target) < tol)[0];

    const assertCloseTo = (actual, expected, tol) => {
        expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tol);
    };

    const defaultOptions = { absTol: 1e-4, relTol: 1e-3 };
    const xmeshStd = linspace(0, 1, 21);
    const tspanStd = linspace(0, 0.5, 11);

    // ==========================================
    // Part I: Basic & Physical PDEs (1 - 10)
    // ==========================================
    describe('Part I: Physical and Mathematical PDEs', () => {

        test('1. Standard 1D Heat Equation (Slab, m=0)', () => {
            const pdefun = (x, t, u, dudx, c, f, s) => {
                c.data[0] = 1.0;
                f.data[0] = dudx.data[0];
                s.data[0] = 0.0;
            };
            const icfun = (x) => [Math.sin(Math.PI * x)];
            const bcfun = (xl, ul, xr, ur, t, pl, ql, pr, qr) => {
                pl.data[0] = ul.data[0]; ql.data[0] = 0.0;
                pr.data[0] = ur.data[0]; qr.data[0] = 0.0;
            };

            const sol = ndarray.optimize.pdepe(0, pdefun, icfun, bcfun, xmeshStd, tspanStd, defaultOptions);
            
            // Analytical solution: u(x,t) = exp(-pi^2 * t) * sin(pi * x)
            const expected = Math.exp(-Math.PI * Math.PI * 0.5) * Math.sin(Math.PI * 0.5);
            assertCloseTo(sol.get(10, 10, 0), expected, 0.01);
        });

        test('2. Cylindrical Heat Equation (m=1)', () => {
            const pdefun = (x, t, u, dudx, c, f, s) => {
                c.data[0] = 1.0; f.data[0] = dudx.data[0]; s.data[0] = 0.0;
            };
            const icfun = (x) => [1.0 - x * x];
            const bcfun = (xl, ul, xr, ur, t, pl, ql, pr, qr) => {
                pl.data[0] = 0.0; ql.data[0] = 1.0;
                pr.data[0] = ur.data[0]; qr.data[0] = 0.0;
            };

            const sol = ndarray.optimize.pdepe(1, pdefun, icfun, bcfun, linspace(0, 1, 21), [0, 0.1], defaultOptions);
            expect(sol.get(1, 0, 0)).toBeGreaterThan(0.0);
        });

        test('3. Spherical Heat Equation (m=2)', () => {
            const pdefun = (x, t, u, dudx, c, f, s) => {
                c.data[0] = 1.0; f.data[0] = dudx.data[0]; s.data[0] = 0.0;
            };
            const icfun = (x) => [1.0];
            const bcfun = (xl, ul, xr, ur, t, pl, ql, pr, qr) => {
                pl.data[0] = 0.0; ql.data[0] = 1.0;
                pr.data[0] = ur.data[0]; qr.data[0] = 0.0;
            };

            const sol = ndarray.optimize.pdepe(2, pdefun, icfun, bcfun, linspace(0, 1, 21), [0, 0.1], defaultOptions);
            expect(sol).toBeDefined();
        });

        test('5. Advection-Diffusion Equation', () => {
            const pdefun = (x, t, u, dudx, c, f, s) => {
                c.data[0] = 1.0;
                f.data[0] = dudx.data[0] * 0.1;
                s.data[0] = -dudx.data[0];
            };
            const icfun = (x) =>[Math.exp(-100 * Math.pow(x - 0.5, 2))];
            const bcfun = (xl, ul, xr, ur, t, pl, ql, pr, qr) => {
                pl.data[0] = ul.data[0]; ql.data[0] = 0.0;
                pr.data[0] = ur.data[0]; qr.data[0] = 0.0;
            };

            const sol = ndarray.optimize.pdepe(0, pdefun, icfun, bcfun, linspace(0, 1, 51), [0, 0.1], defaultOptions);
            expect(sol).toBeDefined();
        });

        test('7. Viscous Burgers Equation', () => {
            const pdefun = (x, t, u, dudx, c, f, s) => {
                c.data[0] = 1.0;
                f.data[0] = dudx.data[0];
                s.data[0] = -u.data[0] * dudx.data[0];
            };
            const icfun = (x) => [Math.sin(Math.PI * x)];
            const bcfun = (xl, ul, xr, ur, t, pl, ql, pr, qr) => {
                pl.data[0] = ul.data[0]; ql.data[0] = 0.0;
                pr.data[0] = ur.data[0]; qr.data[0] = 0.0;
            };

            const sol = ndarray.optimize.pdepe(0, pdefun, icfun, bcfun, xmeshStd,[0, 0.5], defaultOptions);
            expect(sol).toBeDefined();
        });

        test('9. System of 2 Coupled Heat Equations (D=2)', () => {
            const pdefun = (x, t, u, dudx, c, f, s) => {
                c.data[0] = 1.0; c.data[1] = 1.0;
                f.data[0] = dudx.data[0]; f.data[1] = dudx.data[1];
                s.data[0] = u.data[1]; s.data[1] = u.data[0]; // Intercoupled sources
            };
            const icfun = (x) => [1.0, 0.0];
            const bcfun = (xl, ul, xr, ur, t, pl, ql, pr, qr) => {
                pl.data[0] = ul.data[0]; pl.data[1] = ul.data[1];
                ql.data[0] = 0.0; ql.data[1] = 0.0;
                pr.data[0] = ur.data[0]; pr.data[1] = ur.data[1];
                qr.data[0] = 0.0; qr.data[1] = 0.0;
            };

            const sol = ndarray.optimize.pdepe(0, pdefun, icfun, bcfun, xmeshStd, [0, 0.1], defaultOptions);
            expect(sol.shape[2]).toBe(2); // Output dimension should be 2
        });
    });

    // ==========================================
    // Part II: Financial Engineering Tests
    // ==========================================
    describe('Part II: Financial Engineering', () => {
        const finOptions = { absTol: 1e-3, relTol: 1e-2 };
        const rVal = 0.05, qVal = 0.0, sigmaVal = 0.2, K = 100.0;
        const SMesh = linspace(1e-4, 250, 101);
        const TSpan = linspace(0, 1.0, 11);

        const createBSPDE = (r, q, sig) => {
            const sigSq = sig * sig;
            return (S, tau, u, dudx, c, f, s) => {
                const V = u.data[0];
                const dVdS = dudx.data[0];
                c.data[0] = 1.0;
                f.data[0] = 0.5 * sigSq * S * S * dVdS;
                s.data[0] = (r - q - sigSq) * S * dVdS - r * V;
            };
        };

        test('11. European Call Option (ATM)', () => {
            const pdefun = createBSPDE(rVal, qVal, sigmaVal);
            const icfun = (S) =>[Math.max(S - K, 0)];
            const bcfun = (Sl, ul, Sr, ur, tau, pl, ql, pr, qr) => {
                pl.data[0] = ul.data[0]; ql.data[0] = 0.0;
                pr.data[0] = ur.data[0] - (Sr - K * Math.exp(-rVal * tau)); qr.data[0] = 0.0;
            };

            const sol = ndarray.optimize.pdepe(0, pdefun, icfun, bcfun, SMesh, TSpan, finOptions);
            const idx100 = findIndex(SMesh, 100.0, 1.0);
            assertCloseTo(sol.get(10, idx100, 0), 10.45, 0.2);
        });

        test('14. European Put Option (ATM)', () => {
            const pdefun = createBSPDE(rVal, qVal, sigmaVal);
            const icfun = (S) => [Math.max(K - S, 0)];
            const bcfun = (Sl, ul, Sr, ur, tau, pl, ql, pr, qr) => {
                pl.data[0] = ul.data[0] - K * Math.exp(-rVal * tau); ql.data[0] = 0.0;
                pr.data[0] = ur.data[0]; qr.data[0] = 0.0;
            };

            const sol = ndarray.optimize.pdepe(0, pdefun, icfun, bcfun, SMesh, TSpan, finOptions);
            const idx100 = findIndex(SMesh, 100.0, 1.0);
            assertCloseTo(sol.get(10, idx100, 0), 5.57, 0.2); // Put-Call Parity validation
        });

        test('30. Put-Call Parity System (Coupled PDEs D=2)', () => {
            const sigSq = sigmaVal * sigmaVal;
            const pdefun = (S, tau, u, dudx, c, f, s) => {
                // D=2 System: u[0]=Call, u[1]=Put
                f.data[0] = 0.5 * sigSq * S * S * dudx.data[0];
                f.data[1] = 0.5 * sigSq * S * S * dudx.data[1];
                s.data[0] = (rVal - sigSq) * S * dudx.data[0] - rVal * u.data[0];
                s.data[1] = (rVal - sigSq) * S * dudx.data[1] - rVal * u.data[1];
                c.data[0] = 1.0; c.data[1] = 1.0;
            };
            const icfun = (S) =>[Math.max(S - K, 0), Math.max(K - S, 0)];
            const bcfun = (Sl, ul, Sr, ur, tau, pl, ql, pr, qr) => {
                const callAsymp = Sr - K * Math.exp(-rVal * tau);
                pl.data[0] = ul.data[0]; pl.data[1] = ul.data[1] - K * Math.exp(-rVal * tau);
                ql.data[0] = 0.0; ql.data[1] = 0.0;
                pr.data[0] = ur.data[0] - callAsymp; pr.data[1] = ur.data[1];
                qr.data[0] = 0.0; qr.data[1] = 0.0;
            };

            const sol = ndarray.optimize.pdepe(0, pdefun, icfun, bcfun, SMesh, TSpan, finOptions);
            const idx100 = findIndex(SMesh, 100.0, 1.0);
            
            const call = sol.get(10, idx100, 0);
            const put = sol.get(10, idx100, 1);
            
            // Parity Check: C - P = S - K*e^-rt
            const actualDiff = call - put;
            const expectedDiff = 100.0 - 100.0 * Math.exp(-0.05 * 1.0);
            assertCloseTo(actualDiff, expectedDiff, 0.1);
        });

        test('Butterfly Spread Option Pricing', () => {
            const T = 1.0, K1 = 90.0, K2 = 100.0, K3 = 110.0;
            const sMesh = linspace(0.0, 300.0, 500);
            const tauMesh = linspace(0, T, 200);

            const pdefun = (x, tau, u, dudx, c, f, s) => {
                const rLocal = 0.03 + 0.02 * (tau / T);
                const sigmaLocal = 0.2 + 0.1 * Math.sin(2 * Math.PI * tau / T);
                const sigSq = sigmaLocal * sigmaLocal;
                const xSq = x * x;

                c.data[0] = 1.0;
                f.data[0] = 0.5 * sigSq * xSq * dudx.data[0];
                s.data[0] = (rLocal - sigSq) * x * dudx.data[0] - rLocal * u.data[0];
            };

            const icfun = (x) =>[Math.max(x - K1, 0) - 2 * Math.max(x - K2, 0) + Math.max(x - K3, 0)];
            const bcfun = (xl, ul, xr, ur, tau, pl, ql, pr, qr) => {
                pl.data[0] = ul.data[0]; ql.data[0] = 0.0;
                pr.data[0] = ur.data[0]; qr.data[0] = 0.0;
            };

            const sol = ndarray.optimize.pdepe(0, pdefun, icfun, bcfun, sMesh, tauMesh, finOptions);
            const k2Idx = findIndex(sMesh, K2, 0.5);
            
            expect(sol).toBeDefined();
            const centerPrice = sol.get(199, k2Idx, 0);
            expect(centerPrice).toBeGreaterThan(0);
        });
    });

    // ==========================================
    // Big Data / Stress Tests for pdepe
    // ==========================================
    describe('Big Data / Stress Tests', () => {
        const stressOptions = { absTol: 1e-5, relTol: 1e-4 };

        test('BigData_1_UltraHighRes_N5000', () => {
            const N = 5000;
            const xmesh = linspace(0, 1, N);
            
            const pdefun = (x, t, u, dudx, c, f, s) => {
                c.data[0] = 1.0; f.data[0] = dudx.data[0]; s.data[0] = 0.0;
            };
            const icfun = (x) =>[Math.sin(Math.PI * x)];
            const bcfun = (xl, ul, xr, ur, t, pl, ql, pr, qr) => {
                pl.data[0] = ul.data[0]; ql.data[0] = 0.0;
                pr.data[0] = ur.data[0]; qr.data[0] = 0.0;
            };

            const start = Date.now();
            const sol = ndarray.optimize.pdepe(0, pdefun, icfun, bcfun, xmesh,[0, 0.05], stressOptions);
            const duration = Date.now() - start;
            
            console.log(`[Stress 1] N=${N}, Time=${duration}ms`);
            expect(sol.shape[1]).toBe(N);
        }, 30000); // 30s timeout

        test('BigData_2_LargeSystem_D50', () => {
            const D = 50, N = 50;
            const xmesh = linspace(0, 1, N);

            // Accessing raw Float64Array elements ensures zero-allocation and max speed
            const pdefun = (x, t, u, dudx, c, f, s) => {
                const ud = u.data;
                const dd = dudx.data;
                const cd = c.data;
                const fd = f.data;
                const sd = s.data;
                
                for (let i = 0; i < D; i++) {
                    cd[i] = 1.0;
                    fd[i] = dd[i];
                    const prev = i > 0 ? ud[i - 1] : 0.0;
                    const next = i < D - 1 ? ud[i + 1] : 0.0;
                    sd[i] = prev - 2 * ud[i] + next;
                }
            };
            const icfun = (x) => new Array(D).fill(1.0 - x);
            const bcfun = (xl, ul, xr, ur, t, pl, ql, pr, qr) => {
                for (let i = 0; i < D; i++) {
                    pl.data[i] = ul.data[i] - 1.0; ql.data[i] = 0.0;
                    pr.data[i] = ur.data[i];       qr.data[i] = 0.0;
                }
            };

            const start = Date.now();
            const sol = ndarray.optimize.pdepe(0, pdefun, icfun, bcfun, xmesh, [0, 0.1], stressOptions);
            const duration = Date.now() - start;

            console.log(`[Stress 2] D=${D}, N=${N}, TotalDOF=${D*N}, Time=${duration}ms`);
            expect(sol.shape[2]).toBe(D);
        }, 60000); // 60s timeout

        test('BigData_4_PortfolioGrid_D20', () => {
            const D = 20, N = 100;
            const xmesh = linspace(1e-4, 200, N);
            const tspan = linspace(0, 1.0, 21);
            const vols = linspace(0.1, 0.5, D);
            const sigSqn = vols.mul(vols).data;
            const r = 0.05;
            const pdefun = (S, tau, u, dudx, c, f, s) => {
                const ud = u.data; const dd = dudx.data;
                const cd = c.data; const fd = f.data; const sd = s.data;

                for (let i = 0; i < D; i++) {
                    const sigSq = sigSqn[i];
                    cd[i] = 1.0;
                    fd[i] = 0.5 * sigSq * S * S * dd[i];
                    sd[i] = (r - sigSq) * S * dd[i] - r * ud[i];
                }
            };
            const icfun = (S) => new Array(D).fill(Math.max(S - 100.0, 0.0));
            const bcfun = (xl, ul, xr, ur, t, pl, ql, pr, qr) => {
                for (let i = 0; i < D; i++) {
                    pl.data[i] = ul.data[i]; ql.data[i] = 0.0;
                    pr.data[i] = ur.data[i] - (xr - 100.0 * Math.exp(-r * t)); qr.data[i] = 0.0;
                }
            };

            const start = Date.now();
            const sol = ndarray.optimize.pdepe(0, pdefun, icfun, bcfun, xmesh, tspan,
                 {...stressOptions, absTol: 1e-2, relTol: 1e-1 });
            const duration = Date.now() - start;

            console.log(`[Stress 4] D=${D} assets, N=${N}, Time=${duration}ms`);
            expect(sol).toBeDefined();
            // Ground truth analytic value
            assertCloseTo(sol.get(20, 50, 10), 15.278, 0.5);
        }, 60000); // 60s timeout

    });
});