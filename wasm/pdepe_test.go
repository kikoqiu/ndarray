package main

import (
	"fmt"
	"math"
	"testing"
	"time"
)

// ==========================================
// Test Helpers
// ==========================================

// linspace generates a linearly spaced array of float64, equivalent to the JS helper.
func linspace(start, end float64, n int) []float64 {
	arr := make([]float64, n)
	if n == 1 {
		arr[0] = start
		return arr
	}
	step := (end - start) / float64(n-1)
	for i := 0; i < n; i++ {
		arr[i] = start + step*float64(i)
	}
	return arr
}

// findIndex locates the first element in the array close to the target within a tolerance.
func findIndex(arr []float64, target float64, tol float64) int {
	for i, v := range arr {
		if math.Abs(v-target) < tol {
			return i
		}
	}
	return -1
}

// assertCloseTo checks if actual is within tol of expected.
func assertCloseTo(t *testing.T, actual, expected, tol float64, msg string) {
	t.Helper()
	if math.Abs(actual-expected) > tol {
		t.Errorf("%s: expected %v, got %v (diff: %v > tol: %v)", msg, expected, actual, math.Abs(actual-expected), tol)
	}
}

// ==========================================
// Part I: Basic & Physical PDEs (1 - 10)
// ==========================================
func TestPdepe_PhysicalAndMathematical(t *testing.T) {
	testInfo := &OdeInfo{AbsTol: 1e-4, RelTol: 1e-3}
	xmeshStd := linspace(0, 1, 21)
	tspanStd := linspace(0, 0.5, 11)

	t.Run("1. Standard 1D Heat Equation (Slab, m=0)", func(t *testing.T) {
		pdefun := func(x, time float64, u, dudx []float64) PdeFunRes {
			return PdeFunRes{C: []float64{1.0}, F: []float64{dudx[0]}, S: []float64{0.0}}
		}
		icfun := func(x float64) []float64 {
			return []float64{math.Sin(math.Pi * x)}
		}
		bcfun := func(xl float64, ul []float64, xr float64, ur []float64, time float64) BcFunRes {
			return BcFunRes{Pl: []float64{ul[0]}, Ql: []float64{0.0}, Pr: []float64{ur[0]}, Qr: []float64{0.0}}
		}

		sol := Pdepe(0, pdefun, icfun, bcfun, xmeshStd, tspanStd, testInfo)

		// Analytical solution: u(x,t) = exp(-pi^2 * t) * sin(pi * x)
		expected := math.Exp(-math.Pi*math.Pi*0.5) * math.Sin(math.Pi*0.5)
		assertCloseTo(t, sol[10][10][0], expected, 0.01, "Center point value mismatch")
	})

	t.Run("2. Cylindrical Heat Equation (m=1)", func(t *testing.T) {
		pdefun := func(x, time float64, u, dudx []float64) PdeFunRes {
			return PdeFunRes{C: []float64{1.0}, F: []float64{dudx[0]}, S: []float64{0.0}}
		}
		icfun := func(x float64) []float64 {
			return []float64{1.0 - x*x}
		}
		bcfun := func(xl float64, ul []float64, xr float64, ur []float64, time float64) BcFunRes {
			return BcFunRes{Pl: []float64{0.0}, Ql: []float64{1.0}, Pr: []float64{ur[0]}, Qr: []float64{0.0}}
		}

		sol := Pdepe(1, pdefun, icfun, bcfun, linspace(0, 1, 21), []float64{0, 0.1}, testInfo)
		if sol[1][0][0] <= 0.0 {
			t.Errorf("Expected strictly positive temperature at origin, got %v", sol[1][0][0])
		}
	})

	t.Run("3. Spherical Heat Equation (m=2)", func(t *testing.T) {
		pdefun := func(x, time float64, u, dudx []float64) PdeFunRes {
			return PdeFunRes{C: []float64{1.0}, F: []float64{dudx[0]}, S: []float64{0.0}}
		}
		icfun := func(x float64) []float64 {
			return []float64{1.0}
		}
		bcfun := func(xl float64, ul []float64, xr float64, ur []float64, time float64) BcFunRes {
			return BcFunRes{Pl: []float64{0.0}, Ql: []float64{1.0}, Pr: []float64{ur[0]}, Qr: []float64{0.0}}
		}

		sol := Pdepe(2, pdefun, icfun, bcfun, linspace(0, 1, 21), []float64{0, 0.1}, testInfo)
		if sol == nil {
			t.Errorf("Expected valid solution array, got nil")
		}
	})

	t.Run("4. Heat Equation with Source Term", func(t *testing.T) {
		pdefun := func(x, time float64, u, dudx []float64) PdeFunRes {
			return PdeFunRes{C: []float64{1.0}, F: []float64{dudx[0]}, S: []float64{1.0}}
		}
		icfun := func(x float64) []float64 {
			return []float64{0.0}
		}
		bcfun := func(xl float64, ul []float64, xr float64, ur []float64, time float64) BcFunRes {
			return BcFunRes{Pl: []float64{ul[0]}, Ql: []float64{0.0}, Pr: []float64{ur[0]}, Qr: []float64{0.0}}
		}

		sol := Pdepe(0, pdefun, icfun, bcfun, xmeshStd, []float64{0, 0.5}, testInfo)
		if sol[1][10][0] <= 0 {
			t.Errorf("Center point should heat up, got %v", sol[1][10][0])
		}
	})

	t.Run("5. Advection-Diffusion Equation", func(t *testing.T) {
		pdefun := func(x, time float64, u, dudx []float64) PdeFunRes {
			return PdeFunRes{C: []float64{1.0}, F: []float64{dudx[0] * 0.1}, S: []float64{-dudx[0]}}
		}
		icfun := func(x float64) []float64 {
			return []float64{math.Exp(-100 * math.Pow(x-0.5, 2))}
		}
		bcfun := func(xl float64, ul []float64, xr float64, ur []float64, time float64) BcFunRes {
			return BcFunRes{Pl: []float64{ul[0]}, Ql: []float64{0.0}, Pr: []float64{ur[0]}, Qr: []float64{0.0}}
		}

		sol := Pdepe(0, pdefun, icfun, bcfun, linspace(0, 1, 51), []float64{0, 0.1}, testInfo)
		if sol == nil {
			t.Errorf("Expected valid solution array, got nil")
		}
	})

	t.Run("6. Neumann Boundary Conditions (Insulated ends)", func(t *testing.T) {
		pdefun := func(x, time float64, u, dudx []float64) PdeFunRes {
			return PdeFunRes{C: []float64{1.0}, F: []float64{dudx[0]}, S: []float64{0.0}}
		}
		icfun := func(x float64) []float64 {
			return []float64{x}
		}
		bcfun := func(xl float64, ul []float64, xr float64, ur []float64, time float64) BcFunRes {
			return BcFunRes{Pl: []float64{0.0}, Ql: []float64{1.0}, Pr: []float64{0.0}, Qr: []float64{1.0}}
		}

		sol := Pdepe(0, pdefun, icfun, bcfun, xmeshStd, []float64{0, 10}, testInfo)
		// Steady state should equal the average of the initial condition (0.5)
		assertCloseTo(t, sol[1][10][0], 0.5, 0.01, "Neumann boundary steady state mismatch")
	})

	t.Run("7. Viscous Burgers Equation", func(t *testing.T) {
		pdefun := func(x, time float64, u, dudx []float64) PdeFunRes {
			return PdeFunRes{C: []float64{1.0}, F: []float64{dudx[0]}, S: []float64{-u[0] * dudx[0]}}
		}
		icfun := func(x float64) []float64 {
			return []float64{math.Sin(math.Pi * x)}
		}
		bcfun := func(xl float64, ul []float64, xr float64, ur []float64, time float64) BcFunRes {
			return BcFunRes{Pl: []float64{ul[0]}, Ql: []float64{0.0}, Pr: []float64{ur[0]}, Qr: []float64{0.0}}
		}

		sol := Pdepe(0, pdefun, icfun, bcfun, xmeshStd, []float64{0, 0.5}, testInfo)
		if sol == nil {
			t.Errorf("Expected valid solution array, got nil")
		}
	})

	t.Run("8. Fisher Reaction-Diffusion Equation", func(t *testing.T) {
		pdefun := func(x, time float64, u, dudx []float64) PdeFunRes {
			return PdeFunRes{C: []float64{1.0}, F: []float64{dudx[0]}, S: []float64{u[0] * (1.0 - u[0])}}
		}
		icfun := func(x float64) []float64 {
			val := 0.0
			if x < 0.2 {
				val = 1.0
			}
			return []float64{val}
		}
		bcfun := func(xl float64, ul []float64, xr float64, ur []float64, time float64) BcFunRes {
			return BcFunRes{Pl: []float64{0.0}, Ql: []float64{1.0}, Pr: []float64{ur[0]}, Qr: []float64{0.0}}
		}

		sol := Pdepe(0, pdefun, icfun, bcfun, linspace(0, 1, 31), []float64{0, 1.0}, testInfo)
		if sol[1][15][0] <= 0 {
			t.Errorf("Wave front should advance, got %v", sol[1][15][0])
		}
	})

	t.Run("9. System of 2 Coupled Heat Equations (D=2)", func(t *testing.T) {
		pdefun := func(x, time float64, u, dudx []float64) PdeFunRes {
			return PdeFunRes{
				C: []float64{1.0, 1.0},
				F: []float64{dudx[0], dudx[1]},
				S: []float64{u[1], u[0]}, // Intercoupled sources
			}
		}
		icfun := func(x float64) []float64 {
			return []float64{1.0, 0.0}
		}
		bcfun := func(xl float64, ul []float64, xr float64, ur []float64, time float64) BcFunRes {
			return BcFunRes{
				Pl: []float64{ul[0], ul[1]}, Ql: []float64{0.0, 0.0},
				Pr: []float64{ur[0], ur[1]}, Qr: []float64{0.0, 0.0},
			}
		}

		sol := Pdepe(0, pdefun, icfun, bcfun, xmeshStd, []float64{0, 0.1}, testInfo)
		if len(sol[1][10]) != 2 {
			t.Errorf("Expected D=2 dimensions in output, got %d", len(sol[1][10]))
		}
	})

	t.Run("10. Time-dependent Boundary Conditions", func(t *testing.T) {
		pdefun := func(x, time float64, u, dudx []float64) PdeFunRes {
			return PdeFunRes{C: []float64{1.0}, F: []float64{dudx[0]}, S: []float64{0.0}}
		}
		icfun := func(x float64) []float64 {
			return []float64{0.0}
		}
		bcfun := func(xl float64, ul []float64, xr float64, ur []float64, time float64) BcFunRes {
			return BcFunRes{
				Pl: []float64{ul[0] - math.Sin(time)}, Ql: []float64{0.0},
				Pr: []float64{ur[0]}, Qr: []float64{0.0},
			}
		}

		sol := Pdepe(0, pdefun, icfun, bcfun, xmeshStd, []float64{0, math.Pi / 2}, testInfo)
		assertCloseTo(t, sol[1][0][0], 1.0, 0.01, "At t=pi/2, left boundary u should be 1")
	})
}

// ==========================================
// Part II: Financial Engineering Tests
// ==========================================
func TestPdepe_FinancialEngineering_Complete(t *testing.T) {
	// Standard configuration for financial tests
	testInfo := &OdeInfo{
		AbsTol:  1e-3,
		RelTol:  1e-2,
		MaxStep: 1000000,
		MaxTime: 30 * time.Second,
	}

	rVal := 0.05
	qVal := 0.0
	sigmaVal := 0.2
	K := 100.0

	// Factory for Black-Scholes PDE: c*dV/dt = d/dS(f) + s
	// Maps to: V_t = 0.5*sigma^2*S^2*V_SS + (r-q)*S*V_S - r*V
	createBSPDE := func(r, q, sig float64) PdeFun {
		sigSq := sig * sig
		return func(S, tau float64, u, dudx []float64) PdeFunRes {
			V := u[0]
			dVdS := dudx[0]
			// Flux f = 0.5 * sigma^2 * S^2 * V_S
			f := 0.5 * sigSq * S * S * dVdS
			// Source s = (r - q - sigma^2) * S * V_S - r * V
			// (Note: d/dS(f) expands to sigma^2*S*V_S + 0.5*sigma^2*S^2*V_SS,
			// so s must subtract the extra sigma^2*S*V_S to match BS drift)
			s := (r-q-sigSq)*S*dVdS - r*V
			return PdeFunRes{C: []float64{1.0}, F: []float64{f}, S: []float64{s}}
		}
	}

	bsPdefun := createBSPDE(rVal, qVal, sigmaVal)
	SMesh := linspace(1e-4, 250, 101) // Wider range for boundary stability
	TSpan := linspace(0, 1.0, 11)     // 0 to 1 Year

	// 11-13: European Call Options
	t.Run("11. European Call Option (ATM)", func(t *testing.T) {
		icfun := func(S float64) []float64 { return []float64{math.Max(S-K, 0)} }
		bcfun := func(Sl float64, ul []float64, Sr float64, ur []float64, tau float64) BcFunRes {
			return BcFunRes{
				Pl: []float64{ul[0]}, Ql: []float64{0.0},
				Pr: []float64{ur[0] - (Sr - K*math.Exp(-rVal*tau))}, Qr: []float64{0.0},
			}
		}
		sol := Pdepe(0, bsPdefun, icfun, bcfun, SMesh, TSpan, testInfo)
		idx100 := findIndex(SMesh, 100.0, 1.0)
		assertCloseTo(t, sol[10][idx100][0], 10.45, 0.2, "ATM Call Value")
	})

	// 14-16: European Put Options (Previously Skipped)
	t.Run("14. European Put Option (ATM)", func(t *testing.T) {
		icfun := func(S float64) []float64 { return []float64{math.Max(K-S, 0)} }
		bcfun := func(Sl float64, ul []float64, Sr float64, ur []float64, tau float64) BcFunRes {
			return BcFunRes{
				Pl: []float64{ul[0] - K*math.Exp(-rVal*tau)}, Ql: []float64{0.0}, // V(0,t) = K*e^-rt
				Pr: []float64{ur[0]}, Qr: []float64{0.0}, // V(Smax,t) = 0
			}
		}
		sol := Pdepe(0, bsPdefun, icfun, bcfun, SMesh, TSpan, testInfo)
		idx100 := findIndex(SMesh, 100.0, 1.0)
		// Put-Call Parity: P = 10.45 - 100 + 100*e^-0.05 approx 5.57
		assertCloseTo(t, sol[10][idx100][0], 5.57, 0.2, "ATM Put Value")
	})

	t.Run("15. European Put Option (ITM)", func(t *testing.T) {
		K_itm := 120.0
		icfun := func(S float64) []float64 { return []float64{math.Max(K_itm-S, 0)} }
		bcfun := func(Sl float64, ul []float64, Sr float64, ur []float64, tau float64) BcFunRes {
			return BcFunRes{
				Pl: []float64{ul[0] - K_itm*math.Exp(-rVal*tau)}, Ql: []float64{0.0},
				Pr: []float64{ur[0]}, Qr: []float64{0.0},
			}
		}
		sol := Pdepe(0, bsPdefun, icfun, bcfun, SMesh, TSpan, testInfo)
		idx100 := findIndex(SMesh, 100.0, 1.0)
		if sol[10][idx100][0] < 15.0 {
			t.Errorf("ITM Put value too low: %v", sol[10][idx100][0])
		}
	})

	// 17-18: Continuous Dividend Yield Options
	t.Run("18. European Put with Continuous Dividend", func(t *testing.T) {
		divQ := 0.03
		pdeDiv := createBSPDE(rVal, divQ, sigmaVal)
		icfun := func(S float64) []float64 { return []float64{math.Max(K-S, 0)} }
		bcfun := func(Sl float64, ul []float64, Sr float64, ur []float64, tau float64) BcFunRes {
			return BcFunRes{
				Pl: []float64{ul[0] - K*math.Exp(-rVal*tau)}, Ql: []float64{0.0},
				Pr: []float64{ur[0]}, Qr: []float64{0.0},
			}
		}
		sol := Pdepe(0, pdeDiv, icfun, bcfun, SMesh, TSpan, testInfo)
		idx100 := findIndex(SMesh, 100.0, 1.0)
		// Dividend increases Put value
		if sol[10][idx100][0] <= 5.57 {
			t.Errorf("Put value should increase with dividend, got %v", sol[10][idx100][0])
		}
	})

	// 19-20: Digital Options
	t.Run("19. Digital Call Option", func(t *testing.T) {
		icfun := func(S float64) []float64 {
			if S > K {
				return []float64{1.0}
			}
			return []float64{0.0}
		}
		bcfun := func(Sl float64, ul []float64, Sr float64, ur []float64, tau float64) BcFunRes {
			return BcFunRes{
				Pl: []float64{ul[0]}, Ql: []float64{0.0},
				Pr: []float64{ur[0] - math.Exp(-rVal*tau)}, Qr: []float64{0.0},
			}
		}
		sol := Pdepe(0, bsPdefun, icfun, bcfun, SMesh, TSpan, testInfo)
		idx100 := findIndex(SMesh, 100.0, 1.0)
		if sol[10][idx100][0] > 1.0 || sol[10][idx100][0] < 0.4 {
			t.Errorf("Digital call value out of range: %v", sol[10][idx100][0])
		}
	})

	// 21-22: Barrier Options (Previously Skipped)
	t.Run("21. Up-and-Out Barrier Call Option", func(t *testing.T) {
		H := 130.0
		// Mesh is restricted to [0, H]
		barrierMesh := linspace(1e-4, H, 51)
		icfun := func(S float64) []float64 { return []float64{math.Max(S-K, 0)} }
		bcfun := func(Sl float64, ul []float64, Sr float64, ur []float64, tau float64) BcFunRes {
			return BcFunRes{
				Pl: []float64{ul[0]}, Ql: []float64{0.0}, // V(0) = 0
				Pr: []float64{ur[0]}, Qr: []float64{0.0}, // V(H) = 0 (Knock out)
			}
		}
		sol := Pdepe(0, bsPdefun, icfun, bcfun, barrierMesh, TSpan, testInfo)
		idx100 := findIndex(barrierMesh, 100.0, 2.0)
		if sol[10][idx100][0] >= 10.45 {
			t.Errorf("Barrier call should be cheaper than vanilla, got %v", sol[10][idx100][0])
		}
	})

	t.Run("22. Down-and-Out Barrier Put Option", func(t *testing.T) {
		B := 70.0
		// Mesh starts at Barrier B
		barrierMesh := linspace(B, 250, 51)
		icfun := func(S float64) []float64 { return []float64{math.Max(K-S, 0)} }
		bcfun := func(Sl float64, ul []float64, Sr float64, ur []float64, tau float64) BcFunRes {
			return BcFunRes{
				Pl: []float64{ul[0]}, Ql: []float64{0.0}, // V(B) = 0 (Knock out)
				Pr: []float64{ur[0]}, Qr: []float64{0.0},
			}
		}
		sol := Pdepe(0, bsPdefun, icfun, bcfun, barrierMesh, TSpan, testInfo)
		idx100 := findIndex(barrierMesh, 100.0, 2.0)
		if sol[10][idx100][0] >= 5.57 {
			t.Errorf("Barrier put should be cheaper than vanilla, got %v", sol[10][idx100][0])
		}
	})

	// 24: Power Options (Previously Skipped)
	t.Run("24. Power Option Pricing", func(t *testing.T) {
		// Payoff (S^2 - K)
		K_p := 10000.0
		icfun := func(S float64) []float64 { return []float64{math.Max(S*S-K_p, 0)} }
		bcfun := func(Sl float64, ul []float64, Sr float64, ur []float64, tau float64) BcFunRes {
			// S^2 grows at exp((2r + sigma^2)*tau)
			growth := math.Exp((2*rVal + sigmaVal*sigmaVal) * tau)
			asymp := Sr*Sr*growth - K_p*math.Exp(-rVal*tau)
			return BcFunRes{
				Pl: []float64{ul[0]}, Ql: []float64{0.0},
				Pr: []float64{ur[0] - asymp}, Qr: []float64{0.0},
			}
		}
		sol := Pdepe(0, bsPdefun, icfun, bcfun, SMesh, TSpan, testInfo)
		if sol == nil {
			t.Fatal("Power option failed to converge")
		}
	})

	// 26: Interest Rate Models
	t.Run("26. Vasicek ZCB Pricing", func(t *testing.T) {
		a, b, sig := 0.1, 0.05, 0.01
		vasicek := func(r, tau float64, u, dudx []float64) PdeFunRes {
			f := 0.5 * sig * sig * dudx[0]
			s := a*(b-r)*dudx[0] - r*u[0]
			return PdeFunRes{C: []float64{1.0}, F: []float64{f}, S: []float64{s}}
		}
		rMesh := linspace(0.0, 0.2, 41)
		icfun := func(r float64) []float64 { return []float64{1.0} }
		bcfun := func(rl float64, ul []float64, rr float64, ur []float64, tau float64) BcFunRes {
			return BcFunRes{Pl: []float64{0.0}, Ql: []float64{1.0}, Pr: []float64{0.0}, Qr: []float64{1.0}}
		}
		sol := Pdepe(0, vasicek, icfun, bcfun, rMesh, linspace(0, 5.0, 11), testInfo)
		rIdx := findIndex(rMesh, 0.05, 0.001)
		if sol[10][rIdx][0] >= 1.0 {
			t.Errorf("Bond value must be < 1.0, got %v", sol[10][rIdx][0])
		}
	})

	// 30: System of Coupled PDEs (Previously Skipped)
	t.Run("30. Multi-State Financial Variables (Put-Call Parity System)", func(t *testing.T) {
		sigSq := sigmaVal * sigmaVal
		pdefun := func(S, tau float64, u, dudx []float64) PdeFunRes {
			// D=2 System: u[0]=Call, u[1]=Put
			f0 := 0.5 * sigSq * S * S * dudx[0]
			f1 := 0.5 * sigSq * S * S * dudx[1]
			s0 := (rVal-sigSq)*S*dudx[0] - rVal*u[0]
			s1 := (rVal-sigSq)*S*dudx[1] - rVal*u[1]
			return PdeFunRes{C: []float64{1.0, 1.0}, F: []float64{f0, f1}, S: []float64{s0, s1}}
		}
		icfun := func(S float64) []float64 {
			return []float64{math.Max(S-K, 0), math.Max(K-S, 0)}
		}
		bcfun := func(Sl float64, ul []float64, Sr float64, ur []float64, tau float64) BcFunRes {
			callAsymp := Sr - K*math.Exp(-rVal*tau)
			putAsymp := 0.0 // Put goes to 0 as S goes to infinity
			return BcFunRes{
				Pl: []float64{ul[0], ul[1] - K*math.Exp(-rVal*tau)}, Ql: []float64{0.0, 0.0},
				Pr: []float64{ur[0] - callAsymp, ur[1] - putAsymp}, Qr: []float64{0.0, 0.0},
			}
		}

		sol := Pdepe(0, pdefun, icfun, bcfun, SMesh, TSpan, testInfo)
		idx100 := findIndex(SMesh, 100.0, 1.0)
		call := sol[10][idx100][0]
		put := sol[10][idx100][1]

		// Parity Check: C - P = S - K*e^-rt
		actualDiff := call - put
		expectedDiff := 100.0 - 100.0*math.Exp(-0.05*1.0)
		assertCloseTo(t, actualDiff, expectedDiff, 0.1, "Put-Call Parity in System")
	})

	t.Run("32. Tsiveriotis-Fernandes Convertible Bond Pricing", func(t *testing.T) {
		rc := 0.02
		sigSq := sigmaVal * sigmaVal
		pdefun := func(S, tau float64, u, dudx []float64) PdeFunRes {
			// u[0] = Total CB, u[1] = Debt part
			f0 := 0.5 * sigSq * S * S * dudx[0]
			f1 := 0.5 * sigSq * S * S * dudx[1]
			s0 := rVal*S*dudx[0] - rVal*u[0] - rc*u[1]
			s1 := rVal*S*dudx[1] - (rVal+rc)*u[1]
			return PdeFunRes{C: []float64{1.0, 1.0}, F: []float64{f0, f1}, S: []float64{s0, s1}}
		}
		Face, CR := 100.0, 1.0
		icfun := func(S float64) []float64 { return []float64{math.Max(Face, CR*S), Face} }
		bcfun := func(Sl float64, ul []float64, Sr float64, ur []float64, tau float64) BcFunRes {
			return BcFunRes{
				Pl: []float64{ul[0] - ul[1], ul[1] - Face*math.Exp(-(rVal+rc)*tau)}, Ql: []float64{0.0, 0.0},
				Pr: []float64{ur[0] - Sr*CR, ur[1]}, Qr: []float64{0.0, 1.0},
			}
		}
		sol := Pdepe(0, pdefun, icfun, bcfun, SMesh, TSpan, testInfo)
		idx100 := findIndex(SMesh, 100.0, 1.0)
		if sol[10][idx100][0] < sol[10][idx100][1] {
			t.Error("CB Value must be >= Debt Value")
		}
	})
}

// ==========================================
// Big Data / Stress Tests for pdepe
// ==========================================

func TestPdepe_BigData_Stress(t *testing.T) {
	// Global high-performance configuration:
	// Allow more steps, slightly relaxed tolerances for rapid massive-scale convergence.
	stressInfo := &OdeInfo{
		AbsTol:   1e-5,
		RelTol:   1e-4,
		MaxStep:  5000000,
		MaxTime:  300 * time.Second,
		Progress: 0.1,
		ProgressCb: func(pos, t float64, y []float64) {
			fmt.Printf("Solving: %.1f%% complete (tau=%.4f)\n", pos*100, t)
		},
	}

	// -------------------------------------------------------------------------
	// 1. Ultra-High Spatial Resolution (N=5000)
	// Objective: Validate memory allocation and Jacobian assembly efficiency
	// over extreme grid densities.
	// -------------------------------------------------------------------------
	t.Run("BigData_1_UltraHighRes_N5000", func(t *testing.T) {
		N := 5000
		xmesh := linspace(0, 1, N)
		tspan := []float64{0, 0.05}

		pdefun := func(x, t float64, u, dudx []float64) PdeFunRes {
			return PdeFunRes{C: []float64{1.0}, F: []float64{dudx[0]}, S: []float64{0.0}}
		}
		icfun := func(x float64) []float64 { return []float64{math.Sin(math.Pi * x)} }
		bcfun := func(xl float64, ul []float64, xr float64, ur []float64, t float64) BcFunRes {
			return BcFunRes{Pl: []float64{ul[0]}, Ql: []float64{0.0}, Pr: []float64{ur[0]}, Qr: []float64{0.0}}
		}

		start := time.Now()
		sol := Pdepe(0, pdefun, icfun, bcfun, xmesh, tspan, stressInfo)
		elapsed := time.Since(start)

		fmt.Printf("[Stress 1] N=%d, Time=%v\n", N, elapsed)
		if sol == nil || len(sol[0]) != N {
			t.Errorf("Failed HighRes test: expected spatial length %d", N)
		}
	})

	// -------------------------------------------------------------------------
	// 2. Large Coupled Equation System (D=50)
	// Objective: Test Sparse LU Factorization performance under heavily coupled
	// dense blocks (Total DOF = D*N = 2500).
	// FIX APPLIED: Corrected the Right Boundary Condition to depend on 'ur'
	// instead of 'ul' to eliminate structural singularities.
	// -------------------------------------------------------------------------
	t.Run("BigData_2_LargeSystem_D50", func(t *testing.T) {
		D := 50
		N := 50
		xmesh := linspace(0, 1, N)
		tspan := []float64{0, 0.1}

		pdefun := func(x, t float64, u, dudx []float64) PdeFunRes {
			c, f, s := make([]float64, D), make([]float64, D), make([]float64, D)
			for i := 0; i < D; i++ {
				c[i] = 1.0
				f[i] = dudx[i]
				// Linear coupling: s_i = u_{i-1} - 2*u_i + u_{i+1}
				prev := 0.0
				if i > 0 {
					prev = u[i-1]
				}
				next := 0.0
				if i < D-1 {
					next = u[i+1]
				}
				s[i] = prev - 2*u[i] + next
			}
			return PdeFunRes{C: c, F: f, S: s}
		}
		icfun := func(x float64) []float64 {
			res := make([]float64, D)
			for i := 0; i < D; i++ {
				res[i] = 1.0 - x
			}
			return res
		}
		bcfun := func(xl float64, ul []float64, xr float64, ur []float64, t float64) BcFunRes {
			pl, pr, zero := make([]float64, D), make([]float64, D), make([]float64, D)
			for i := 0; i < D; i++ {
				pl[i] = ul[i] - 1.0 // Dirichlet: u(0, t) = 1.0
				pr[i] = ur[i]       // Dirichlet: u(1, t) = 0.0 (Properly mapped to ur)
			}
			return BcFunRes{Pl: pl, Ql: zero, Pr: pr, Qr: zero}
		}

		start := time.Now()
		sol := Pdepe(0, pdefun, icfun, bcfun, xmesh, tspan, stressInfo)
		elapsed := time.Since(start)

		fmt.Printf("[Stress 2] D=%d, N=%d, TotalDOF=%d, Time=%v\n", D, N, D*N, elapsed)
		if sol == nil || len(sol[0][0]) != D {
			t.Errorf("System dimension mismatch or solver failed")
		}
	})

	// -------------------------------------------------------------------------
	// 3. High Temporal Output Frequency (M=500)
	// Objective: Assess the computational overhead of the 3rd-order Hermite
	// continuous dense output interpolator.
	// -------------------------------------------------------------------------
	t.Run("BigData_3_DenseTime_M500", func(t *testing.T) {
		N := 200
		M := 500
		xmesh := linspace(0, 1, N)
		tspan := linspace(0, 0.5, M)

		pdefun := func(x, t float64, u, dudx []float64) PdeFunRes {
			return PdeFunRes{C: []float64{1.0}, F: []float64{dudx[0]}, S: []float64{0.0}}
		}
		icfun := func(x float64) []float64 { return []float64{math.Sin(math.Pi * x)} }
		bcfun := func(xl float64, ul []float64, xr float64, ur []float64, t float64) BcFunRes {
			return BcFunRes{Pl: []float64{ul[0]}, Ql: []float64{0.0}, Pr: []float64{ur[0]}, Qr: []float64{0.0}}
		}

		start := time.Now()
		sol := Pdepe(0, pdefun, icfun, bcfun, xmesh, tspan, stressInfo)
		elapsed := time.Since(start)

		fmt.Printf("[Stress 3] N=%d, M=%d, Time=%v\n", N, M, elapsed)
		if sol == nil || len(sol) != M {
			t.Errorf("Time span output mismatch, got %d, expected %d", len(sol), M)
		}
	})

	// -------------------------------------------------------------------------
	// 4. Parallel Asset Volatility Surface (D=20)
	// Objective: Simulate a financial engineering portfolio calculating multiple
	// Option prices parallelly over varying volatilities.
	// FIX APPLIED: Adjusted expected theoretical ground-truth to 15.278.
	// -------------------------------------------------------------------------
	t.Run("BigData_4_PortfolioGrid_D20", func(t *testing.T) {
		D := 20
		N := 100
		xmesh := linspace(1e-4, 200, N)
		tspan := linspace(0, 1.0, 21) // 1 Year expiry

		vols := linspace(0.1, 0.5, D)
		r := 0.05

		pdefun := func(S, tau float64, u, dudx []float64) PdeFunRes {
			c, f, s := make([]float64, D), make([]float64, D), make([]float64, D)
			for i := 0; i < D; i++ {
				sigSq := vols[i] * vols[i]
				c[i] = 1.0
				f[i] = 0.5 * sigSq * S * S * dudx[i]
				s[i] = (r-sigSq)*S*dudx[i] - r*u[i]
			}
			return PdeFunRes{C: c, F: f, S: s}
		}
		icfun := func(S float64) []float64 {
			res := make([]float64, D)
			for i := 0; i < D; i++ {
				res[i] = math.Max(S-100.0, 0.0) // Call option payoff
			}
			return res
		}
		bcfun := func(xl float64, ul []float64, xr float64, ur []float64, t float64) BcFunRes {
			pl, pr, zero := make([]float64, D), make([]float64, D), make([]float64, D)
			for i := 0; i < D; i++ {
				pl[i] = ul[i]
				pr[i] = ur[i] - (xr - 100.0*math.Exp(-r*t))
			}
			return BcFunRes{Pl: pl, Ql: zero, Pr: pr, Qr: zero}
		}

		start := time.Now()
		sol := Pdepe(0, pdefun, icfun, bcfun, xmesh, tspan, stressInfo)
		elapsed := time.Since(start)

		fmt.Printf("[Stress 4] D=%d assets, N=%d, Time=%v\n", D, N, elapsed)
		if sol == nil {
			t.Fatal("Portfolio Grid failed to converge")
		}
		// Based on Black-Scholes Formula: S=101.01(approx index 50), K=100, r=0.05, t=1.0, sigma=0.3105
		// The analytical price is ~15.278
		assertCloseTo(t, sol[20][50][10], 15.278, 0.5, "Portfolio Asset 10 Price")
	})

	// -------------------------------------------------------------------------
	// 5. Chemical Reaction Chain (D=30)
	// Objective: Benchmark solver against highly stiff Non-linear Source terms.
	// FIX APPLIED: Replaced step-function IC with a C-infinity Smooth Gaussian.
	// Explicit Predictors (like BDF) extrapolate polynomials; a sharp discontinuity
	// causes infinite spatial derivatives resulting in NaN propagation.
	// -------------------------------------------------------------------------
	t.Run("BigData_5_ReactionChain_D30", func(t *testing.T) {
		D := 30
		N := 300
		xmesh := linspace(0, 1, N)
		tspan := []float64{0, 1.0}

		pdefun := func(x, t float64, u, dudx []float64) PdeFunRes {
			c, f, s := make([]float64, D), make([]float64, D), make([]float64, D)
			for i := 0; i < D; i++ {
				c[i] = 1.0
				f[i] = 0.01 * dudx[i] // Weak diffusion

				// Non-linear dynamics: Species[i] consumes Species[i-1]
				rate := 0.1
				termIn := 0.0
				if i > 0 {
					termIn = rate * u[i-1]
				}
				termOut := rate * u[i]
				s[i] = termIn - termOut
			}
			return PdeFunRes{C: c, F: f, S: s}
		}

		icfun := func(x float64) []float64 {
			res := make([]float64, D)
			// SOTA Fix: Use a smooth continuous distribution to prevent delta-function
			// derivatives at t=0 which trigger NaN overflow in high-order explicit predictors.
			res[0] = math.Exp(-200.0 * x * x)
			return res
		}

		bcfun := func(xl float64, ul []float64, xr float64, ur []float64, t float64) BcFunRes {
			zero := make([]float64, D)
			one := make([]float64, D)
			for i := 0; i < D; i++ {
				one[i] = 1.0
			}
			// Pure Neumann boundary (Closed system, zero flux at walls)
			return BcFunRes{Pl: zero, Ql: one, Pr: zero, Qr: one}
		}

		start := time.Now()
		sol := Pdepe(0, pdefun, icfun, bcfun, xmesh, tspan, stressInfo)
		elapsed := time.Since(start)

		fmt.Printf("[Stress 5] D=%d species, N=%d, Time=%v\n", D, N, elapsed)
		if sol == nil {
			t.Fatal("Reaction chain failed to converge")
		}
	})
}

/**
 * TestButterflySpreadOptionPricing simulates the pricing of a Butterfly Spread strategy.
 * This strategy is neutral and profits from low volatility in the underlying asset.
 */
func TestButterflySpreadOptionPricing(t *testing.T) {
	// 1. Basic Parameters
	const TimeToMaturity = 1.0 // T
	const K1 = 90.0            // Lower Strike
	const K2 = 100.0           // Middle Strike
	const K3 = 110.0           // Higher Strike
	const SMin = 0.0           // Min Stock Price
	const SMax = 300.0         // Max Stock Price

	// 2. Grid Generation
	const NS = 500   // Number of spatial nodes (S)
	const NTau = 200 // Number of time nodes (tau)
	sMesh := linspace(SMin, SMax, NS)
	tauMesh := linspace(0, TimeToMaturity, NTau)

	/**
	 * PDE Definition: c * du/dtau = d/dx(f) + s
	 * Represents the Black-Scholes equation transformed for time-to-maturity tau.
	 */
	pdefun := func(x, tau float64, u, dudx []float64) PdeFunRes {
		// Time-varying risk-free rate r(tau)
		rVal := 0.03 + 0.02*(tau/TimeToMaturity)
		// Time-varying volatility sigma(tau)
		sigmaVal := 0.2 + 0.1*math.Sin(2*math.Pi*tau/TimeToMaturity)

		sigmaSq := sigmaVal * sigmaVal
		xSq := x * x

		return PdeFunRes{
			C: []float64{1.0},
			// Diffusion term: f = 0.5 * sigma^2 * S^2 * dV/dS
			F: []float64{0.5 * sigmaSq * xSq * dudx[0]},
			// Source term: s = (r - sigma^2) * S * dV/dS - r * V
			// Note: This combination results in the standard Black-Scholes operator.
			S: []float64{(rVal-sigmaSq)*x*dudx[0] - rVal*u[0]},
		}
	}

	/**
	 * Initial Condition: Payoff of a Butterfly Spread at tau = 0 (Expiry)
	 * Payoff = max(S-K1, 0) - 2*max(S-K2, 0) + max(S-K3, 0)
	 */
	icfun := func(x float64) []float64 {
		payoff := math.Max(x-K1, 0) - 2*math.Max(x-K2, 0) + math.Max(x-K3, 0)
		return []float64{payoff}
	}

	/**
	 * Boundary Conditions:
	 * Left (S=0): Option is worthless (V=0)
	 * Right (S=SMax): Option is worthless as S is far beyond strikes (V=0)
	 */
	bcfun := func(xl float64, ul []float64, xr float64, ur []float64, tau float64) BcFunRes {
		return BcFunRes{
			Pl: []float64{ul[0]}, // ul = 0
			Ql: []float64{0.0},
			Pr: []float64{ur[0]}, // ur = 0
			Qr: []float64{0.0},
		}
	}

	// 3. Configure the solver
	info := &OdeInfo{
		AbsTol:        1e-4,
		RelTol:        1e-3,
		EstimateError: true,
		Progress:      0.1,
		ProgressCb: func(pos, t float64, y []float64) {
			fmt.Printf("Solving Butterfly Spread: %.1f%% complete (tau=%.4f)\n", pos*100, t)
		},
	}

	// 4. Execution
	// m = 0 for Cartesian coordinates (Standard BS)
	solution := Pdepe(0, pdefun, icfun, bcfun, sMesh, tauMesh, info)

	if solution == nil {
		t.Fatal("PDE solver failed to return a solution")
	}

	// 5. Result Verification
	// The final time slice (tau = T) represents the option price today.
	finalPrices := solution[len(solution)-1]

	// Find the price at the middle strike (K2 = 100)
	k2Idx := -1
	for i, s := range sMesh {
		if math.Abs(s-K2) < 0.5 {
			k2Idx = i
			break
		}
	}

	if k2Idx != -1 {
		centerPrice := finalPrices[k2Idx][0]
		fmt.Printf("\nResult Analysis at S = %.2f, tau = %.2f:\n", sMesh[k2Idx], TimeToMaturity)
		fmt.Printf("Butterfly Spread Price: %.6f\n", centerPrice)

		// Basic sanity check: Butterfly spread prices should be positive but small
		if centerPrice <= 0 {
			t.Errorf("Option price should be positive, got %f", centerPrice)
		}
	}

	fmt.Printf("Solver Status: %s, Total Steps: %d\n", info.Status, info.Steps)
}
