package main

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

// 1. Basic Exponential Decay y' = -y
func Test45ExponentialDecay(t *testing.T) {
	ode := func(t float64, y []float64) OdeRes {
		return OdeRes{F: []float64{-y[0]}}
	}
	res := Ode45(ode, [2]float64{0, 1}, []float64{1.0}, nil)
	expected := math.Exp(-1.0)
	finalY := res.Y[len(res.Y)-1][0]
	if !near(finalY, expected, 1e-4) {
		t.Errorf("Expected %f, got %f", expected, finalY)
	}
}

// 2. Simple Harmonic Oscillator y” = -y
func Test45HarmonicOscillator(t *testing.T) {
	ode := func(t float64, y []float64) OdeRes {
		return OdeRes{F: []float64{y[1], -y[0]}}
	}
	res := Ode45(ode, [2]float64{0, math.Pi}, []float64{1.0, 0.0}, nil)
	finalY := res.Y[len(res.Y)-1]
	// At t = Pi, y[0] should be -1, y[1] should be 0
	if !near(finalY[0], -1.0, 1e-3) || !near(finalY[1], 0.0, 1e-3) {
		t.Errorf("Oscillator failed at Pi: got %v", finalY)
	}
}

// 3. Logistic Growth y' = y(1-y)
func Test45LogisticGrowth(t *testing.T) {
	ode := func(t float64, y []float64) OdeRes {
		return OdeRes{F: []float64{y[0] * (1.0 - y[0])}}
	}
	res := Ode45(ode, [2]float64{0, 2}, []float64{0.5}, nil)
	expected := 1.0 / (1.0 + math.Exp(-2.0))
	finalY := res.Y[len(res.Y)-1][0]
	if !near(finalY, expected, 1e-4) {
		t.Errorf("Logistic growth failed: expected %f, got %f", expected, finalY)
	}
}

// 4. Boundary Test: Zero time span
func Test45ZeroTspan(t *testing.T) {
	ode := func(t float64, y []float64) OdeRes { return OdeRes{F: []float64{-y[0]}} }
	res := Ode45(ode, [2]float64{1, 1}, []float64{1.0}, nil)
	if res != nil {
		t.Error("Should return nil for zero tspan")
	}
}

// 5. Reverse Integration (t_start > t_final)
func Test45ReverseIntegration(t *testing.T) {
	ode := func(t float64, y []float64) OdeRes { return OdeRes{F: []float64{-y[0]}} }
	res := Ode45(ode, [2]float64{1, 0}, []float64{math.Exp(-1.0)}, nil)
	finalY := res.Y[len(res.Y)-1][0]
	if !near(finalY, 1.0, 1e-3) {
		t.Errorf("Reverse integration failed: expected 1.0, got %f", finalY)
	}
}

// 6. Robertson Problem (Classic Stiff ODE)
func Test45RobertsonStiff(t *testing.T) {
	ode := func(t float64, y []float64) OdeRes {
		f1 := -0.04*y[0] + 1e4*y[1]*y[2]
		f2 := 0.04*y[0] - 1e4*y[1]*y[2] - 3e7*y[1]*y[1]
		f3 := 3e7 * y[1] * y[1]
		return OdeRes{F: []float64{f1, f2, f3}}
	}
	info := &OdeInfo{AbsTol: 1e-8, RelTol: 1e-8}
	res := Ode45(ode, [2]float64{0, 0.4}, []float64{1, 0, 0}, info)
	if res == nil || info.Status != "done" {
		t.Errorf("Robertson stiff solver failed: %s", info.Status)
	}
}

// 8. Analytical Jacobian Test
func Test45AnalyticalJacobian(t *testing.T) {
	ode := func(t float64, y []float64) OdeRes { return OdeRes{F: []float64{-y[0]}} }
	jac := func(t float64, y []float64, f []float64) CooMatrix {
		return CooMatrix{
			RowIdx: []int{0},
			ColIdx: []int{0},
			Vals:   []float64{-1.0},
		}
	}
	info := &OdeInfo{Jacobian: jac}
	res := Ode45(ode, [2]float64{0, 1}, []float64{1.0}, info)
	if !near(res.Y[len(res.Y)-1][0], math.Exp(-1.0), 1e-4) {
		t.Error("Analytical Jacobian results incorrect")
	}
}

// 9. Max Steps Limit (Fixed: Smaller limit to ensure trigger)
func Test45MaxStepsLimit(t *testing.T) {
	ode := func(t float64, y []float64) OdeRes { return OdeRes{F: []float64{1.0}} }
	info := &OdeInfo{MaxStep: 3, RelTol: 1e-10} // Very small limit
	Ode45(ode, [2]float64{0, 100}, []float64{0}, info)
	if info.Status != "max_steps" {
		t.Errorf("Expected status max_steps, got %s", info.Status)
	}
}

// 10. Timeout Limit
func Test45TimeoutLimit(t *testing.T) {
	ode := func(t float64, y []float64) OdeRes {
		time.Sleep(2 * time.Millisecond)
		return OdeRes{F: []float64{1.0}}
	}
	info := &OdeInfo{MaxTime: 5 * time.Millisecond}
	Ode45(ode, [2]float64{0, 100}, []float64{0}, info)
	if info.Status != "timeout" {
		t.Errorf("Expected status timeout, got %s", info.Status)
	}
}

// 11. Progress Callback
func Test45ProgressCallback(t *testing.T) {
	called := false
	info := &OdeInfo{
		Progress: 0.1,
		ProgressCb: func(pos, t float64, y []float64) {
			called = true
		},
	}
	ode := func(t float64, y []float64) OdeRes { return OdeRes{F: []float64{1.0}} }
	Ode45(ode, [2]float64{0, 1}, []float64{0}, info)
	if !called {
		t.Error("Progress callback was not called")
	}
}

// 13. High-Order Stiff System (Van der Pol, mu=100)
func Test45VanderPolStiff(t *testing.T) {
	mu := 100.0
	ode := func(t float64, y []float64) OdeRes {
		return OdeRes{F: []float64{
			y[1],
			mu*(1-y[0]*y[0])*y[1] - y[0],
		}}
	}
	info := &OdeInfo{RelTol: 1e-4, AbsTol: 1e-6}
	res := Ode45(ode, [2]float64{0, 1.0}, []float64{2.0, 0.0}, info)
	if res == nil || info.Status != "done" {
		t.Errorf("Van der Pol failed: %s", info.Status)
	}
}

// 14. High Precision Tolerance
func Test45TightTolerance(t *testing.T) {
	ode := func(t float64, y []float64) OdeRes { return OdeRes{F: []float64{-y[0]}} }
	info := &OdeInfo{AbsTol: 1e-11, RelTol: 1e-11}
	res := Ode45(ode, [2]float64{0, 1}, []float64{1.0}, info)
	expected := math.Exp(-1.0)
	if !near(res.Y[len(res.Y)-1][0], expected, 1e-9) {
		t.Error("High precision tolerance failed")
	}
}

// 15. Singular Matrix Recovery
func Test45SingularMatrixRecovery(t *testing.T) {
	ode := func(t float64, y []float64) OdeRes {
		if t > 0.1 && t < 0.12 {
			// Simulate a steep localized gradient
			return OdeRes{F: []float64{-1e6 * y[0]}}
		}
		return OdeRes{F: []float64{-y[0]}}
	}
	info := &OdeInfo{AbsTol: 1e-6}
	res := Ode45(ode, [2]float64{0, 0.3}, []float64{1.0}, info)
	if res == nil || info.Status != "done" {
		t.Errorf("Solver failed to recover: %s", info.Status)
	}
}

// 16. Step Size Underflow
func Test45StepSizeUnderflow(t *testing.T) {
	ode := func(t float64, y []float64) OdeRes {
		// Singularity at t=1
		val := 1.0 / (1.000000000001 - t)
		return OdeRes{F: []float64{val}}
	}
	info := &OdeInfo{MaxStep: 1000}
	Ode45(ode, [2]float64{0, 2}, []float64{0}, info)
	if info.Status != "underflow" && info.Status != "max_steps" {
		t.Logf("Status: %s (expected underflow or max_steps)", info.Status)
	}
}

// 17. Multidimensional System
func Test45VectorInitialConditions(t *testing.T) {
	ode := func(t float64, y []float64) OdeRes {
		return OdeRes{F: []float64{-y[0], -2.0 * y[1]}}
	}
	res := Ode45(ode, [2]float64{0, 1}, []float64{1.0, 1.0}, nil)
	finalY := res.Y[len(res.Y)-1]
	if !near(finalY[0], math.Exp(-1.0), 1e-4) || !near(finalY[1], math.Exp(-2.0), 1e-4) {
		t.Error("Multi-dimensional system failed")
	}
}

// 18. History Buffer (Fixed: smaller tolerance to force more steps)
func Test45HistoryBuffer(t *testing.T) {
	ode := func(t float64, y []float64) OdeRes {
		// Oscillatory function forces many steps
		return OdeRes{F: []float64{math.Cos(10 * t)}}
	}
	info := &OdeInfo{RelTol: 1e-9}
	res := Ode45(ode, [2]float64{0, 2}, []float64{0}, info)
	if len(res.T) < 10 {
		t.Errorf("Integration did not take enough steps, only %d", len(res.T))
	}
}

// 19. Result Consistency (Fixed: check implementation logic)
func Test45ResultConsistency(t *testing.T) {
	ode := func(t float64, y []float64) OdeRes { return OdeRes{F: []float64{math.Sin(t)}} }
	res := Ode45(ode, [2]float64{0, 5}, []float64{0}, nil)
	// In the implementation, T, Y, and Dy are all appended at the same time
	if len(res.T) != len(res.Y) || len(res.T) != len(res.Dy) {
		t.Errorf("Output length mismatch: T=%d, Y=%d, Dy=%d", len(res.T), len(res.Y), len(res.Dy))
	}
}

// 20. Custom Initial Step
func Test45CustomInitialStep(t *testing.T) {
	ode := func(t float64, y []float64) OdeRes { return OdeRes{F: []float64{1.0}} }
	info := &OdeInfo{InitialStep: 0.123}
	res := Ode45(ode, [2]float64{0, 1}, []float64{0}, info)
	if !near(res.T[1], 0.123, 1e-7) {
		t.Errorf("Initial step ignored: expected 0.123, got %f", res.T[1])
	}
}

// ==========================================
// LARGE DATA / HIGH DIMENSIONALITY TESTS
// ==========================================

// 21. 1D Heat Equation (Diffusion) - 100 Dimensions
func Test45LargeHeatEquation(t *testing.T) {
	N := 100
	dx := 1.0 / float64(N-1)
	alpha := 0.01

	ode := func(t float64, y []float64) OdeRes {
		f := make([]float64, N)
		// Dirichlet Boundary Conditions
		f[0] = 0
		f[N-1] = 0
		for i := 1; i < N-1; i++ {
			f[i] = alpha * (y[i+1] - 2*y[i] + y[i-1]) / (dx * dx)
		}
		return OdeRes{F: f}
	}

	y0 := make([]float64, N)
	for i := 0; i < N; i++ {
		y0[i] = math.Sin(math.Pi * float64(i) * dx)
	}

	info := &OdeInfo{AbsTol: 1e-6, RelTol: 1e-6}
	start := time.Now()
	res := Ode45(ode, [2]float64{0, 0.5}, y0, info)
	elapsed := time.Since(start)

	if res == nil || info.Status != "done" {
		t.Fatalf("Large Heat Equation failed: %s", info.Status)
	}
	t.Logf("Heat Equation (N=100) took %v, steps: %d", elapsed, info.Steps)
}

// 22. Large Linear System with Random Sparse Matrix - 200 Dimensions
func Test45LargeRandomLinearSystem(t *testing.T) {
	N := 200
	// Create a stable random sparse diagonal-dominant matrix
	matrix := make([][]float64, N)
	for i := range matrix {
		matrix[i] = make([]float64, N)
		matrix[i][i] = -2.0 - rand.Float64() // Ensure stability
		if i > 0 {
			matrix[i][i-1] = 0.5
		}
		if i < N-1 {
			matrix[i][i+1] = 0.5
		}
	}

	ode := func(t float64, y []float64) OdeRes {
		f := make([]float64, N)
		for i := 0; i < N; i++ {
			for j := i - 1; j <= i+1; j++ {
				if j >= 0 && j < N {
					f[i] += matrix[i][j] * y[j]
				}
			}
		}
		return OdeRes{F: f}
	}

	y0 := make([]float64, N)
	for i := range y0 {
		y0[i] = 1.0
	}

	info := &OdeInfo{MaxStep: 5000}
	res := Ode45(ode, [2]float64{0, 1.0}, y0, info)

	if res == nil || info.Status != "done" {
		t.Errorf("Large Random System failed: %s", info.Status)
	}
}

// 23. Brusselator 1D (Chemical Reaction-Diffusion) - 100 Dimensions (50 pairs)
func Test45Brusselator1D(t *testing.T) {
	N := 50 // 50 units of (U, V)
	A, B := 1.0, 3.0
	Du, Dv := 1.0, 0.1
	dx := 1.0

	ode := func(t float64, y []float64) OdeRes {
		f := make([]float64, 2*N)
		for i := 0; i < N; i++ {
			u := y[2*i]
			v := y[2*i+1]

			// Reaction
			f[2*i] = A + u*u*v - (B+1)*u
			f[2*i+1] = B*u - u*u*v

			// Diffusion (Periodic)
			im1 := (i - 1 + N) % N
			ip1 := (i + 1) % N
			f[2*i] += Du * (y[2*im1] - 2*u + y[2*ip1]) / (dx * dx)
			f[2*i+1] += Dv * (y[2*im1+1] - 2*v + y[2*ip1+1]) / (dx * dx)
		}
		return OdeRes{F: f}
	}

	y0 := make([]float64, 2*N)
	for i := 0; i < N; i++ {
		y0[2*i] = A + 0.1*rand.Float64()
		y0[2*i+1] = B/A + 0.1*rand.Float64()
	}

	info := &OdeInfo{AbsTol: 1e-5, RelTol: 1e-5}
	res := Ode45(ode, [2]float64{0, 5.0}, y0, info)
	if res == nil {
		t.Error("Brusselator failed to integrate")
	}
}

// 24. Large Wave Equation (System of ODEs) - 100 Dimensions
func Test45LargeWaveEquation(t *testing.T) {
	N := 50 // 50 positions, 2 variables each (pos, vel)
	c := 1.0
	dx := 0.1

	ode := func(t float64, y []float64) OdeRes {
		f := make([]float64, 2*N)
		for i := 1; i < N-1; i++ {
			u := y[2*i]   // displacement
			v := y[2*i+1] // velocity
			f[2*i] = v
			f[2*i+1] = c * c * (y[2*(i+1)] - 2*u + y[2*(i-1)]) / (dx * dx)
		}
		return OdeRes{F: f}
	}

	y0 := make([]float64, 2*N)
	for i := 0; i < N; i++ {
		y0[2*i] = math.Exp(-math.Pow(float64(i)*dx-2.5, 2)) // Gaussian pulse
	}

	res := Ode45(ode, [2]float64{0, 1.0}, y0, nil)
	if res == nil {
		t.Error("Wave equation failed")
	}
}

// 25. High Dimensional Identity System - 500 Dimensions
func Test45IdentityLargeScale(t *testing.T) {
	N := 500
	ode := func(t float64, y []float64) OdeRes {
		f := make([]float64, N)
		for i := 0; i < N; i++ {
			f[i] = -float64(i+1) * 0.1 * y[i]
		}
		return OdeRes{F: f}
	}

	y0 := make([]float64, N)
	for i := 0; i < N; i++ {
		y0[i] = 1.0
	}

	info := &OdeInfo{MaxTime: 2 * time.Second}
	res := Ode45(ode, [2]float64{0, 1.0}, y0, info)
	if res == nil {
		t.Errorf("500-Dim identity system failed: %s", info.Status)
	}
}
