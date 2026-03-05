package main

import (
	"fmt"
	"math"
	"time"
)

// ==========================================
// Ode45: Dormand-Prince 5(4) Explicit RK
// ==========================================

// Dormand-Prince 5(4) Coefficients
const (
	dpC2 = 1.0 / 5.0
	dpC3 = 3.0 / 10.0
	dpC4 = 4.0 / 5.0
	dpC5 = 8.0 / 9.0

	dpA21 = 1.0 / 5.0
	dpA31 = 3.0 / 40.0
	dpA32 = 9.0 / 40.0
	dpA41 = 44.0 / 45.0
	dpA42 = -56.0 / 15.0
	dpA43 = 32.0 / 9.0
	dpA51 = 19372.0 / 6561.0
	dpA52 = -25360.0 / 2187.0
	dpA53 = 64448.0 / 6561.0
	dpA54 = -212.0 / 729.0
	dpA61 = 9017.0 / 3168.0
	dpA62 = -355.0 / 33.0
	dpA63 = 46732.0 / 5247.0
	dpA64 = 49.0 / 176.0
	dpA65 = -5103.0 / 18656.0

	// b1 = dpA71, b3 = dpA73, b4 = dpA74, b5 = dpA75, b6 = dpA76
	dpA71 = 35.0 / 384.0
	dpA73 = 500.0 / 1113.0
	dpA74 = 125.0 / 192.0
	dpA75 = -2187.0 / 6784.0
	dpA76 = 11.0 / 84.0

	// Error estimation coefficients
	dpE1 = 71.0 / 57600.0
	dpE3 = -71.0 / 16695.0
	dpE4 = 71.0 / 1920.0
	dpE5 = -17253.0 / 339200.0
	dpE6 = 22.0 / 525.0
	dpE7 = -1.0 / 40.0
)

// Ode45Workspace holds all pre-allocated memory slices for the Ode45 solver
// to guarantee zero-allocation and maximize performance during integration.
type Ode45Workspace struct {
	k1    []float64
	k2    []float64
	k3    []float64
	k4    []float64
	k5    []float64
	k6    []float64
	k7    []float64
	yTemp []float64
	yNext []float64
}

func newOde45Workspace(dim int) *Ode45Workspace {
	return &Ode45Workspace{
		k1:    make([]float64, dim),
		k2:    make([]float64, dim),
		k3:    make([]float64, dim),
		k4:    make([]float64, dim),
		k5:    make([]float64, dim),
		k6:    make([]float64, dim),
		k7:    make([]float64, dim),
		yTemp: make([]float64, dim),
		yNext: make([]float64, dim),
	}
}

// evalExplicitDeriv is a helper to evaluate the ODE function.
// For standard ODEs, M is nil. If a diagonal Mass matrix M is provided,
// it computes dy/dt = F(t, y) ./ diag(M).
func evalExplicitDeriv(odefun OdeFunc, t float64, y []float64, kOut []float64) {
	res := odefun(t, y)
	if res.M == nil {
		copy(kOut, res.F)
	} else {
		for d := 0; d < len(y); d++ {
			mVal := res.M[d]
			if math.Abs(mVal) <= 1e-12 {
				panic(fmt.Sprintf("Ode45: mathematically impossible to solve DAEs. Singular mass matrix detected at index %d. Please use Ode15s instead.", d))
			} else {
				kOut[d] = res.F[d] / mVal
			}
		}
	}
}

// Ode45 is a high-precision Explicit Runge-Kutta (4,5) solver.
// Ideal for solving non-stiff systems of differential equations.
func Ode45(odefun OdeFunc, tspan [2]float64, y0 []float64, info *OdeInfo) *OdeResult {
	if info == nil {
		info = &OdeInfo{}
	}

	absTol := info.AbsTol
	if absTol <= 0 {
		absTol = 1e-6
	}
	relTol := info.RelTol
	if relTol <= 0 {
		relTol = absTol
	}

	maxStepsLimit := info.MaxStep
	if maxStepsLimit <= 0 {
		maxStepsLimit = 2000000
	}
	maxTime := info.MaxTime
	if maxTime <= 0 {
		maxTime = 1200 * time.Second
	}
	startTime := time.Now()

	yCurr := cloneSlice(y0)
	dim := len(yCurr)

	tStart := tspan[0]
	tFinal := tspan[1]
	t := tStart

	if tStart == tFinal {
		return nil
	}

	tSpan := tFinal - tStart
	direction := 1.0
	if tSpan < 0 {
		direction = -1.0
	}

	var testProgress func(t float64, y []float64)
	if info.Progress > 0 {
		progress := info.Progress
		if progress <= 0 || progress >= 1 {
			progress = 0.1
		}
		lastProgress := 0.0
		testProgress = func(t float64, y []float64) {
			pos := (t - tStart) / tSpan
			if pos-lastProgress >= progress {
				lastProgress = pos
				if info.ProgressCb != nil {
					info.ProgressCb(pos, t, y)
				} else {
					fmt.Printf("Progress at %.1f%%, y=%v\n", pos*100, y[0])
				}
			}
		}
	}

	machineEps := math.Nextafter(1.0, 2.0) - 1.0
	timeTolerance := (math.Abs(tFinal) + 1.0) * machineEps * 1e4
	minHLimit := 1e-14

	h := math.Abs(info.InitialStep)
	if h == 0 {
		h = math.Abs(tSpan) * 0.01
		if h > 0.1 {
			h = 0.1
		}
	}
	if h < minHLimit {
		h = minHLimit
	}
	h *= direction

	info.T = make([]float64, 0, 1000)
	info.Y = make([][]float64, 0, 1000)
	info.Dy = make([][]float64, 0, 1000)

	info.T = append(info.T, t)
	info.Y = append(info.Y, cloneSlice(yCurr))

	info.Steps = 0
	info.FailedSteps = 0
	info.Status = "running"

	ws := newOde45Workspace(dim)
	evalExplicitDeriv(odefun, t, yCurr, ws.k1)
	info.Dy = append(info.Dy, cloneSlice(ws.k1))

	done := false
	steps := 0

	for !done {
		if steps >= maxStepsLimit {
			info.Status = "max_steps"
			break
		}
		if time.Since(startTime) > maxTime {
			info.Status = "timeout"
			break
		}

		distToEnd := tFinal - t
		distAbs := math.Abs(distToEnd)
		hAbs := math.Abs(h)

		if distAbs <= timeTolerance {
			done = true
			break
		}

		lastStep := false
		if hAbs >= distAbs {
			h = distToEnd
			lastStep = true
		}

		// --- Stage 2 ---
		for i := 0; i < dim; i++ {
			ws.yTemp[i] = yCurr[i] + h*(dpA21*ws.k1[i])
		}
		evalExplicitDeriv(odefun, t+(dpC2*h), ws.yTemp, ws.k2)

		// --- Stage 3 ---
		for i := 0; i < dim; i++ {
			ws.yTemp[i] = yCurr[i] + h*(dpA31*ws.k1[i]+dpA32*ws.k2[i])
		}
		evalExplicitDeriv(odefun, t+(dpC3*h), ws.yTemp, ws.k3)

		// --- Stage 4 ---
		for i := 0; i < dim; i++ {
			ws.yTemp[i] = yCurr[i] + h*(dpA41*ws.k1[i]+dpA42*ws.k2[i]+dpA43*ws.k3[i])
		}
		evalExplicitDeriv(odefun, t+(dpC4*h), ws.yTemp, ws.k4)

		// --- Stage 5 ---
		for i := 0; i < dim; i++ {
			ws.yTemp[i] = yCurr[i] + h*(dpA51*ws.k1[i]+dpA52*ws.k2[i]+dpA53*ws.k3[i]+dpA54*ws.k4[i])
		}
		evalExplicitDeriv(odefun, t+(dpC5*h), ws.yTemp, ws.k5)

		// --- Stage 6 ---
		for i := 0; i < dim; i++ {
			ws.yTemp[i] = yCurr[i] + h*(dpA61*ws.k1[i]+dpA62*ws.k2[i]+dpA63*ws.k3[i]+dpA64*ws.k4[i]+dpA65*ws.k5[i])
		}
		evalExplicitDeriv(odefun, t+h, ws.yTemp, ws.k6) // c6 = 1.0

		// --- Stage 7 (Result Candidate & Next K1) ---
		for i := 0; i < dim; i++ {
			ws.yNext[i] = yCurr[i] + h*(dpA71*ws.k1[i]+dpA73*ws.k3[i]+dpA74*ws.k4[i]+dpA75*ws.k5[i]+dpA76*ws.k6[i])
		}
		evalExplicitDeriv(odefun, t+h, ws.yNext, ws.k7)

		// --- Error Calculation ---
		maxNormErr := 0.0
		for i := 0; i < dim; i++ {
			errTerm := dpE1*ws.k1[i] + dpE3*ws.k3[i] + dpE4*ws.k4[i] + dpE5*ws.k5[i] + dpE6*ws.k6[i] + dpE7*ws.k7[i]
			errAbsVal := math.Abs(errTerm * h)

			yMax := math.Abs(yCurr[i])
			yNextAbs := math.Abs(ws.yNext[i])
			if yNextAbs > yMax {
				yMax = yNextAbs
			}

			sc := absTol + (relTol * yMax)
			ratio := errAbsVal / sc

			// Safety check for NaN limits
			if math.IsNaN(ratio) {
				maxNormErr = math.MaxFloat64
				break
			}
			if ratio > maxNormErr {
				maxNormErr = ratio
			}
		}

		// --- Step Accept / Reject Logic ---
		if maxNormErr <= 1.0 {
			// ACCEPT
			t += h
			copy(yCurr, ws.yNext)
			copy(ws.k1, ws.k7) // FSAL property: current k7 becomes next step's k1
			steps++

			info.T = append(info.T, t)
			info.Y = append(info.Y, cloneSlice(yCurr))
			info.Dy = append(info.Dy, cloneSlice(ws.k1))

			if info.Cb != nil {
				info.Cb(t, yCurr)
			}
			if testProgress != nil {
				testProgress(t, yCurr)
			}

			if lastStep {
				done = true
				break
			}

			// Grow step
			factor := 5.0
			if maxNormErr > 1e-15 {
				factor = 0.9 * math.Pow(maxNormErr, -0.2)
			}
			if factor > 5.0 {
				factor = 5.0
			} else if factor < 0.1 {
				factor = 0.1
			}
			h *= factor

		} else {
			// REJECT
			info.FailedSteps++

			// Shrink step
			factor := 0.9 * math.Pow(maxNormErr, -0.2)
			if math.IsNaN(factor) {
				factor = 0.1
			}
			if factor < 0.1 {
				factor = 0.1
			} else if factor > 0.8 {
				factor = 0.8
			}

			h *= factor

			// Underflow Safety
			if math.Abs(h) < minHLimit {
				ConsoleLog("ode45: Step size underflow limit reached.")
				info.Status = "underflow"
				break
			}
		}
	}

	info.ExecTime = time.Since(startTime)
	info.Steps = steps

	if info.Status == "running" {
		info.Status = "done"
		return &OdeResult{
			T:  info.T,
			Y:  info.Y,
			Dy: info.Dy,
		}
	}

	return nil
}
