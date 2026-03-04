package main

import (
	"fmt"
	"math"
	"time"
)

// ==========================================
// 1. Data Structures & Types
// ==========================================

// LUResult represents the output of a sparse LU factorization.
type LUResult struct {
	L *LFactor
	U *UFactor
}

// SparseMatrixCSC defines the interface for Compressed Sparse Column matrices.
type SparseMatrixCSC interface {
	LU() (LUResult, error)
}

// CooMatrix represents a matrix in Coordinate format.
type CooMatrix struct {
	RowIdx []int
	ColIdx []int
	Vals   []float64
}

// OdeRes is the return type for the ODE function.
// If M is nil, it represents an identity mass matrix (standard ODE).
// If M is provided, it represents the diagonal of the Mass matrix for DAEs.
type OdeRes struct {
	M []float64
	F []float64
}

// OdeFunc defines the signature of the system of equations.
type OdeFunc func(t float64, y []float64) OdeRes

// JacobianFunc defines the analytical Jacobian callback.
type JacobianFunc func(t float64, y []float64, f []float64) CooMatrix

// OdeInfo holds configuration and runtime status.
type OdeInfo struct {
	// Configuration
	AbsTol        float64
	RelTol        float64
	EstimateError bool
	InitialStep   float64
	Progress      float64
	ProgressCb    func(pos, t float64, y []float64)
	MaxStep       int
	MaxTime       time.Duration
	Cb            func(t float64, y []float64)
	Jacobian      JacobianFunc

	// Output & Status
	T                  []float64
	Y                  [][]float64
	Dy                 [][]float64
	Steps              int
	FailedSteps        int
	Status             string
	GlobalErrorHistory [][]float64
	GlobalError        []float64
	ExecTime           time.Duration
}

// OdeResult holds the final integration arrays.
type OdeResult struct {
	T  []float64
	Y  [][]float64
	Dy [][]float64
}

// historyPoint stores previous successful steps for BDF extrapolation.
type historyPoint struct {
	t float64
	y []float64
	f []float64
	M []float64
}

// bdfCoeffs stores Classical BDF Coefficients (Order 1 to 5).
type bdfCoeffs struct {
	alpha []float64
	beta  float64
}

var bdfKArr = []float64{0, 1, 2, 3, 4, 5, 6}

var BDF = []bdfCoeffs{
	{},                                 // 0
	{alpha: []float64{1.0}, beta: 1.0}, // 1st Order (Backward Euler)
	{alpha: []float64{4.0 / 3.0, -1.0 / 3.0}, beta: 2.0 / 3.0},                                                        // 2nd Order
	{alpha: []float64{18.0 / 11.0, -9.0 / 11.0, 2.0 / 11.0}, beta: 6.0 / 11.0},                                        // 3rd Order
	{alpha: []float64{48.0 / 25.0, -36.0 / 25.0, 16.0 / 25.0, -3.0 / 25.0}, beta: 12.0 / 25.0},                        // 4th Order
	{alpha: []float64{300.0 / 137.0, -300.0 / 137.0, 200.0 / 137.0, -75.0 / 137.0, 12.0 / 137.0}, beta: 60.0 / 137.0}, // 5th Order
}

// OdeWorkspace holds all pre-allocated memory slices internally used by the solver
// to guarantee zero-allocation and maximize performance during hot iterations.
type OdeWorkspace struct {
	yPred    []float64
	yStar    [][]float64
	C        []float64
	yTmpCurr []float64
	negG     []float64
	yTmp     []float64
	deltaY   []float64
	dyCurr   []float64
	rhsE     []float64
	eTmp     []float64
	yPert    []float64

	D [][][]float64

	luRowIdx []int
	luColIdx []int
	luVals   []float64

	colCounts []int
	curPos    []int
	rowIdxCSC []int
	valsCSC   []float64

	lRow  [][]int
	lVals [][]float64
	uRow  [][]int
	uVals [][]float64
	uDiag []float64
	P     []int
	PInv  []int
	X     []float64

	lColPtr   []int
	lRowIdx   []int
	lValsFlat []float64

	uColPtr   []int
	uRowIdx   []int
	uValsFlat []float64
}

func newOdeWorkspace(dim int) *OdeWorkspace {
	ws := &OdeWorkspace{
		yPred:    make([]float64, dim),
		C:        make([]float64, dim),
		yTmpCurr: make([]float64, dim),
		negG:     make([]float64, dim),
		yTmp:     make([]float64, dim),
		deltaY:   make([]float64, dim),
		dyCurr:   make([]float64, dim),
		rhsE:     make([]float64, dim),
		eTmp:     make([]float64, dim),
		yPert:    make([]float64, dim),

		yStar: make([][]float64, 6),
		D:     make([][][]float64, 7),

		luRowIdx: make([]int, 0, dim*4),
		luColIdx: make([]int, 0, dim*4),
		luVals:   make([]float64, 0, dim*4),

		colCounts: make([]int, dim),
		curPos:    make([]int, dim),
		rowIdxCSC: make([]int, 0, dim*4),
		valsCSC:   make([]float64, 0, dim*4),

		lRow:  make([][]int, dim),
		lVals: make([][]float64, dim),
		uRow:  make([][]int, dim),
		uVals: make([][]float64, dim),
		uDiag: make([]float64, dim),
		P:     make([]int, dim),
		PInv:  make([]int, dim),
		X:     make([]float64, dim),

		lColPtr:   make([]int, 0, dim+1),
		lRowIdx:   make([]int, 0, dim*4),
		lValsFlat: make([]float64, 0, dim*4),

		uColPtr:   make([]int, 0, dim+1),
		uRowIdx:   make([]int, 0, dim*4),
		uValsFlat: make([]float64, 0, dim*4),
	}
	for i := 0; i < 6; i++ {
		ws.yStar[i] = make([]float64, dim)
	}
	for i := 0; i < 7; i++ {
		ws.D[i] = make([][]float64, 7)
		for j := 0; j < 7; j++ {
			ws.D[i][j] = make([]float64, dim)
		}
	}
	for i := 0; i < dim; i++ {
		ws.lRow[i] = make([]int, 0, 16)
		ws.lVals[i] = make([]float64, 0, 16)
		ws.uRow[i] = make([]int, 0, 16)
		ws.uVals[i] = make([]float64, 0, 16)
	}
	return ws
}

// ==========================================
// 2. High-Precision Stiff ODE Solver
// ==========================================

// Ode15s is the main solver function.
func Ode15s(odefun OdeFunc, tspan [2]float64, y0 []float64, info *OdeInfo) *OdeResult {
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
	jacobianEps := math.Sqrt(machineEps)
	tinyLte := 1e-10
	mZeroTol := 1e-12

	newtonTol := 0.05

	h := math.Abs(info.InitialStep)
	if h == 0 {
		h = math.Abs(tSpan) * 0.01
	}
	if h < 1e-12 {
		h = 1e-12
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

	k := 1

	res0 := odefun(t, yCurr)
	M0 := res0.M
	f0 := res0.F

	dy0 := make([]float64, dim)
	for d := 0; d < dim; d++ {
		mVal := 1.0
		if M0 != nil {
			mVal = M0[d]
		}
		if math.Abs(mVal) <= mZeroTol {
			dy0[d] = 0.0
		} else {
			dy0[d] = f0[d] / mVal
		}
	}
	info.Dy = append(info.Dy, dy0)

	history := make([]historyPoint, 1, 7)
	history[0] = historyPoint{
		t: t,
		y: cloneSlice(yCurr),
		f: cloneSlice(f0),
	}
	if M0 != nil {
		history[0].M = cloneSlice(M0)
	}

	var globalError []float64
	if info.EstimateError {
		globalError = make([]float64, dim)
		info.GlobalErrorHistory = make([][]float64, 0, 1000)
		info.GlobalErrorHistory = append(info.GlobalErrorHistory, cloneSlice(globalError))
	}

	updateJacobian := true
	updateLU := true
	stepsSinceJacobian := 0
	lastHBeta := 0.0
	var JM []float64

	// Pre-allocate structs to avoid GC allocations in loop
	ws := newOdeWorkspace(dim)
	var Jf CooMatrix
	jgSparse := &CSCMatrix{
		ColPtr: make([]int, 0, dim+1),
		RowIdx: make([]int, 0, dim*4),
		Vals:   make([]float64, 0, dim*4),
	}
	luRes := LUResult{
		L: &LFactor{
			ColPtr: make([]int, 0, dim+1),
			RowIdx: make([]int, 0, dim*4),
			Vals:   make([]float64, 0, dim*4),
			P:      make([]int, 0, dim),
		},
		U: &UFactor{
			ColPtr: make([]int, 0, dim+1),
			RowIdx: make([]int, 0, dim*4),
			Vals:   make([]float64, 0, dim*4),
			Diag:   make([]float64, 0, dim),
		},
	}

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

		if distAbs <= timeTolerance {
			done = true
			break
		}

		lastStep := false
		if math.Abs(h) >= distAbs {
			h = distToEnd
			lastStep = true
			updateLU = true
		}

		tNext := t + h
		pts := history
		if len(history) > k+1 {
			pts = history[:k+1]
		}

		var yStar [][]float64

		if len(pts) == 1 {
			mStart := pts[0].M
			fStart := pts[0].f
			for d := 0; d < dim; d++ {
				mVal := 1.0
				if mStart != nil {
					mVal = mStart[d]
				}
				if math.Abs(mVal) <= mZeroTol {
					ws.yPred[d] = pts[0].y[d]
				} else {
					ws.yPred[d] = pts[0].y[d] + (h * fStart[d] / mVal)
				}
			}
			copy(ws.yStar[0], pts[0].y)
			yStar = ws.yStar[:1]
		} else {
			computeDividedDifferences(pts, dim, ws.D)
			evalPoly(pts, ws.D, tNext, dim, ws.yPred)
			for j := 1; j <= k; j++ {
				evalPoly(pts, ws.D, tNext-(h*bdfKArr[j]), dim, ws.yStar[j-1])
			}
			yStar = ws.yStar[:k]
		}

		for d := 0; d < dim; d++ {
			ws.C[d] = 0.0
		}
		for j := 1; j <= k; j++ {
			alphaJ := BDF[k].alpha[j-1]
			for d := 0; d < dim; d++ {
				ws.C[d] += alphaJ * yStar[j-1][d]
			}
		}

		betaK := BDF[k].beta
		hBeta := h * betaK

		if stepsSinceJacobian >= 400 {
			updateJacobian = true
		}

		if updateJacobian {
			resPred := odefun(tNext, ws.yPred)
			fPred := resPred.F
			JM = resPred.M
			getJacobian(info, odefun, dim, tNext, ws.yPred, fPred, jacobianEps, ws, &Jf)

			stepsSinceJacobian = 0
			updateLU = true
			updateJacobian = false
		}

		if !updateLU {
			hBetaRatio := math.Abs(hBeta / lastHBeta)
			if !(hBetaRatio >= 0.8 && hBetaRatio <= 1.25) {
				updateLU = true
			}
		}

		if updateLU {
			ws.luRowIdx = ws.luRowIdx[:0]
			ws.luColIdx = ws.luColIdx[:0]
			ws.luVals = ws.luVals[:0]

			for i := 0; i < dim; i++ {
				mVal := 1.0
				if JM != nil {
					mVal = JM[i]
				}
				if mVal != 0 {
					ws.luRowIdx = append(ws.luRowIdx, i)
					ws.luColIdx = append(ws.luColIdx, i)
					ws.luVals = append(ws.luVals, mVal)
				}
			}

			for i := 0; i < len(Jf.Vals); i++ {
				jTerm := -(hBeta * Jf.Vals[i])
				ws.luRowIdx = append(ws.luRowIdx, Jf.RowIdx[i])
				ws.luColIdx = append(ws.luColIdx, Jf.ColIdx[i])
				ws.luVals = append(ws.luVals, jTerm)
			}

			err := buildCSC(jgSparse, dim, dim, ws.luRowIdx, ws.luColIdx, ws.luVals, ws)
			if err != nil {
				h *= 0.2
				info.FailedSteps++
				lastStep = false
				updateJacobian = true
				continue
			}

			err = jgSparse.LUWithWorkspace(&luRes, ws)
			if err != nil {
				h *= 0.2
				info.FailedSteps++
				lastStep = false
				updateJacobian = true

				if !(math.Abs(h) >= minHLimit) {
					fmt.Println("ode15s: Step size underflow limit reached.")
					info.Status = "underflow"
					break
				}
				continue
			}

			lastHBeta = hBeta
			updateLU = false
		}

		copy(ws.yTmpCurr, ws.yPred)
		newtonConverged := false
		var MCurr []float64
		var fCurr []float64
		oldDeltaNorm := -1.0

		for iter := 0; iter < 5; iter++ {
			resCurr := odefun(tNext, ws.yTmpCurr)
			MCurr = resCurr.M
			fCurr = resCurr.F

			for d := 0; d < dim; d++ {
				yDiff := ws.yTmpCurr[d] - ws.C[d]
				mTerm := yDiff
				if MCurr != nil {
					mTerm = MCurr[d] * yDiff
				}
				gVal := mTerm - (hBeta * fCurr[d])
				ws.negG[d] = -gVal
			}

			luRes.L.SolveLowerTriangularInPlace(ws.negG, ws.yTmp)
			luRes.U.SolveUpperTriangularInPlace(ws.yTmp, ws.deltaY)

			stepConverged := true
			currentDeltaNorm := 0.0

			for d := 0; d < dim; d++ {
				ws.yTmpCurr[d] += ws.deltaY[d]

				maxY := math.Abs(yCurr[d])
				if !(math.Abs(ws.yTmpCurr[d]) <= maxY) {
					maxY = math.Abs(ws.yTmpCurr[d])
				}
				sc := absTol + (relTol * maxY)

				ratio := math.Abs(ws.deltaY[d]) / sc
				if !(ratio <= currentDeltaNorm) {
					currentDeltaNorm = ratio
				}

				if !(math.Abs(ws.deltaY[d]) <= (sc * newtonTol)) {
					stepConverged = false
				}
			}

			if stepConverged {
				newtonConverged = true
				break
			}

			if iter > 0 && oldDeltaNorm >= 0 {
				theta := currentDeltaNorm / oldDeltaNorm
				if theta > 0.8 {
					break
				}
			}
			oldDeltaNorm = currentDeltaNorm
		}

		if !newtonConverged {
			if stepsSinceJacobian > 0 {
				updateJacobian = true
				continue
			}
			h *= 0.5
			info.FailedSteps++
			lastStep = false
			updateJacobian = true
			continue
		}

		LTENorm := 0.0
		invKPlus1 := 1.0 / bdfKArr[k+1]

		for d := 0; d < dim; d++ {
			mVal := 1.0
			if MCurr != nil {
				mVal = MCurr[d]
			}
			if math.Abs(mVal) <= mZeroTol {
				continue
			}

			errVal := (ws.yTmpCurr[d] - ws.yPred[d]) * invKPlus1
			maxY := math.Abs(yCurr[d])
			if !(math.Abs(ws.yTmpCurr[d]) <= maxY) {
				maxY = math.Abs(ws.yTmpCurr[d])
			}
			sc := absTol + (relTol * maxY)

			ratio := math.Abs(errVal) / sc
			if !(ratio <= LTENorm) {
				LTENorm = ratio // If ratio is NaN, LTENorm becomes NaN (which safely triggers rejection later)
			}
		}

		if LTENorm <= 1.0 {
			if info.EstimateError {
				for d := 0; d < dim; d++ {
					mVal := 1.0
					if MCurr != nil {
						mVal = MCurr[d]
					}
					if math.Abs(mVal) <= mZeroTol {
						ws.rhsE[d] = globalError[d]
					} else {
						errValTrue := (ws.yTmpCurr[d] - ws.yPred[d]) * invKPlus1
						ws.rhsE[d] = globalError[d] + errValTrue
					}
				}

				luRes.L.SolveLowerTriangularInPlace(ws.rhsE, ws.eTmp)
				luRes.U.SolveUpperTriangularInPlace(ws.eTmp, globalError)
				info.GlobalErrorHistory = append(info.GlobalErrorHistory, cloneSlice(globalError))
			}

			stepsSinceJacobian++
			t = tNext
			copy(yCurr, ws.yTmpCurr)

			resFinal := odefun(t, yCurr)
			fFinal := resFinal.F
			MFinal := resFinal.M

			// Avoid slice/struct allocations by using a shifting buffer pattern locally.
			lastIdx := len(history) - 1
			var recycleY, recycleF, recycleM []float64
			if len(history) == 7 {
				recycleY = history[6].y
				recycleF = history[6].f
				recycleM = history[6].M
			} else {
				history = append(history, historyPoint{})
				lastIdx = len(history) - 1
				recycleY = make([]float64, dim)
				recycleF = make([]float64, dim)
				if MFinal != nil {
					recycleM = make([]float64, dim)
				}
			}

			for i := lastIdx; i > 0; i-- {
				history[i] = history[i-1]
			}

			history[0].t = t
			history[0].y = recycleY
			history[0].f = recycleF
			history[0].M = recycleM

			copy(history[0].y, yCurr)
			copy(history[0].f, fFinal)
			if MFinal != nil {
				if history[0].M == nil {
					history[0].M = make([]float64, dim)
				}
				copy(history[0].M, MFinal)
			} else {
				history[0].M = nil
			}

			for d := 0; d < dim; d++ {
				ws.dyCurr[d] = (yCurr[d] - ws.C[d]) / hBeta
			}

			info.T = append(info.T, t)
			info.Y = append(info.Y, cloneSlice(yCurr))
			info.Dy = append(info.Dy, cloneSlice(ws.dyCurr))

			if info.Cb != nil {
				info.Cb(t, yCurr)
			}
			if testProgress != nil {
				testProgress(t, yCurr)
			}
			steps++

			if lastStep {
				done = true
				break
			}

			computeDividedDifferences(history, dim, ws.D)
			DNew := ws.D
			LHist := len(history)
			maxHOpt := 0.0
			nextK := k

			startM := k - 1
			if startM < 1 {
				startM = 1
			}
			endM := k + 1
			if endM > 5 {
				endM = 5
			}

			for m := startM; m <= endM; m++ {
				errNormM := 0.0

				if m == k {
					errNormM = LTENorm
				} else if m+1 < LHist {
					term := 1.0
					for i := 1; i <= m+1; i++ {
						term *= (history[0].t - history[i].t)
					}

					invMPlus1 := 1.0 / bdfKArr[m+1]

					for d := 0; d < dim; d++ {
						mVal := 1.0
						if MFinal != nil {
							mVal = MFinal[d]
						}
						if math.Abs(mVal) <= mZeroTol {
							continue
						}

						errVal := DNew[0][m+1][d] * term * invMPlus1
						sc := absTol + (relTol * math.Abs(yCurr[d]))
						// Ensure NaN is captured in error norm evaluation
						ratio := math.Abs(errVal) / sc
						if !(ratio <= errNormM) {
							errNormM = ratio // If ratio is NaN, errNormM becomes NaN (which safely triggers rejection later)
						}
					}
				} else {
					continue
				}

				if errNormM < tinyLte {
					errNormM = tinyLte
				}

				hOptM := 0.9 * math.Pow(errNormM, -1.0/float64(m+1))

				if m == k-1 {
					hOptM *= 1.2
				}
				if m == k+1 {
					hOptM *= 0.8
				}

				if hOptM > maxHOpt {
					maxHOpt = hOptM
					nextK = m
				}
			}

			factor := maxHOpt
			if factor < 0.2 {
				factor = 0.2
			} else if factor > 5.0 {
				factor = 5.0
			}

			if factor >= 1.5 || factor <= 0.8 || nextK != k {
				h *= factor
				k = nextK
			}

		} else {
			info.FailedSteps++
			lastStep = false
			updateJacobian = true

			if LTENorm < tinyLte {
				LTENorm = tinyLte
			}

			factor := 0.9 * math.Pow(LTENorm, -1.0/float64(k+1))
			if factor < 0.1 {
				factor = 0.1
			}
			h *= factor

			if !(math.Abs(h) >= minHLimit) {
				fmt.Println("ode15s: Step size underflow limit reached.")
				info.Status = "underflow"
				break
			}
		}
	}

	info.ExecTime = time.Since(startTime)
	info.Steps = steps
	if info.EstimateError {
		info.GlobalError = globalError
	}

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

// ==========================================
// 3. Mathematical Helpers
// ==========================================

func computeDividedDifferences(pts []historyPoint, dim int, D [][][]float64) {
	m := len(pts) - 1

	for i := 0; i <= m; i++ {
		copy(D[i][0], pts[i].y)
	}

	for j := 1; j <= m; j++ {
		for i := 0; i <= m-j; i++ {
			dx := pts[i].t - pts[i+j].t
			invDx := 1.0 / dx
			for d := 0; d < dim; d++ {
				D[i][j][d] = (D[i][j-1][d] - D[i+1][j-1][d]) * invDx
			}
		}
	}
}

func evalPoly(pts []historyPoint, D [][][]float64, tTarget float64, dim int, res []float64) {
	m := len(pts) - 1
	for d := 0; d < dim; d++ {
		res[d] = 0.0
	}
	term := 1.0
	for j := 0; j <= m; j++ {
		for d := 0; d < dim; d++ {
			res[d] += D[0][j][d] * term
		}
		if j < m {
			term *= (tTarget - pts[j].t)
		}
	}
}

func getJacobian(info *OdeInfo, odefun OdeFunc, dim int, tVal float64, yVal []float64, fVal []float64, jacobianEps float64, ws *OdeWorkspace, Jf *CooMatrix) {
	if info.Jacobian != nil {
		userJf := info.Jacobian(tVal, yVal, fVal)
		Jf.RowIdx = append(Jf.RowIdx[:0], userJf.RowIdx...)
		Jf.ColIdx = append(Jf.ColIdx[:0], userJf.ColIdx...)
		Jf.Vals = append(Jf.Vals[:0], userJf.Vals...)
		return
	}

	Jf.RowIdx = Jf.RowIdx[:0]
	Jf.ColIdx = Jf.ColIdx[:0]
	Jf.Vals = Jf.Vals[:0]

	for j := 0; j < dim; j++ {
		copy(ws.yPert, yVal)
		delta := math.Abs(yVal[j]) * jacobianEps

		if delta == 0 {
			delta = jacobianEps
		}

		invDelta := 1.0 / delta
		ws.yPert[j] += delta

		resPert := odefun(tVal, ws.yPert)
		fPert := resPert.F

		for i := 0; i < dim; i++ {
			diff := (fPert[i] - fVal[i]) * invDelta
			if diff != 0 {
				Jf.RowIdx = append(Jf.RowIdx, i)
				Jf.ColIdx = append(Jf.ColIdx, j)
				Jf.Vals = append(Jf.Vals, diff)
			}
		}
	}
}

func cloneSlice(src []float64) []float64 {
	if src == nil {
		return nil
	}
	dst := make([]float64, len(src))
	copy(dst, src)
	return dst
}

// ==========================================
// 4. Sparse Matrix CSC & Factorization
// ==========================================

// CSCMatrix implements Compressed Sparse Column format.
type CSCMatrix struct {
	Rows   int
	Cols   int
	ColPtr []int
	RowIdx []int
	Vals   []float64
}

// NewSparseMatrixCSCFromCOO constructs a CSC matrix from COO coordinates.
// It safely sums duplicate coordinate entries which is critical for finite-difference Jacobian assembly.
// Preserves the identical API wrapper over optimized zero-allocation logic.
func NewSparseMatrixCSCFromCOO(rows, cols int, rowIdx, colIdx []int, vals []float64) (*CSCMatrix, error) {
	A := &CSCMatrix{}
	ws := newOdeWorkspace(cols)
	err := buildCSC(A, rows, cols, rowIdx, colIdx, vals, ws)
	if err != nil {
		return nil, err
	}
	return A, nil
}

// buildCSC provides inner allocation-free CSC assembling process
func buildCSC(A *CSCMatrix, rows, cols int, rowIdx, colIdx []int, vals []float64, ws *OdeWorkspace) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("invalid matrix dimensions: %dx%d", rows, cols)
	}

	for i := 0; i < cols; i++ {
		ws.colCounts[i] = 0
	}
	for _, c := range colIdx {
		ws.colCounts[c]++
	}

	if cap(A.ColPtr) < cols+1 {
		A.ColPtr = make([]int, cols+1)
	} else {
		A.ColPtr = A.ColPtr[:cols+1]
	}

	A.ColPtr[0] = 0
	for i := 0; i < cols; i++ {
		A.ColPtr[i+1] = A.ColPtr[i] + ws.colCounts[i]
	}

	if cap(ws.rowIdxCSC) < len(vals) {
		ws.rowIdxCSC = make([]int, len(vals))
		ws.valsCSC = make([]float64, len(vals))
	} else {
		ws.rowIdxCSC = ws.rowIdxCSC[:len(vals)]
		ws.valsCSC = ws.valsCSC[:len(vals)]
	}

	copy(ws.curPos, A.ColPtr[:cols])

	for i := 0; i < len(vals); i++ {
		c := colIdx[i]
		pos := ws.curPos[c]
		ws.rowIdxCSC[pos] = rowIdx[i]
		ws.valsCSC[pos] = vals[i]
		ws.curPos[c]++
	}

	A.RowIdx = A.RowIdx[:0]
	A.Vals = A.Vals[:0]

	for c := 0; c < cols; c++ {
		start := A.ColPtr[c]
		end := A.ColPtr[c+1]

		if end > start {
			// Fast insertion sort to arrange indices within the small column buffer safely
			for i := start + 1; i < end; i++ {
				rTmp := ws.rowIdxCSC[i]
				vTmp := ws.valsCSC[i]
				j := i - 1
				for j >= start && ws.rowIdxCSC[j] > rTmp {
					ws.rowIdxCSC[j+1] = ws.rowIdxCSC[j]
					ws.valsCSC[j+1] = ws.valsCSC[j]
					j--
				}
				ws.rowIdxCSC[j+1] = rTmp
				ws.valsCSC[j+1] = vTmp
			}

			A.ColPtr[c] = len(A.RowIdx)
			curR := ws.rowIdxCSC[start]
			curV := ws.valsCSC[start]
			for i := start + 1; i < end; i++ {
				if ws.rowIdxCSC[i] == curR {
					curV += ws.valsCSC[i]
				} else {
					A.RowIdx = append(A.RowIdx, curR)
					A.Vals = append(A.Vals, curV)
					curR = ws.rowIdxCSC[i]
					curV = ws.valsCSC[i]
				}
			}
			A.RowIdx = append(A.RowIdx, curR)
			A.Vals = append(A.Vals, curV)
		} else {
			A.ColPtr[c] = len(A.RowIdx)
		}
	}
	A.ColPtr[cols] = len(A.RowIdx)
	A.Rows = rows
	A.Cols = cols

	return nil
}

// LU performs Left-Looking Sparse LU Factorization with Partial Pivoting.
func (A *CSCMatrix) LU() (LUResult, error) {
	ws := newOdeWorkspace(A.Cols)
	var res LUResult
	err := A.LUWithWorkspace(&res, ws)
	return res, err
}

// LUWithWorkspace exposes internal preallocated fast factorization mechanics.
func (A *CSCMatrix) LUWithWorkspace(res *LUResult, ws *OdeWorkspace) error {
	n := A.Cols
	if n != A.Rows {
		return fmt.Errorf("LU factorization requires a square matrix")
	}

	for i := 0; i < n; i++ {
		ws.P[i] = i
		ws.PInv[i] = i
		ws.lRow[i] = ws.lRow[i][:0]
		ws.lVals[i] = ws.lVals[i][:0]
		ws.uRow[i] = ws.uRow[i][:0]
		ws.uVals[i] = ws.uVals[i][:0]
	}

	for k := 0; k < n; k++ {
		for i := 0; i < n; i++ {
			ws.X[i] = 0.0
		}

		for p := A.ColPtr[k]; p < A.ColPtr[k+1]; p++ {
			origR := A.RowIdx[p]
			ws.X[ws.PInv[origR]] = A.Vals[p]
		}

		for j := 0; j < k; j++ {
			if ws.X[j] != 0 {
				ws.uRow[k] = append(ws.uRow[k], j)
				ws.uVals[k] = append(ws.uVals[k], ws.X[j])

				xj := ws.X[j]
				lRowJ := ws.lRow[j]
				lValsJ := ws.lVals[j]

				for idx := 0; idx < len(lRowJ); idx++ {
					r := lRowJ[idx]
					ws.X[r] -= xj * lValsJ[idx]
				}
			}
		}

		pivotVal := 0.0
		p := k
		for i := k; i < n; i++ {
			if !(math.Abs(ws.X[i]) <= math.Abs(pivotVal)) {
				pivotVal = ws.X[i]
				p = i
			}
		}

		if pivotVal == 0.0 {
			return fmt.Errorf("structural singularity encountered at column %d", k)
		}

		if p != k {
			ws.X[k], ws.X[p] = ws.X[p], ws.X[k]

			origK := ws.P[k]
			origP := ws.P[p]
			ws.P[k], ws.P[p] = origP, origK
			ws.PInv[origP] = k
			ws.PInv[origK] = p

			for j := 0; j < k; j++ {
				lRowJ := ws.lRow[j]
				for idx := 0; idx < len(lRowJ); idx++ {
					if lRowJ[idx] == k {
						lRowJ[idx] = p
					} else if lRowJ[idx] == p {
						lRowJ[idx] = k
					}
				}
			}
		}

		ws.uDiag[k] = ws.X[k]

		invPivot := 1.0 / ws.X[k]
		for i := k + 1; i < n; i++ {
			if ws.X[i] != 0 {
				ws.lRow[k] = append(ws.lRow[k], i)
				ws.lVals[k] = append(ws.lVals[k], ws.X[i]*invPivot)
			}
		}
	}

	ws.lColPtr = ws.lColPtr[:0]
	ws.lRowIdx = ws.lRowIdx[:0]
	ws.lValsFlat = ws.lValsFlat[:0]

	for k := 0; k < n; k++ {
		ws.lColPtr = append(ws.lColPtr, len(ws.lRowIdx))
		ws.lRowIdx = append(ws.lRowIdx, ws.lRow[k]...)
		ws.lValsFlat = append(ws.lValsFlat, ws.lVals[k]...)
	}
	ws.lColPtr = append(ws.lColPtr, len(ws.lRowIdx))

	ws.uColPtr = ws.uColPtr[:0]
	ws.uRowIdx = ws.uRowIdx[:0]
	ws.uValsFlat = ws.uValsFlat[:0]

	for k := 0; k < n; k++ {
		ws.uColPtr = append(ws.uColPtr, len(ws.uRowIdx))
		ws.uRowIdx = append(ws.uRowIdx, ws.uRow[k]...)
		ws.uValsFlat = append(ws.uValsFlat, ws.uVals[k]...)
	}
	ws.uColPtr = append(ws.uColPtr, len(ws.uRowIdx))

	if res.L == nil {
		res.L = &LFactor{}
	}
	if res.U == nil {
		res.U = &UFactor{}
	}

	res.L.n = n
	res.L.ColPtr = append(res.L.ColPtr[:0], ws.lColPtr...)
	res.L.RowIdx = append(res.L.RowIdx[:0], ws.lRowIdx...)
	res.L.Vals = append(res.L.Vals[:0], ws.lValsFlat...)
	res.L.P = append(res.L.P[:0], ws.P...)

	res.U.n = n
	res.U.ColPtr = append(res.U.ColPtr[:0], ws.uColPtr...)
	res.U.RowIdx = append(res.U.RowIdx[:0], ws.uRowIdx...)
	res.U.Vals = append(res.U.Vals[:0], ws.uValsFlat...)
	res.U.Diag = append(res.U.Diag[:0], ws.uDiag...)

	return nil
}

// ==========================================
// 5. Back-Substitution Solvers
// ==========================================

type LFactor struct {
	n      int
	ColPtr []int
	RowIdx []int
	Vals   []float64
	P      []int
}

// SolveLowerTriangular iteratively solves L * y = P * b
func (L *LFactor) SolveLowerTriangular(b []float64) []float64 {
	x := make([]float64, L.n)
	L.SolveLowerTriangularInPlace(b, x)
	return x
}

// SolveLowerTriangularInPlace operates without allocating a slice structure.
func (L *LFactor) SolveLowerTriangularInPlace(b, x []float64) {
	for i := 0; i < L.n; i++ {
		x[i] = b[L.P[i]]
	}

	for j := 0; j < L.n; j++ {
		xj := x[j]
		if xj != 0 {
			for p := L.ColPtr[j]; p < L.ColPtr[j+1]; p++ {
				r := L.RowIdx[p]
				x[r] -= xj * L.Vals[p]
			}
		}
	}
}

type UFactor struct {
	n      int
	ColPtr []int
	RowIdx []int
	Vals   []float64
	Diag   []float64
}

// SolveUpperTriangular iteratively solves U * x = y
func (U *UFactor) SolveUpperTriangular(b []float64) []float64 {
	x := make([]float64, U.n)
	U.SolveUpperTriangularInPlace(b, x)
	return x
}

// SolveUpperTriangularInPlace operates without allocating a slice structure.
func (U *UFactor) SolveUpperTriangularInPlace(b, x []float64) {
	copy(x, b)

	for j := U.n - 1; j >= 0; j-- {
		x[j] /= U.Diag[j]
		xj := x[j]

		if xj != 0 {
			for p := U.ColPtr[j]; p < U.ColPtr[j+1]; p++ {
				r := U.RowIdx[p]
				x[r] -= xj * U.Vals[p]
			}
		}
	}
}
