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

	info.T = []float64{t}
	info.Y = [][]float64{cloneSlice(yCurr)}
	info.Dy = [][]float64{}
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

	history := []historyPoint{{t: t, y: cloneSlice(yCurr), f: cloneSlice(f0), M: cloneSlice(M0)}}

	var globalError []float64
	if info.EstimateError {
		globalError = make([]float64, dim)
		info.GlobalErrorHistory = [][]float64{cloneSlice(globalError)}
	}

	updateJacobian := true
	updateLU := true
	stepsSinceJacobian := 0
	lastHBeta := 0.0
	var JM []float64
	var Jf CooMatrix
	var luRes LUResult

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

		var yPred []float64
		var yStar [][]float64
		var DOld [][][]float64

		if len(pts) == 1 {
			yPred = make([]float64, dim)
			mStart := pts[0].M
			fStart := pts[0].f
			for d := 0; d < dim; d++ {
				mVal := 1.0
				if mStart != nil {
					mVal = mStart[d]
				}
				if math.Abs(mVal) <= mZeroTol {
					yPred[d] = pts[0].y[d]
				} else {
					yPred[d] = pts[0].y[d] + (h * fStart[d] / mVal)
				}
			}
			yStar = [][]float64{cloneSlice(pts[0].y)}
		} else {
			DOld = computeDividedDifferences(pts, dim)
			yPred = evalPoly(pts, DOld, tNext, dim)
			yStar = make([][]float64, 0, k)
			for j := 1; j <= k; j++ {
				yStar = append(yStar, evalPoly(pts, DOld, tNext-(h*bdfKArr[j]), dim))
			}
		}

		C := make([]float64, dim)
		for j := 1; j <= k; j++ {
			alphaJ := BDF[k].alpha[j-1]
			for d := 0; d < dim; d++ {
				C[d] += alphaJ * yStar[j-1][d]
			}
		}

		betaK := BDF[k].beta
		hBeta := h * betaK

		if stepsSinceJacobian >= 400 {
			updateJacobian = true
		}

		if updateJacobian {
			resPred := odefun(tNext, yPred)
			fPred := resPred.F
			JM = resPred.M
			Jf = getJacobian(info, odefun, dim, tNext, yPred, fPred, jacobianEps)

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
			rowIdx := make([]int, 0, dim+len(Jf.Vals))
			colIdx := make([]int, 0, dim+len(Jf.Vals))
			vals := make([]float64, 0, dim+len(Jf.Vals))

			for i := 0; i < dim; i++ {
				mVal := 1.0
				if JM != nil {
					mVal = JM[i]
				}
				if mVal != 0 {
					rowIdx = append(rowIdx, i)
					colIdx = append(colIdx, i)
					vals = append(vals, mVal)
				}
			}

			for i := 0; i < len(Jf.Vals); i++ {
				jTerm := -(hBeta * Jf.Vals[i])
				rowIdx = append(rowIdx, Jf.RowIdx[i])
				colIdx = append(colIdx, Jf.ColIdx[i])
				vals = append(vals, jTerm)
			}

			jgSparse, err := NewSparseMatrixCSCFromCOO(dim, dim, rowIdx, colIdx, vals)
			if err != nil {
				h *= 0.2
				info.FailedSteps++
				lastStep = false
				updateJacobian = true
				continue
			}

			luRes, err = jgSparse.LU()
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

		yTmpCurr := cloneSlice(yPred)
		newtonConverged := false
		var MCurr []float64
		var fCurr []float64
		oldDeltaNorm := -1.0

		for iter := 0; iter < 5; iter++ {
			resCurr := odefun(tNext, yTmpCurr)
			MCurr = resCurr.M
			fCurr = resCurr.F

			negG := make([]float64, dim)
			for d := 0; d < dim; d++ {
				yDiff := yTmpCurr[d] - C[d]
				mTerm := yDiff
				if MCurr != nil {
					mTerm = MCurr[d] * yDiff
				}
				gVal := mTerm - (hBeta * fCurr[d])
				negG[d] = -gVal
			}

			yTmp := luRes.L.SolveLowerTriangular(negG)
			deltaY := luRes.U.SolveUpperTriangular(yTmp)

			stepConverged := true
			currentDeltaNorm := 0.0

			for d := 0; d < dim; d++ {
				yTmpCurr[d] += deltaY[d]

				maxY := math.Abs(yCurr[d])
				if !(math.Abs(yTmpCurr[d]) <= maxY) {
					maxY = math.Abs(yTmpCurr[d])
				}
				sc := absTol + (relTol * maxY)

				ratio := math.Abs(deltaY[d]) / sc
				if !(ratio <= currentDeltaNorm) {
					currentDeltaNorm = ratio
				}

				if !(math.Abs(deltaY[d]) <= (sc * newtonTol)) {
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

			errVal := (yTmpCurr[d] - yPred[d]) * invKPlus1
			maxY := math.Abs(yCurr[d])
			if !(math.Abs(yTmpCurr[d]) <= maxY) {
				maxY = math.Abs(yTmpCurr[d])
			}
			sc := absTol + (relTol * maxY)

			ratio := math.Abs(errVal) / sc
			if !(ratio <= LTENorm) {
				LTENorm = ratio // If ratio is NaN, LTENorm becomes NaN (which safely triggers rejection later)
			}
		}

		if LTENorm <= 1.0 {
			if info.EstimateError {
				rhsE := make([]float64, dim)
				for d := 0; d < dim; d++ {
					mVal := 1.0
					if MCurr != nil {
						mVal = MCurr[d]
					}
					if math.Abs(mVal) <= mZeroTol {
						rhsE[d] = globalError[d]
					} else {
						errValTrue := (yTmpCurr[d] - yPred[d]) * invKPlus1
						rhsE[d] = globalError[d] + errValTrue
					}
				}

				eTmp := luRes.L.SolveLowerTriangular(rhsE)
				globalError = luRes.U.SolveUpperTriangular(eTmp)
				info.GlobalErrorHistory = append(info.GlobalErrorHistory, cloneSlice(globalError))
			}

			stepsSinceJacobian++
			t = tNext
			yCurr = yTmpCurr

			resFinal := odefun(t, yCurr)
			fFinal := resFinal.F
			MFinal := resFinal.M

			history = append([]historyPoint{{t: t, y: cloneSlice(yCurr), f: cloneSlice(fFinal), M: cloneSlice(MFinal)}}, history...)
			if len(history) > 7 {
				history = history[:7]
			}

			dyCurr := make([]float64, dim)
			for d := 0; d < dim; d++ {
				dyCurr[d] = (yCurr[d] - C[d]) / hBeta
			}

			info.T = append(info.T, t)
			info.Y = append(info.Y, cloneSlice(yCurr))
			info.Dy = append(info.Dy, dyCurr)

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

			DNew := computeDividedDifferences(history, dim)
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

func computeDividedDifferences(pts []historyPoint, dim int) [][][]float64 {
	m := len(pts) - 1
	D := make([][][]float64, m+1)

	for i := 0; i <= m; i++ {
		D[i] = make([][]float64, 0, m+1-i)
		D[i] = append(D[i], cloneSlice(pts[i].y))
	}

	for j := 1; j <= m; j++ {
		for i := 0; i <= m-j; i++ {
			dx := pts[i].t - pts[i+j].t
			invDx := 1.0 / dx
			diff := make([]float64, dim)
			for d := 0; d < dim; d++ {
				diff[d] = (D[i][j-1][d] - D[i+1][j-1][d]) * invDx
			}
			D[i] = append(D[i], diff)
		}
	}
	return D
}

func evalPoly(pts []historyPoint, D [][][]float64, tTarget float64, dim int) []float64 {
	m := len(pts) - 1
	res := make([]float64, dim)
	term := 1.0
	for j := 0; j <= m; j++ {
		for d := 0; d < dim; d++ {
			res[d] += D[0][j][d] * term
		}
		if j < m {
			term *= (tTarget - pts[j].t)
		}
	}
	return res
}

func getJacobian(info *OdeInfo, odefun OdeFunc, dim int, tVal float64, yVal []float64, fVal []float64, jacobianEps float64) CooMatrix {
	if info.Jacobian != nil {
		return info.Jacobian(tVal, yVal, fVal)
	}

	rowIdx := make([]int, 0)
	colIdx := make([]int, 0)
	vals := make([]float64, 0)

	for j := 0; j < dim; j++ {
		yPert := cloneSlice(yVal)
		delta := math.Abs(yVal[j]) * jacobianEps

		if delta == 0 {
			delta = jacobianEps
		}

		invDelta := 1.0 / delta
		yPert[j] += delta

		resPert := odefun(tVal, yPert)
		fPert := resPert.F

		for i := 0; i < dim; i++ {
			diff := (fPert[i] - fVal[i]) * invDelta
			if diff != 0 {
				rowIdx = append(rowIdx, i)
				colIdx = append(colIdx, j)
				vals = append(vals, diff)
			}
		}
	}

	return CooMatrix{RowIdx: rowIdx, ColIdx: colIdx, Vals: vals}
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
func NewSparseMatrixCSCFromCOO(rows, cols int, rowIdx, colIdx []int, vals []float64) (*CSCMatrix, error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("invalid matrix dimensions: %dx%d", rows, cols)
	}

	colData := make([]map[int]float64, cols)
	for i := 0; i < cols; i++ {
		colData[i] = make(map[int]float64)
	}

	for i := 0; i < len(vals); i++ {
		c := colIdx[i]
		r := rowIdx[i]
		v := vals[i]
		colData[c][r] += v
	}

	colPtr := make([]int, cols+1)
	var rowIdxCSC []int
	var valsCSC []float64

	for c := 0; c < cols; c++ {
		colPtr[c] = len(rowIdxCSC)
		rIdxs := make([]int, 0, len(colData[c]))
		for r := range colData[c] {
			rIdxs = append(rIdxs, r)
		}

		// Bubble sort for small NNZ columns (Highly efficient for standard PDE/ODE banded patterns)
		for i := 0; i < len(rIdxs)-1; i++ {
			for j := i + 1; j < len(rIdxs); j++ {
				if rIdxs[i] > rIdxs[j] {
					rIdxs[i], rIdxs[j] = rIdxs[j], rIdxs[i]
				}
			}
		}

		for _, r := range rIdxs {
			rowIdxCSC = append(rowIdxCSC, r)
			valsCSC = append(valsCSC, colData[c][r])
		}
	}
	colPtr[cols] = len(rowIdxCSC)

	return &CSCMatrix{
		Rows:   rows,
		Cols:   cols,
		ColPtr: colPtr,
		RowIdx: rowIdxCSC,
		Vals:   valsCSC,
	}, nil
}

// LU performs Left-Looking Sparse LU Factorization with Partial Pivoting.
// Utilizes a Sparse Accumulator (SPA) for mathematically rigorous structural singularity avoidance.
func (A *CSCMatrix) LU() (LUResult, error) {
	n := A.Cols
	if n != A.Rows {
		return LUResult{}, fmt.Errorf("LU factorization requires a square matrix")
	}

	lRow := make([][]int, n)
	lVals := make([][]float64, n)

	uRow := make([][]int, n)
	uVals := make([][]float64, n)
	uDiag := make([]float64, n)

	P := make([]int, n)
	PInv := make([]int, n)
	for i := 0; i < n; i++ {
		P[i] = i
		PInv[i] = i
	}

	X := make([]float64, n)

	for k := 0; k < n; k++ {
		for i := 0; i < n; i++ {
			X[i] = 0.0
		}

		for p := A.ColPtr[k]; p < A.ColPtr[k+1]; p++ {
			origR := A.RowIdx[p]
			X[PInv[origR]] = A.Vals[p]
		}

		for j := 0; j < k; j++ {
			if X[j] != 0 {
				uRow[k] = append(uRow[k], j)
				uVals[k] = append(uVals[k], X[j])

				xj := X[j]
				lRowJ := lRow[j]
				lValsJ := lVals[j]

				for idx := 0; idx < len(lRowJ); idx++ {
					r := lRowJ[idx]
					X[r] -= xj * lValsJ[idx]
				}
			}
		}

		pivotVal := 0.0
		p := k
		for i := k; i < n; i++ {
			if !(math.Abs(X[i]) <= math.Abs(pivotVal)) {
				pivotVal = X[i]
				p = i
			}
		}

		if pivotVal == 0.0 {
			return LUResult{}, fmt.Errorf("structural singularity encountered at column %d", k)
		}

		if p != k {
			X[k], X[p] = X[p], X[k]

			origK := P[k]
			origP := P[p]
			P[k], P[p] = origP, origK
			PInv[origP] = k
			PInv[origK] = p

			for j := 0; j < k; j++ {
				lRowJ := lRow[j]
				for idx := 0; idx < len(lRowJ); idx++ {
					if lRowJ[idx] == k {
						lRowJ[idx] = p
					} else if lRowJ[idx] == p {
						lRowJ[idx] = k
					}
				}
			}
		}

		uDiag[k] = X[k]

		invPivot := 1.0 / X[k]
		for i := k + 1; i < n; i++ {
			if X[i] != 0 {
				lRow[k] = append(lRow[k], i)
				lVals[k] = append(lVals[k], X[i]*invPivot)
			}
		}
	}

	lColPtr := make([]int, n+1)
	var lRowIdx []int
	var lValsFlat []float64
	for k := 0; k < n; k++ {
		lColPtr[k] = len(lRowIdx)
		lRowIdx = append(lRowIdx, lRow[k]...)
		lValsFlat = append(lValsFlat, lVals[k]...)
	}
	lColPtr[n] = len(lRowIdx)

	uColPtr := make([]int, n+1)
	var uRowIdx []int
	var uValsFlat []float64
	for k := 0; k < n; k++ {
		uColPtr[k] = len(uRowIdx)
		uRowIdx = append(uRowIdx, uRow[k]...)
		uValsFlat = append(uValsFlat, uVals[k]...)
	}
	uColPtr[n] = len(uRowIdx)

	return LUResult{
		L: &LFactor{n: n, ColPtr: lColPtr, RowIdx: lRowIdx, Vals: lValsFlat, P: P},
		U: &UFactor{n: n, ColPtr: uColPtr, RowIdx: uRowIdx, Vals: uValsFlat, Diag: uDiag},
	}, nil
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
	return x
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
	return x
}
