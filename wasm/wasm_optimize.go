//go:build wasm

package main

import (
	"fmt"
	"math"
	"syscall/js"
	"unsafe"

	"github.com/kikoqiu/golp"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/optimize"
	"gonum.org/v1/gonum/stat"
)

// LinProg_F64 solves a linear programming problem using the custom high-performance solver.
//
// Arguments:
//   - cPtr, cLen: Objective coefficients. (Number of variables n = cLen)
//   - gPtr, gRows: Inequality matrix G (flattened row-major). Size assumed: gRows * cLen.
//   - hPtr: Inequality RHS vector h. Size assumed: gRows.
//   - aPtr, aRows: Equality matrix A (flattened row-major). Size assumed: aRows * cLen.
//   - bPtr: Equality RHS vector b. Size assumed: aRows.
//   - boundsPtr: Interleaved bounds [Low0, High0, Low1, High1...]. Size assumed: 2 * cLen.
//   - xResultPtr: Buffer to store x solution.
//   - objValPtr: Buffer to store objective value.
//   - statusPtr: Buffer to store status code.
//
//go:wasmexport LinProg_F64
func LinProg_F64(
	cPtr unsafe.Pointer, cLen int32,
	gPtr unsafe.Pointer, gRows int32,
	hPtr unsafe.Pointer,
	aPtr unsafe.Pointer, aRows int32,
	bPtr unsafe.Pointer,
	boundsPtr unsafe.Pointer,
	xResultPtr unsafe.Pointer, objValPtr unsafe.Pointer, statusPtr unsafe.Pointer,
) {
	// 1. Convert Pointers to Go Slices
	// Objective C
	cData := ptrToF64Slice(cPtr, cLen)

	// Inequality (G * x <= h)
	// G size is Rows * Cols (Cols == cLen)
	gData := ptrToF64Slice(gPtr, gRows*cLen)
	hData := ptrToF64Slice(hPtr, gRows)

	// Equality (A * x = b)
	// A size is Rows * Cols (Cols == cLen)
	aData := ptrToF64Slice(aPtr, aRows*cLen)
	bData := ptrToF64Slice(bPtr, aRows)

	// Bounds: Interleaved [Low, High, Low, High...]
	// Total length is 2 * cLen
	boundsData := ptrToF64Slice(boundsPtr, 2*cLen)

	// Outputs
	xResultData := ptrToF64Slice(xResultPtr, cLen)
	objValData := ptrToF64Slice(objValPtr, 1)
	statusData := ptrToI32Slice(statusPtr, 1)

	// 2. Construct Bounds Structure
	numVars := int(cLen)
	var bounds []golp.Bounds

	if numVars > 0 {
		bounds = make([]golp.Bounds, numVars)
		// If boundsPtr was nil, boundsData is nil.
		// We can supply defaults if data is missing, or parse if present.

		for i := 0; i < numVars; i++ {
			lower := 0.0
			upper := math.Inf(1)

			if len(boundsData) > 0 {
				lIdx := i * 2
				uIdx := i*2 + 1

				// Read Interleaved Data
				// Handle NaN check if necessary, otherwise assign directly
				valLow := boundsData[lIdx]
				if !math.IsNaN(valLow) {
					lower = valLow
				} else {
					// Fallback for NaN (optional, based on requirement)
					lower = 0.0
				}

				valHigh := boundsData[uIdx]
				if !math.IsNaN(valHigh) {
					upper = valHigh
				} else {
					// Fallback for NaN
					upper = math.Inf(1)
				}
			}

			bounds[i] = golp.Bounds{Lower: lower, Upper: upper}
		}
	}

	// 3. Create Problem Instance
	prob := golp.NewDenseProblem(cData, gData, hData, aData, bData, bounds)

	// 4. Solve
	sol, err := golp.SolveDense(prob)

	// 5. Handle Result
	if err != nil {
		statusData[0] = -1 // Error
		fmt.Printf("LinProg_F64 Error: %v\n", err)
		return
	}

	// Map internal status code to integer codes
	switch sol.StatusCode {
	case golp.StatusOptimal:
		statusData[0] = 0 // Optimal

		// Copy result vector safely
		limit := len(sol.X)
		if len(xResultData) < limit {
			limit = len(xResultData)
		}
		copy(xResultData[:limit], sol.X[:limit])

		objValData[0] = sol.ObjValue

	case golp.StatusInfeasible:
		statusData[0] = 1 // Infeasible
		objValData[0] = 0

	case golp.StatusUnbounded:
		statusData[0] = 2 // Unbounded
		objValData[0] = math.Inf(-1)

	default:
		statusData[0] = -1 // Unknown/Other
	}
}

// --- Linear Regression ---

//go:wasmexport LinearRegression_F64
func LinearRegression_F64(xPtr, yPtr unsafe.Pointer, n int32, alphaPtr, betaPtr unsafe.Pointer) {
	xData := ptrToF64Slice(xPtr, n)
	yData := ptrToF64Slice(yPtr, n)
	alpha, beta := stat.LinearRegression(xData, yData, nil, false)
	ptrToF64Slice(alphaPtr, 1)[0] = alpha
	ptrToF64Slice(betaPtr, 1)[0] = beta
}

// --- Non-Linear Optimization ---

// Minimize_F64 is the entry point exported to WebAssembly.
// It bridges the Gonum optimization library with JavaScript implementations of the objective function.
//
//go:wasmexport Minimize_F64
func Minimize_F64(
	x0Ptr unsafe.Pointer, n int32,
	resultPtr unsafe.Pointer,
	statsPtr unsafe.Pointer,
) {
	defer func() {
		if r := recover(); r != nil {
			ConsoleLog("Recovered in Minimize_F64:", r)
		}
	}()

	// Retrieve the JavaScript global functions.
	// These functions must accept pointers (numbers) and length, not arrays.
	jsFunc := js.Global().Get("ndarray_minimize_func")
	jsGrad := js.Global().Get("ndarray_minimize_grad")

	p := optimize.Problem{
		// Objective Function
		Func: func(x []float64) float64 {
			if len(x) == 0 {
				return 0
			}
			// Zero-Copy Optimization:
			// Instead of copying data to JS, we pass the memory address of the Go slice.
			xPtr := uintptr(unsafe.Pointer(&x[0]))

			// Call JS: func(ptr, length) -> number
			// JS should create a Float64Array view on WASM memory at xPtr.
			res := jsFunc.Invoke(xPtr, len(x))
			return res.Float()
		},
	}

	// Gradient Function (if provided by JS)
	if jsGrad.Truthy() {
		p.Grad = func(grad, x []float64) {
			if len(x) == 0 {
				return
			}
			// Get memory addresses for input (x) and output (grad)
			xPtr := uintptr(unsafe.Pointer(&x[0]))
			gradPtr := uintptr(unsafe.Pointer(&grad[0]))

			// Call JS: func(xPtr, gradPtr, length)
			// JS writes the calculated gradient directly into the memory at gradPtr.
			jsGrad.Invoke(xPtr, gradPtr, len(x))

			// No explicit copy-back (CopyBytesToGo) is needed here.
			// Since JS wrote directly to the memory backing the 'grad' slice,
			// Go sees the changes immediately.
		}
	}

	// Convert the raw pointer x0Ptr to a Go slice for Gonum to use.
	x0 := ptrToF64Slice(x0Ptr, n)

	// Run the optimization
	// Note: 'settings' is nil here, using default settings.
	result, err := optimize.Minimize(p, x0, nil, nil)

	statsData := ptrToF64Slice(statsPtr, 6)
	if err != nil {
		fmt.Printf("Minimize_F64 error: %v\n", err)
		statsData[0] = float64(optimize.Failure)
		return
	}

	// Prepare to write results back to the caller's memory (resultPtr/statsPtr).
	stats := result.Stats

	resultData := ptrToF64Slice(resultPtr, n)
	// Copy the optimized parameters (X) into the result buffer.
	copy(resultData, result.X)

	// Populate statistics.
	statsData[0] = float64(result.Status)
	if result.Status.Early() {
		statsData[0] = statsData[0] * -1
	}
	statsData[1] = result.F
	statsData[2] = float64(stats.MajorIterations)
	statsData[3] = float64(stats.FuncEvaluations)
	statsData[4] = float64(stats.GradEvaluations)
	statsData[5] = float64(stats.Runtime)
}

// --- Polynomial Regression (Polyfit) ---

// Polyfit_F64 fits a polynomial of the specified degree to a set of 2D points using least squares.
// The computed coefficients are returned in ascending order of degree (c0 + c1*x + c2*x^2 + ...).
//
// Arguments:
//   - xPtr, yPtr: Pointers to x and y data arrays.
//   - n: Number of data points.
//   - degree: Degree of the fitting polynomial.
//   - coeffsPtr: Buffer to store the resulting coefficients. Size assumed: degree + 1.
//
//go:wasmexport Polyfit_F64
func Polyfit_F64(xPtr, yPtr unsafe.Pointer, n int32, degree int32, coeffsPtr unsafe.Pointer) {
	xData := ptrToF64Slice(xPtr, n)
	yData := ptrToF64Slice(yPtr, n)
	coeffsData := ptrToF64Slice(coeffsPtr, degree+1)

	rows := int(n)
	cols := int(degree + 1)

	// Create the Vandermonde matrix V where V[i, j] = x[i]^j
	v := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			v.Set(i, j, math.Pow(xData[i], float64(j)))
		}
	}

	// Create the Y vector
	yVec := mat.NewVecDense(rows, yData)

	// Solve the linear least squares problem V * c = Y
	var c mat.VecDense
	err := c.SolveVec(v, yVec)
	if err != nil {
		fmt.Printf("Polyfit_F64 Error: %v\n", err)
		// On error (e.g., singular/ill-conditioned matrix), fallback to NaN
		for i := 0; i < cols; i++ {
			coeffsData[i] = math.NaN()
		}
		return
	}

	// Copy result coefficients back to the WASM memory buffer
	for i := 0; i < cols; i++ {
		coeffsData[i] = c.AtVec(i)
	}
}

func f64SliceToBytes(s []float64) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)*8)
}
