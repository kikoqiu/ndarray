//go:build wasm

package main

import (
	"syscall/js"
	"time"
	"unsafe"
)

// Ode_Solve_F64 is the bridge for both Ode45 and Ode15s.
// It uses a config buffer to pass parameters and a stats buffer to return results/pointers.
//
//go:wasmexport Ode_Solve_F64
func Ode_Solve_F64(
	y0Ptr unsafe.Pointer, dim int32,
	tspanPtr unsafe.Pointer,
	configPtr unsafe.Pointer, //[absTol, relTol, initialStep, method(45=0, 15s=1)]
	statsPtr unsafe.Pointer, // [status, steps, runtime, tPtr, yPtr, dyPtr, length]
	hasM int32, // Indicates if Mass Matrix is used (for DAE systems)
) {
	y0 := ptrToF64Slice(y0Ptr, int32(dim))
	tspanRaw := ptrToF64Slice(tspanPtr, 2)
	tspan := [2]float64{tspanRaw[0], tspanRaw[1]}
	config := ptrToF64Slice(configPtr, 6)
	stats := ptrToF64Slice(statsPtr, 7)

	// Retrieve JS callbacks from global scope
	jsOdeFun := js.Global().Get("ndarray_ode_func")
	jsJacFun := js.Global().Get("ndarray_ode_jac")

	// We pass pointers to Go-allocated slices for F and M to JS.
	// JS will write directly into these.
	fRes := make([]float64, len(y0))
	fPtr := uintptr(unsafe.Pointer(&fRes[0]))
	var mPtr uintptr = uintptr(0)
	var mRes []float64 = nil
	if hasM == 1 {
		mRes = make([]float64, len(y0))
		mPtr = uintptr(unsafe.Pointer(&mRes[0]))
	}
	resOdefun := OdeRes{F: fRes, M: mRes}

	// Define Go-compatible OdeFunc (Performance Optimized)
	odefun := func(t float64, y []float64) OdeRes {
		yPtr := uintptr(unsafe.Pointer(&y[0]))

		// Call JS: odefun(t, yPtr, fPtr, mPtr, length)
		jsOdeFun.Invoke(t, yPtr, fPtr, mPtr, len(y))

		return resOdefun
	}

	info := &OdeInfo{
		AbsTol:      config[0],
		RelTol:      config[1],
		InitialStep: config[2],
		MaxStep:     int(config[4]),
		MaxTime:     time.Duration(config[5]) * time.Millisecond,
	}

	// Attach Jacobian if Ode15s and provided
	if config[3] == 1 && jsJacFun.Truthy() {
		// Pre-allocate zero-copy buffers for the maximum possible non-zeros (dense matrix size).
		maxNnz := int(dim * dim)

		// index array format: [row0, col0, row1, col1, ...] (int32 limits JS BigInt overhead)
		jacIndex := make([]int32, maxNnz*2)
		jacVals := make([]float64, maxNnz)

		jacIndexPtr := uintptr(unsafe.Pointer(&jacIndex[0]))
		jacValsPtr := uintptr(unsafe.Pointer(&jacVals[0]))

		// Internal integer buffers for the actual CooMatrix representation
		jacCooRow := make([]int, maxNnz)
		jacCooCol := make([]int, maxNnz)

		info.Jacobian = func(t float64, y []float64, f []float64) CooMatrix {
			yPtr := uintptr(unsafe.Pointer(&y[0]))
			fPtr := uintptr(unsafe.Pointer(&f[0]))

			// Call JS: jac(t, yPtr, fPtr, indexPtr, valPtr, dim)
			// The JS function must return the actual number of non-zeros (nnz).
			nnzJs := jsJacFun.Invoke(t, yPtr, fPtr, jacIndexPtr, jacValsPtr, len(y)).Int()

			// Fast mapping from flattened 1D int32 pair-array back to independent int slices
			for i := 0; i < nnzJs; i++ {
				jacCooRow[i] = int(jacIndex[i*2])
				jacCooCol[i] = int(jacIndex[i*2+1])
			}

			return CooMatrix{
				RowIdx: jacCooRow[:nnzJs],
				ColIdx: jacCooCol[:nnzJs],
				Vals:   jacVals[:nnzJs],
			}
		}
	}

	var res *OdeResult
	startTime := time.Now()

	// Dispatch based on method
	if config[3] == 0 {
		res = Ode45(odefun, tspan, y0, info)
	} else {
		res = Ode15s(odefun, tspan, y0, info)
	}

	if res == nil {
		if info.Status == "underflow" {
			stats[0] = -4
		} else if info.Status == "timeout" {
			stats[0] = -3
		} else if info.Status == "max_steps" {
			stats[0] = -2
		} else {
			stats[0] = -1 // General error
		}
		return
	}

	// Instead of copying the whole result matrix to JS, we pass the
	// pointers of the internal slices directly to the JS layer.

	// Flatten Y and Dy if needed, or pass pointer to the first element if contiguous.
	// For info.Y ([][]float64), we create a flat view for the JS layer.
	totalSteps := len(res.T)
	flatY := make([]float64, totalSteps*int(dim))
	flatDy := make([]float64, totalSteps*int(dim))
	for i := 0; i < totalSteps; i++ {
		copy(flatY[i*int(dim):], res.Y[i])
		copy(flatDy[i*int(dim):], res.Dy[i])
	}

	stats[0] = 1                                             // status: success
	stats[1] = float64(info.Steps)                           // steps
	stats[2] = float64(time.Since(startTime).Milliseconds()) // runtime
	stats[3] = float64(uintptr(unsafe.Pointer(&res.T[0])))   // Pointer to T array
	stats[4] = float64(uintptr(unsafe.Pointer(&flatY[0])))   // Pointer to flattened Y
	stats[5] = float64(uintptr(unsafe.Pointer(&flatDy[0])))  // Pointer to flattened Dy
	stats[6] = float64(totalSteps)                           // Number of time points
}

// Pdepe_Solve_F64 is the zero-copy bridge for the PDEPE Method of Lines (MOL) solver.
//
//go:wasmexport Pdepe_Solve_F64
func Pdepe_Solve_F64(
	m int32, // Symmetry parameter
	xmeshPtr unsafe.Pointer, xLen int32,
	tspanPtr unsafe.Pointer, tLen int32,
	dim int32, // Equation dimensions (D)
	configPtr unsafe.Pointer, //[absTol, relTol, maxStep, maxTime, estimateError]
	statsPtr unsafe.Pointer, // [status, runtime, solPtr]
	errorPtr unsafe.Pointer, // Pointer to error estimate array (if requested)
) {
	xmesh := ptrToF64Slice(xmeshPtr, xLen)
	tspan := ptrToF64Slice(tspanPtr, tLen)
	config := ptrToF64Slice(configPtr, 5)
	stats := ptrToF64Slice(statsPtr, 3)

	jsPdeFun := js.Global().Get("ndarray_pde_fun")
	jsIcFun := js.Global().Get("ndarray_ic_fun")
	jsBcFun := js.Global().Get("ndarray_bc_fun")

	// ==========================================
	// 1. Initial Condition Callback (Called N times)
	// ==========================================
	icfun := func(x float64) []float64 {
		res := make([]float64, dim)
		ptr := uintptr(unsafe.Pointer(&res[0]))
		// JS writes directly to 'res' via WASM memory
		jsIcFun.Invoke(x, float64(ptr), dim)
		return res
	}

	// ==========================================
	// 2. PDE Equation Callback (Extreme High Frequency)
	// Zero-Allocation Ring Buffer implementation
	// ==========================================
	// During a single RHS evaluation, pdefun is called approx 2*N times.
	// We allocate a pool large enough for the entire grid sweep to prevent any GC pressure.
	poolSize := int(xLen) * 2
	cPool := make([]float64, poolSize*int(dim))
	fPool := make([]float64, poolSize*int(dim))
	sPool := make([]float64, poolSize*int(dim))
	callIdx := 0

	pdefun := func(x, t float64, u, dudx []float64) PdeFunRes {
		if callIdx >= poolSize {
			callIdx = 0 // Wrap around for next MOL evaluation step
		}

		start := callIdx * int(dim)
		end := start + int(dim)
		cSlice := cPool[start:end]
		fSlice := fPool[start:end]
		sSlice := sPool[start:end]

		uPtr := uintptr(unsafe.Pointer(&u[0]))
		dudxPtr := uintptr(unsafe.Pointer(&dudx[0]))
		cPtr := uintptr(unsafe.Pointer(&cSlice[0]))
		fPtr := uintptr(unsafe.Pointer(&fSlice[0]))
		sPtr := uintptr(unsafe.Pointer(&sSlice[0]))

		jsPdeFun.Invoke(x, t, float64(uPtr), float64(dudxPtr), float64(cPtr), float64(fPtr), float64(sPtr), dim)
		callIdx++

		return PdeFunRes{C: cSlice, F: fSlice, S: sSlice}
	}

	// ==========================================
	// 3. Boundary Condition Callback
	// ==========================================
	plRes := make([]float64, dim)
	qlRes := make([]float64, dim)
	prRes := make([]float64, dim)
	qrRes := make([]float64, dim)

	plPtr := float64(uintptr(unsafe.Pointer(&plRes[0])))
	qlPtr := float64(uintptr(unsafe.Pointer(&qlRes[0])))
	prPtr := float64(uintptr(unsafe.Pointer(&prRes[0])))
	qrPtr := float64(uintptr(unsafe.Pointer(&qrRes[0])))

	bcfun := func(xl float64, ul []float64, xr float64, ur []float64, t float64) BcFunRes {
		ulPtr := float64(uintptr(unsafe.Pointer(&ul[0])))
		urPtr := float64(uintptr(unsafe.Pointer(&ur[0])))

		jsBcFun.Invoke(xl, ulPtr, xr, urPtr, t, plPtr, qlPtr, prPtr, qrPtr, dim)

		return BcFunRes{Pl: plRes, Ql: qlRes, Pr: prRes, Qr: qrRes}
	}

	// ==========================================
	// 4. Execute PDEPE Solver
	// ==========================================
	info := &OdeInfo{
		AbsTol:        config[0],
		RelTol:        config[1],
		MaxStep:       int(config[2]),
		MaxTime:       time.Duration(config[3]) * time.Millisecond,
		EstimateError: config[4] != 0,
	}

	startTime := time.Now()
	resSol := Pdepe(int(m), pdefun, icfun, bcfun, xmesh, tspan, info)

	if resSol == nil {
		if info.Status == "underflow" {
			stats[0] = -4
		} else if info.Status == "timeout" {
			stats[0] = -3
		} else if info.Status == "max_steps" {
			stats[0] = -2
		} else {
			stats[0] = -1 // General error
		}
		return
	}

	// ==========================================
	// 5. Tensor Flattening
	// ==========================================
	// resSol is [tLen][xLen][dim]. We flatten it to a 1D slice.
	totalSize := int(tLen * xLen * dim)
	flatSol := make([]float64, totalSize)
	idx := 0

	for i := 0; i < int(tLen); i++ {
		for j := 0; j < int(xLen); j++ {
			copy(flatSol[idx:idx+int(dim)], resSol[i][j])
			idx += int(dim)
		}
	}
	if info.EstimateError && info.GlobalError != nil && errorPtr != nil {
		errorSlice := ptrToF64Slice(errorPtr, int32(xLen*dim))
		// If error estimates were requested, we also need to flatten that tensor.
		copy(errorSlice, info.GlobalError)
	}

	stats[0] = 1                                             // Status Success
	stats[1] = float64(time.Since(startTime).Milliseconds()) // Runtime (ms)
	stats[2] = float64(uintptr(unsafe.Pointer(&flatSol[0]))) // Pointer to Flat Sol Tensor
}
