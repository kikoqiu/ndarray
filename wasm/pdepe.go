package main

import (
	"math"
	"time"
)

// ==========================================
// 1. Data Structures & Callback Signatures
// ==========================================

// PdeFunRes holds the results of the PDE equation function.
type PdeFunRes struct {
	C []float64 // c(x, t, u, dudx)
	F []float64 // f(x, t, u, dudx)
	S []float64 // s(x, t, u, dudx)
}

// PdeFun defines the PDE equation callback: {c, f, s} = pdefun(x, t, u, dudx)
type PdeFun func(x, t float64, u, dudx []float64) PdeFunRes

// IcFun defines the initial conditions callback: u0 = icfun(x)
type IcFun func(x float64) []float64

// BcFunRes holds the results of the Boundary condition function.
type BcFunRes struct {
	Pl []float64
	Ql []float64
	Pr []float64
	Qr []float64
}

// BcFun defines the boundary conditions callback: {pl, ql, pr, qr} = bcfun(xl, ul, xr, ur, t)
type BcFun func(xl float64, ul []float64, xr float64, ur []float64, t float64) BcFunRes

// ==========================================
// pdeWorkspace handles all pre-allocated memory slices internally
// to guarantee zero-allocation during the intense PDE MOL integration loop.
// ==========================================
type pdeWorkspace struct {
	dY       []float64
	M        []float64
	F_mid    [][]float64
	C_node   [][]float64
	S_node   [][]float64
	uMid     []float64
	dudxMid  []float64
	dudxNode []float64

	// Jacobian perturbation arrays
	yPert    []float64
	deltas   []float64
	fValBase []float64
	rowIdx   []int
	colIdx   []int
	vals     []float64
}

func newPdeWorkspace(N, D int) *pdeWorkspace {
	return &pdeWorkspace{
		dY:       make([]float64, N*D),
		M:        make([]float64, N*D),
		F_mid:    make([][]float64, N-1),
		C_node:   make([][]float64, N),
		S_node:   make([][]float64, N),
		uMid:     make([]float64, D),
		dudxMid:  make([]float64, D),
		dudxNode: make([]float64, D),

		yPert:    make([]float64, N*D),
		deltas:   make([]float64, N),
		fValBase: make([]float64, N*D),
		rowIdx:   make([]int, 0, N*D*3),
		colIdx:   make([]int, 0, N*D*3),
		vals:     make([]float64, 0, N*D*3),
	}
}

// ==========================================
// 2. High-Performance 1D PDE Solver
// ==========================================

// Pdepe is a High-performance 1D Parabolic and Elliptic PDE Solver.
// Solves equations of the form:
// c(x,t,u,Du/Dx) * Du/Dt = x^(-m) * D/Dx( x^m * f(x,t,u,Du/Dx) ) + s(x,t,u,Du/Dx)
//
// Parameters:
//
//	m: Symmetry parameter: 0 (slab), 1 (cylinder), 2 (sphere).
//	pdefun: Equation definitions callback.
//	icfun: Initial conditions callback.
//	bcfun: Boundary conditions callback.
//	xmesh: Spatial grid points [x_0, x_1, ..., x_N].
//	tspan: Time output points[t_0, t_1, ..., t_M].
//	info: Configuration and Status object forwarded to Ode15s.
//
// Returns a 3D slice[time_index][spatial_index][equation_index].
func Pdepe(m int, pdefun PdeFun, icfun IcFun, bcfun BcFun, xmesh []float64, tspan []float64, info *OdeInfo) [][][]float64 {
	mPlus1 := float64(m + 1)

	N := len(xmesh)
	if N < 3 {
		panic("pdepe: xmesh must contain at least 3 spatial points.")
	}

	// Dynamic Pre-calculated Tolerances & Global Constants Hoisting mapping typical 53-bit precision
	eps := math.Nextafter(1.0, 2.0) - 1.0
	hMinTol := 1e4 * eps
	zeroTol := 1e-13
	qZeroTol := 1e-12 // Dirichlet BC evaluation threshold

	// Check if origin symmetry mechanism is triggered (Cylinder & Sphere at x=0)
	isSymLeft := m > 0 && xmesh[0] == 0.0

	// Geometry caching to prevent inner-loop overhead
	Xmid := make([]float64, N-1)
	dx := make([]float64, N-1)
	preInvDx := make([]float64, N-1)

	for i := 0; i < N-1; i++ {
		Xmid[i] = (xmesh[i] + xmesh[i+1]) * 0.5
		dx[i] = xmesh[i+1] - xmesh[i]
		preInvDx[i] = 1.0 / dx[i]
	}

	preInvDx2 := make([]float64, N-1)
	for i := 1; i < N-1; i++ {
		preInvDx2[i] = 1.0 / (dx[i-1] + dx[i])
	}

	// Geometry Exponents mapped for symmetry parameter (m)
	powM := func(x float64) float64 {
		if m == 0 {
			return 1.0
		}
		if m == 1 {
			return x
		}
		if m == 2 {
			return x * x
		}
		return math.Pow(x, float64(m))
	}

	powMp1 := func(x float64) float64 {
		if m == 0 {
			return x
		}
		if m == 1 {
			return x * x
		}
		return math.Pow(x, float64(m+1))
	}

	powMX := make([]float64, N)
	for i := 0; i < N; i++ {
		powMX[i] = powM(xmesh[i])
	}

	powMXmid := make([]float64, N-1)
	for i := 0; i < N-1; i++ {
		powMXmid[i] = powM(Xmid[i])
	}

	// Control Volume Formulations (Protects against origin singularities when m > 0)
	V := make([]float64, N)
	preInvV := make([]float64, N)
	for i := 0; i < N; i++ {
		leftEdge := xmesh[0]
		if i > 0 {
			leftEdge = Xmid[i-1]
		}
		rightEdge := xmesh[N-1]
		if i < N-1 {
			rightEdge = Xmid[i]
		}
		V[i] = (powMp1(rightEdge) - powMp1(leftEdge)) / mPlus1
		preInvV[i] = 1.0 / V[i]
	}

	// Extract Initial State and Infer Equation Dimensions (D) optimized initialization
	u0Res0 := icfun(xmesh[0])
	D := len(u0Res0)
	U0Flat := make([]float64, 0, N*D)
	U0Flat = append(U0Flat, u0Res0...)
	for i := 1; i < N; i++ {
		U0Flat = append(U0Flat, icfun(xmesh[i])...)
	}

	getU := func(Y []float64, i int) []float64 {
		return Y[i*D : (i+1)*D]
	}

	// Pre-allocate MOL workspace for Zero-Allocation loop
	ws := newPdeWorkspace(N, D)

	// 3. Method of Lines (MOL) Core Assembly with Differential-Algebraic (DAE) Integration Map
	odeSys := func(t float64, Y []float64) OdeRes {
		// a. Compute Interface Fluxes (F_mid)
		for i := 0; i < N-1; i++ {
			uL := getU(Y, i)
			uR := getU(Y, i+1)

			for d := 0; d < D; d++ {
				ws.uMid[d] = (uL[d] + uR[d]) * 0.5
				ws.dudxMid[d] = (uR[d] - uL[d]) * preInvDx[i]
			}
			resMid := pdefun(Xmid[i], t, ws.uMid, ws.dudxMid)
			ws.F_mid[i] = resMid.F
		}

		// b. Point-wise Node Properties and Interior PDE Formulation
		for i := 0; i < N; i++ {
			uNode := getU(Y, i)

			if i == 0 {
				uR := getU(Y, 1)
				for d := 0; d < D; d++ {
					ws.dudxNode[d] = (uR[d] - uNode[d]) * preInvDx[0]
				}
			} else if i == N-1 {
				uL := getU(Y, N-2)
				for d := 0; d < D; d++ {
					ws.dudxNode[d] = (uNode[d] - uL[d]) * preInvDx[N-2]
				}
			} else {
				uL := getU(Y, i-1)
				uR := getU(Y, i+1)
				for d := 0; d < D; d++ {
					ws.dudxNode[d] = (uR[d] - uL[d]) * preInvDx2[i]
				}
			}

			pdeRes := pdefun(xmesh[i], t, uNode, ws.dudxNode)
			ws.C_node[i] = pdeRes.C
			ws.S_node[i] = pdeRes.S

			// Compute Interior Points
			if i > 0 && i < N-1 {
				for d := 0; d < D; d++ {
					fluxR := powMXmid[i] * ws.F_mid[i][d]
					fluxL := powMXmid[i-1] * ws.F_mid[i-1][d]
					fluxDiff := (fluxR - fluxL) * preInvV[i]

					cVal := ws.C_node[i][d]
					// Regularize values close to zero, pass NaN through explicitly
					if !math.IsNaN(cVal) {
						if math.Abs(cVal) < zeroTol {
							if cVal >= 0 {
								cVal = zeroTol
							} else {
								cVal = -zeroTol
							}
						}
					}

					ws.M[i*D+d] = cVal
					ws.dY[i*D+d] = fluxDiff + ws.S_node[i][d]
				}
			}
		}

		// c. Boundary Conditions Blending
		bcRes := bcfun(xmesh[0], getU(Y, 0), xmesh[N-1], getU(Y, N-1), t)

		// -- Left Boundary --
		if isSymLeft {
			for d := 0; d < D; d++ {
				// Zero Flux enforced for origin in cylinder/sphere mappings
				fluxR := powMXmid[0] * ws.F_mid[0][d]
				fluxDiff := fluxR * preInvV[0] // fluxL = 0

				cVal := ws.C_node[0][d]
				if !math.IsNaN(cVal) {
					if math.Abs(cVal) < zeroTol {
						if cVal >= 0 {
							cVal = zeroTol
						} else {
							cVal = -zeroTol
						}
					}
				}

				ws.M[d] = cVal
				ws.dY[d] = fluxDiff + ws.S_node[0][d]
			}
		} else {
			for d := 0; d < D; d++ {
				if !(math.Abs(bcRes.Ql[d]) > qZeroTol) { // Catches NaN effectively
					// Exact Algebraic Dirichlet Constraints -> M(d)=0
					ws.M[d] = 0.0
					ws.dY[d] = bcRes.Pl[d]
				} else {
					fL := -bcRes.Pl[d] / bcRes.Ql[d]
					fluxR := powMXmid[0] * ws.F_mid[0][d]
					fluxL := powMX[0] * fL
					fluxDiff := (fluxR - fluxL) * preInvV[0]

					cVal := ws.C_node[0][d]
					if !math.IsNaN(cVal) {
						if math.Abs(cVal) < zeroTol {
							if cVal >= 0 {
								cVal = zeroTol
							} else {
								cVal = -zeroTol
							}
						}
					}

					ws.M[d] = cVal
					ws.dY[d] = fluxDiff + ws.S_node[0][d]
				}
			}
		}

		// -- Right Boundary --
		offset := (N - 1) * D
		for d := 0; d < D; d++ {
			if math.Abs(bcRes.Qr[d]) <= qZeroTol {
				// Exact Algebraic Dirichlet Constraints -> M(d)=0
				ws.M[offset+d] = 0.0
				ws.dY[offset+d] = bcRes.Pr[d]
			} else {
				fR := -bcRes.Pr[d] / bcRes.Qr[d]
				fluxR := powMX[N-1] * fR
				fluxL := powMXmid[N-2] * ws.F_mid[N-2][d]
				fluxDiff := (fluxR - fluxL) * preInvV[N-1]

				cVal := ws.C_node[N-1][d]
				if !math.IsNaN(cVal) {
					if math.Abs(cVal) < zeroTol {
						if cVal >= 0 {
							cVal = zeroTol
						} else {
							cVal = -zeroTol
						}
					}
				}

				ws.M[offset+d] = cVal
				ws.dY[offset+d] = fluxDiff + ws.S_node[N-1][d]
			}
		}

		return OdeRes{M: ws.M, F: ws.dY}
	}

	// 4. Stiff ODE Global Integration setup
	if info == nil {
		info = &OdeInfo{}
	}
	if info.AbsTol <= 0 {
		info.AbsTol = 1e-5
	}
	if info.RelTol <= 0 {
		info.RelTol = 1e-4
	}
	if info.MaxStep <= 0 {
		info.MaxStep = 10000000
	}
	if info.MaxTime <= 0 {
		info.MaxTime = 10000000 * time.Millisecond
	}

	// Inject custom high-performance graph-colored sparse Jacobian algorithm for MOL PDE
	// Reduces O(N^2) evaluation scaling directly down to O(1) via 3-color structural perturbation
	if info.Jacobian == nil {
		jacobianEps := math.Sqrt(eps)

		info.Jacobian = func(tVal float64, yVal []float64, fVal []float64) CooMatrix {
			ws.rowIdx = ws.rowIdx[:0]
			ws.colIdx = ws.colIdx[:0]
			ws.vals = ws.vals[:0]

			// Anti-aliasing protection: fVal is backed by ws.dY which will be overwritten by odeSys.
			copy(ws.fValBase, fVal)
			copy(ws.yPert, yVal)

			for color := 0; color < 3; color++ {
				for d := 0; d < D; d++ {
					hasPert := false

					// Perturb all independent structural blocks simultaneously
					for i := color; i < N; i += 3 {
						j := i*D + d
						delta := math.Abs(yVal[j]) * jacobianEps
						if delta < jacobianEps {
							delta = jacobianEps
						}
						ws.deltas[i] = delta
						ws.yPert[j] += delta
						hasPert = true
					}
					if !hasPert {
						continue
					}

					resPert := odeSys(tVal, ws.yPert)
					fPert := resPert.F

					for i := color; i < N; i += 3 {
						j := i*D + d
						invDelta := 1.0 / ws.deltas[i]

						// Limit dependent residual checks to mathematically adjacent physical nodes only
						startNode := i - 1
						if startNode < 0 {
							startNode = 0
						}
						endNode := i + 1
						if endNode > N-1 {
							endNode = N - 1
						}

						for node := startNode; node <= endNode; node++ {
							for dAff := 0; dAff < D; dAff++ {
								r := node*D + dAff
								diff := (fPert[r] - ws.fValBase[r]) * invDelta
								if diff != 0 {
									ws.rowIdx = append(ws.rowIdx, r)
									ws.colIdx = append(ws.colIdx, j)
									ws.vals = append(ws.vals, diff)
								}
							}
						}
						// Direct local restore completely avoids slice cloning
						ws.yPert[j] = yVal[j]
					}
				}
			}
			return CooMatrix{RowIdx: ws.rowIdx, ColIdx: ws.colIdx, Vals: ws.vals}
		}
	}

	// Execute full span sweep
	tspanOde := [2]float64{tspan[0], tspan[len(tspan)-1]}
	res := Ode15s(odeSys, tspanOde, U0Flat, info)
	info.Jacobian = nil // Clean up Jacobian reference to prevent memory leaks

	if res == nil {
		ConsoleLog("pdepe: Underlying ode15s integration failed catastrophically: " + info.Status)
		return nil
	}

	odeT := res.T
	odeY := res.Y
	odeDy := res.Dy

	tSpanDir := 1.0
	if tspan[len(tspan)-1] < tspan[0] {
		tSpanDir = -1.0
	}

	// 5. Piecewise Cubic Hermite Interpolation (Continuous dense output exactly matching tspan)
	// Block Memory Allocation for maximal CPU Cache optimization (avoids loop allocations entirely)
	totalNodes := len(tspan) * N
	totalEqs := totalNodes * D

	sol := make([][][]float64, len(tspan))
	gridOutBlock := make([][]float64, totalNodes)
	eqOutBlock := make([]float64, totalEqs)
	stateAtTs := make([]float64, N*D)

	idx := 0
	k := 0

	for tIdx, ts := range tspan {
		for k < len(odeT)-2 && (odeT[k+1]-ts)*tSpanDir < 0 {
			k++
		}

		if k+1 >= len(odeT) {
			copy(stateAtTs, odeY[k])
		} else {
			t0 := odeT[k]
			t1 := odeT[k+1]
			y0 := odeY[k]
			y1 := odeY[k+1]
			dy0 := odeDy[k]
			dy1 := odeDy[k+1]

			h := t1 - t0

			if ts == t0 || math.Abs(h) <= hMinTol {
				copy(stateAtTs, y0)
			} else if ts == t1 {
				copy(stateAtTs, y1)
			} else {
				// High-fidelity continuous spline evaluation mimicking interior ODE structural dynamics
				s := (ts - t0) / h
				s2 := s * s
				s3 := s2 * s

				h00 := 1.0 - 3.0*s2 + 2.0*s3
				h01 := 3.0*s2 - 2.0*s3
				h10 := h * (s - 2.0*s2 + s3)
				h11 := h * (s3 - s2)

				for j := 0; j < N*D; j++ {
					stateAtTs[j] = h00*y0[j] + h01*y1[j] + h10*dy0[j] + h11*dy1[j]
				}
			}
		}

		// 6. Formatting output mapping sub-slices sequentially over contiguous memory blocks
		gridOut := gridOutBlock[tIdx*N : (tIdx+1)*N]
		for i := 0; i < N; i++ {
			eqOut := eqOutBlock[idx*D : (idx+1)*D]
			for d := 0; d < D; d++ {
				eqOut[d] = stateAtTs[i*D+d]
			}
			gridOut[i] = eqOut
			idx++
		}
		sol[tIdx] = gridOut
	}

	return sol
}
