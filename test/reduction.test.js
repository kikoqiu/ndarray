const ndarray = require('../dist/ndarray.cjs');

// Helper function to get a column from a 2D nested array
const getCol = (matrix, col) => matrix.map(row => row[col]);

describe('Reduction Operations', () => {
    // 2D Array: [[0,1,2,3], [4,5,6,7], [8,9,10,11]]
    const arr2D = ndarray.arange(12).reshape([3, 4]);
    // 3D Array
    const arr3D = ndarray.arange(24).reshape([2, 3, 4]);

    describe('sum()', () => {
        test('computes global sum', () => {
            const expected = (12 * 11) / 2; // Sum of 0..11
            expect(arr2D.sum()).toBe(expected);
        });
        test('computes sum along axis 0 (columns) in 2D', () => {
            const result = arr2D.sum(0);
            // Expected: [0+4+8, 1+5+9, 2+6+10, 3+7+11]
            expect(result.copy().data).toEqual(new Float64Array([12, 15, 18, 21]));
        });
        test('computes sum along axis 1 (rows) in 2D', () => {
            const result = arr2D.sum(1);
            // Expected: [0+1+2+3, 4+5+6+7, 8+9+10+11]
            expect(result.copy().data).toEqual(new Float64Array([6, 22, 38]));
        });
        test('computes sum correctly on a non-contiguous (transposed) view', () => {
            const t = arr2D.transpose(); // shape [4,3]
            expect(t.sum(0).copy().data).toEqual(new Float64Array([6, 22, 38]));
            expect(t.sum(1).copy().data).toEqual(new Float64Array([12, 15, 18, 21]));
        });
        test('computes sum along axis -1 (last axis) in 3D', () => {
            const result = arr3D.sum(-1); // same as axis=2
            const expected = new Float64Array([6, 22, 38, 54, 70, 86]);
            expect(result.copy().data).toEqual(expected);
        });
    });

    describe('mean()', () => {
        test('computes global mean', () => {
            const expected = ((12 * 11) / 2) / 12;
            expect(arr2D.mean()).toBe(expected);
        });
        test('computes mean along axis 0 (columns) in 2D', () => {
            const result = arr2D.mean(0);
            // Expected: [(0+4+8)/3, (1+5+9)/3, ...]
            expect(result.copy().data).toEqual(new Float64Array([4, 5, 6, 7]));
        });
        test('computes mean along axis 1 (rows) in 2D', () => {
            const result = arr2D.mean(1);
            // Expected: [(0+1+2+3)/4, ...]
            expect(result.copy().data).toEqual(new Float64Array([1.5, 5.5, 9.5]));
        });
    });

    describe('max()', () => {
        test('computes global max', () => {
            expect(arr2D.max()).toBe(11);
        });
        test('computes max along axis 1 (rows) in 2D', () => {
            const maxResult = arr2D.max(1); // [3, 7, 11]
            expect(maxResult.copy().data).toEqual(new Float64Array([3, 7, 11]));
        });
        test('computes max along axis 0 on a 3D array', () => {
            const maxResult = arr3D.max(0); // Shape [3,4]
            // Expected: for each [i,j], max(arr3D(0,i,j), arr3D(1,i,j))
            // e.g., max(0, 12)=12, max(1,13)=13
            const expected = ndarray.arange(12, 24).reshape([3, 4]);
            expect(maxResult.copy().data).toEqual(expected.data);
        });
    });
    
    describe('min()', () => {
        test('computes global min', () => {
            const arr = ndarray.array([5, 2, 8, 1, 9]);
            expect(arr.min()).toBe(1);
        });
        test('computes min along axis 1 (rows) in 2D', () => {
            const minResult = arr2D.min(1); // [0, 4, 8]
            expect(minResult.copy().data).toEqual(new Float64Array([0, 4, 8]));
        });
    });
    
    describe('var() & std()', () => {
        // Use a simpler array for easier manual calculation
        const simpleArr = ndarray.array([1, 2, 3, 4, 5]);
        
        test('computes global variance and standard deviation', () => {
            // mean = 3
            // var = ((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 5
            //     = (4 + 1 + 0 + 1 + 4) / 5 = 10 / 5 = 2
            const variance = simpleArr.var();
            const stddev = simpleArr.std();
            expect(variance).toBeCloseTo(2);
            expect(stddev).toBeCloseTo(Math.sqrt(2));
        });

        test('computes variance along an axis (2D)', () => {
            const a = ndarray.array([[1, 2], [3, 4]]);
            // axis=0: mean is [2, 3]
            // var col 0: ((1-2)^2 + (3-2)^2) / 2 = (1+1)/2 = 1
            // var col 1: ((2-3)^2 + (4-3)^2) / 2 = (1+1)/2 = 1
            const var0 = a.var(0);
            expect(var0.copy().data).toEqual(new Float64Array([1, 1]));

            const var1 = a.var(1);
            expect(var1.copy().data).toEqual(new Float64Array([0.25, 0.25]));
        });

        test('computes standard deviation along an axis (2D)', () => {
            const a = ndarray.array([[1, 2], [3, 4]]);
            const std1 = a.std(1); 
            expect(std1.copy().data).toEqual(new Float64Array([0.5, 0.5]));
        });
    });
});



describe('NDArray Slicing and Reduction Integration', () => {
    let arr1D, arr2D, arr3D;

    beforeEach(() => {
        // [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        arr1D = ndarray.arange(10); 
        // 3x4: [[0,1,2,3], [4,5,6,7], [8,9,10,11]]
        arr2D = ndarray.arange(12).reshape([3, 4]);
        // 2x3x4
        arr3D = ndarray.arange(24).reshape([2, 3, 4]);
    });

    // --- 1D Slices ---
    test('1. 1D Slice: sum of subset [2:7]', () => {
        const view = arr1D.slice([2, 7]); // [2, 3, 4, 5, 6]
        expect(view.sum()).toBe(20);
    });

    test('2. 1D Slice: mean of strided slice [::2]', () => {
        const view = arr1D.slice([null, null, 2]); // [0, 2, 4, 6, 8]
        expect(view.mean()).toBe(4);
    });

    test('3. 1D Slice: max of reversed slice [::-1]', () => {
        const view = arr1D.slice([null, null, -1]);
        expect(view.max()).toBe(9);
    });

    test('4. 1D Slice: var of negative step subset [5:2:-1]', () => {
        const view = arr1D.slice([5, 2, -1]); // [5, 4, 3]
        // mean=4, var=((5-4)^2 + 0 + (3-4)^2)/3 = 2/3
        expect(view.var()).toBeCloseTo(2/3);
    });

    // --- 2D Row/Column Views ---
    test('5. rowview: sum of a single row', () => {
        const view = arr2D.rowview(1); // [4, 5, 6, 7]
        expect(view.sum()).toBe(22);
    });

    test('6. colview: mean of a single column', () => {
        const view = arr2D.colview(2); // [2, 6, 10]
        expect(view.mean()).toBe(6);
    });

    test('7. colview: max of a reversed column slice', () => {
        const view = arr2D.slice([null, null, -1], 0); // column 0 reversed: [8, 4, 0]
        expect(view.max()).toBe(8);
    });

    // --- 2D Sub-matrix Reductions ---
    test('8. 2D Block: Global sum of 2x2 sub-matrix', () => {
        const view = arr2D.slice([0, 2], [1, 3]); // [[1, 2], [5, 6]]
        expect(view.sum()).toBe(14);
    });

    test('9. 2D Block: sum along axis 0 (columns)', () => {
        const view = arr2D.slice([0, 2], [1, 4]); // [[1, 2, 3], [5, 6, 7]]
        const res = view.sum(0); // [1+5, 2+6, 3+7]
        expect(Array.from(res.copy().data)).toEqual([6, 8, 10]);
    });

    test('10. 2D Block: mean along axis 1 (rows)', () => {
        const view = arr2D.slice([1, 3], [0, 3]); // [[4, 5, 6], [8, 9, 10]]
        const res = view.mean(1); // [5, 9]
        expect(Array.from(res.copy().data)).toEqual([5, 9]);
    });

    // --- Transposed Slices ---
    test('11. Transpose -> Slice: sum of row 0', () => {
        const view = arr2D.transpose().slice(0); // row 0 of T is col 0 of original: [0, 4, 8]
        expect(view.sum()).toBe(12);
    });

    test('12. Slice -> Transpose: mean along axis 0', () => {
        const view = arr2D.slice([0, 2], [0, 2]).transpose(); // [[0, 4], [1, 5]]
        const res = view.mean(0); 
        // Col 0 mean: (0+1)/2 = 0.5; Col 1 mean: (4+5)/2 = 4.5
        expect(Array.from(res.copy().data)).toEqual([0.5, 4.5]);
    });


    // --- 3D Slice Reductions ---
    test('13. 3D Plane: sum of arr3D[0, :, :]', () => {
        const view = arr3D.slice(0); // First 3x4 matrix
        expect(view.sum()).toBe((12 * 11) / 2);
    });

    test('14. 3D Pillar: mean of arr3D[:, :, 1]', () => {
        const view = arr3D.slice(null, null, 1); // shape [2, 3], values: [[1,5,9], [13,17,21]]
        expect(view.mean()).toBe(11);
    });

    test('15. 3D Sub-cube: max of [:, 1:, 2:]', () => {
        // axis 1: index 1,2; axis 2: index 2,3
        const view = arr3D.slice(null, [1, null], [2, null]); // shape [2, 2, 2]
        expect(view.max()).toBe(23);
    });

    test('16. 3D Slice: sum along axis 0 of sub-cube', () => {
        const view = arr3D.slice(null, 0, [0, 2]); // shape [2, 2], values: [[0,1], [12,13]]
        const res = view.sum(0); // [12, 14]
        expect(Array.from(res.copy().data)).toEqual([12, 14]);
    });

    // --- Negative Step & Reductions ---
    test('17. 2D Reverse: sum(0) on reversed rows', () => {
        const view = arr2D.slice([null, null, -1], null); // Rows: [8,9,10,11], [4,5,6,7], [0,1,2,3]
        const res = view.sum(0); // Same as original sum(0)
        expect(Array.from(res.copy().data)).toEqual([12, 15, 18, 21]);
    });

    test('18. 2D Reverse: mean(1) on reversed columns', () => {
        const view = arr2D.slice(null, [null, null, -1]); // Cols reversed
        const res = view.mean(1); // Same as original mean(1)
        expect(Array.from(res.copy().data)).toEqual([1.5, 5.5, 9.5]);
    });

    // --- Deep Chained Slices ---
    test('19. Chained: sum of colview of rowview', () => {
        // arr2D: 3x4
        const row = arr2D.slice([0, 2], null); // 2x4 block
        const col = row.slice(null, [1, 3]);   // 2x2 block: [[1,2], [5,6]]
        expect(col.sum()).toBe(14);
    });

    test('20. Chained: mean of reversed transposed slice', () => {
        const view = arr2D.transpose().slice([null, null, -1], 1); // Pick col 1 of T, reverse it: [7, 6, 5, 4]
        expect(view.mean()).toBe(5.5);
    });

    // --- Edge Cases ---
    test('21. Scalar Slice: sum of a single element', () => {
        const view = arr2D.slice(1, 2); // element 6
        expect(view.ndim).toBe(0);
        expect(view.sum()).toBe(6);
    });

    test('22. Empty Slice: sum of empty array', () => {
        const view = arr1D.slice([5, 2]); // Empty
        expect(view.sum()).toBe(0);
    });

    test('23. Large Step: min of slice [::100]', () => {
        const view = arr1D.slice([null, null, 100]); // [0]
        expect(view.min()).toBe(0);
    });

    // --- var/std on Slices ---
    test('24. Slice var: variance of [1, 3, 5]', () => {
        const view = arr1D.slice([1, 6, 2]); // [1, 3, 5], mean=3
        // var = ((1-3)^2 + 0 + (5-3)^2)/3 = 8/3
        expect(view.var()).toBeCloseTo(8/3);
    });

    test('25. Slice std: std of colview', () => {
        const view = arr2D.colview(0); // [0, 4, 8], mean=4
        // var = (16 + 0 + 16)/3 = 32/3
        expect(view.std()).toBeCloseTo(Math.sqrt(32/3));
    });

    // --- Multiple Axis Reductions (if supported) or Chained Reductions ---
    test('26. Chained Reductions: sum(0).sum(0)', () => {
        const view = arr2D.slice([0, 2], [0, 2]); // [[0, 1], [4, 5]]
        const res = view.sum(0).sum(0); 
        // Reduction on 1D result returns a 0D NDArray
        expect(res.get()).toBe(10); 
    });

    // --- Non-contiguous Offset Stress ---
    test('27. Offset Stress: sum of late sub-matrix', () => {
        const view = arr3D.slice(1, [1, 3], [2, 4]); // arr3D[1, 1:3, 2:4]
        // Plane 1: [[12..15],[16..19],[20..23]]
        // Slice: [[18, 19], [22, 23]]
        expect(view.sum()).toBe(18 + 19 + 22 + 23);
    });

    test('28. Negative Stride Stress: sum(1) on full reverse', () => {
        const view = arr2D.slice([null, null, -1], [null, null, -1]);
        // [[11,10,9,8], [7,6,5,4], [3,2,1,0]]
        const res = view.sum(1);
        expect(Array.from(res.copy().data)).toEqual([38, 22, 6]);
    });

    // --- Mix Reductions ---
    test('29. Mix: max(0) on a sliced view', () => {
        const view = arr2D.slice([0, 3], [1, 3]); // [[1,2], [5,6], [9,10]]
        const res = view.max(0); // [9, 10]
        expect(Array.from(res.copy().data)).toEqual([9, 10]);
    });

    test('30. std(0) on a transposed rowview', () => {
        // View is [[0,4], [1,5], [2,6], [3,7]]
        const view = arr2D.transpose().slice(null, [0, 2]); 
        const res = view.std(0);
        // Column 0: [0, 1, 2, 3] -> Mean 1.5 -> Var 1.25
        expect(res.get(0)).toBeCloseTo(Math.sqrt(1.25));
    });
});




describe('NDArray Reduction: Logical Path Coverage', () => {
    let arr1D, arr2D, arr3D;

    beforeEach(() => {
        // [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        arr1D = ndarray.arange(10);
        // 3x4: [[0..3], [4..7], [8..11]]
        arr2D = ndarray.arange(12).reshape([3, 4]);
        // 2x3x4
        arr3D = ndarray.arange(24).reshape([2, 3, 4]);
    });

    // --- Path 1: Global Reductions (axis === null) ---
    
    test('Global Path: Contiguous 1D slice', () => {
        const view = arr1D.slice([2, 5]); // [2, 3, 4]
        expect(view.isContiguous).toBe(true);
        // Hits: if (this.isContiguous) { for loop }
        expect(view.sum()).toBe(9); 
    });

    test('Global Path: Non-contiguous 1D slice (step)', () => {
        const view = arr1D.slice([0, 6, 2]); // [0, 2, 4]
        expect(view.isContiguous).toBe(false);
        // Hits: else { this.iterate(...) }
        expect(view.sum()).toBe(6);
    });

    test('Global Path: Transposed 2D view', () => {
        const view = arr2D.transpose();
        expect(view.isContiguous).toBe(false);
        expect(view.mean()).toBe(5.5);
    });

    // --- Path 2: Contiguous Last-Axis (Fast Path) ---

    test('Fast Path: Summing last axis of contiguous 2D', () => {
        // Hits: if (this.isContiguous && axis === this.ndim - 1)
        const res = arr2D.sum(1); // Rows: [6, 22, 38]
        expect(Array.from(res.data)).toEqual([6, 22, 38]);
    });

    test('Fast Path: Sliced contiguous block', () => {
        const view = arr2D.slice([0, 2], null); // First two rows, still contiguous
        expect(view.isContiguous).toBe(true);
        const res = view.max(1); // Last axis (1)
        expect(Array.from(res.data)).toEqual([3, 7]);
    });

    // --- Path 3: Generic Block-Jumping (Intermediate Axis or Non-contiguous) ---

    test('Generic Path: Reduction on axis 0 (intermediate/first)', () => {
        // Hits: Scenario 3: Generic block-jumping path
        const res = arr2D.sum(0); // [12, 15, 18, 21]
        expect(Array.from(res.data)).toEqual([12, 15, 18, 21]);
    });

    test('Generic Path: 3D array middle axis', () => {
        // arr3D shape [2, 3, 4], reduce axis 1
        const res = arr3D.sum(1); 
        expect(res.shape).toEqual(new Int32Array([2, 4]));
        // Check first element: 0+4+8 = 12
        expect(res.get(0, 0)).toBe(12);
    });

    test('Generic Path: Non-contiguous view (transposed sum)', () => {
        const view = arr2D.transpose(); // Shape [4, 3]
        // Even if we reduce last axis (1), it's not contiguous anymore
        expect(view.isContiguous).toBe(false); 
        const res = view.sum(1); // Summing original columns: [12, 15, 18, 21]
        expect(Array.from(res.data)).toEqual([12, 15, 18, 21]);
    });

    test('Generic Path: Strided slice (step > 1) axis reduction', () => {
        const view = arr2D.slice(null, [0, 4, 2]); // Cols 0 and 2: [[0,2],[4,6],[8,10]]
        const res = view.sum(0); // [0+4+8, 2+6+10] = [12, 18]
        expect(Array.from(res.data)).toEqual([12, 18]);
    });

    // --- Path 4: 0D / Scalar Handling (Bug Fix Verification) ---

    test('Scalar Path: sum() of a 0D slice', () => {
        const view = arr2D.slice(1, 1); // Value 5, ndim 0
        expect(view.ndim).toBe(0);
        // Hits: if (this.ndim === 0) return finalFn(...)
        expect(view.sum()).toBe(5);
    });

    test('Scalar Path: mean() of a 0D slice', () => {
        const view = arr1D.slice(5); // Value 5
        expect(view.mean()).toBe(5);
    });

    test('Scalar Path: std() of a 0D slice', () => {
        const view = arr1D.slice(9); // Value 9
        // var of a single point is 0
        expect(view.std()).toBe(0);
    });

    // --- Complex Composition Stress Tests ---

    test('Composition: Reduce reversed non-contiguous view', () => {
        // [0, 1, 2, 3, 4, 5] -> reversed [5, 4, 3, 2, 1, 0] -> slice [1, 4] -> [4, 3, 2]
        const view = arr1D.slice([0, 6]).slice([null, null, -1]).slice([1, 4]);
        expect(view.sum()).toBe(9);
    });

    test('Composition: 3D transpose -> slice -> mean(axis)', () => {
        // arr3D [2, 3, 4] -> transpose [2, 0, 1] -> [4, 2, 3]
        // slice(0) -> [2, 3] matrix
        const view = arr3D.transpose([2, 0, 1]).slice(0); 
        // Original col 0 was: [[0,4,8], [12,16,20]]
        const res = view.mean(1); // mean of rows
        expect(res.get(0)).toBe(4);  // (0+4+8)/3
        expect(res.get(1)).toBe(16); // (12+16+20)/3
    });

    // --- Variance & StdDev Specifics ---

    test('Statistics: Global variance of a slice', () => {
        const view = arr1D.slice([0, 3]); // [0, 1, 2] -> mean 1
        // var: ((0-1)^2 + 0 + (2-1)^2)/3 = 2/3
        expect(view.var()).toBeCloseTo(0.6666666666666666);
    });

    test('Statistics: Axis variance of a slice', () => {
        // view: [[0, 1], [4, 5]]
        const view = arr2D.slice([0, 2], [0, 2]);
        const res = view.var(1); // row-wise var
        // row 0: [0, 1] -> mean 0.5 -> var ((0-0.5)^2 + (1-0.5)^2)/2 = 0.25
        expect(Array.from(res.data)).toEqual([0.25, 0.25]);
    });

    test('Statistics: Axis std of a transposed slice', () => {
        // view: [[0, 4], [1, 5]]
        const view = arr2D.slice([0, 2], [0, 2]).transpose();
        const res = view.std(0); // col-wise std
        expect(res.get(0)).toBeCloseTo(0.5); // std of [0, 1]
    });

    // --- Boundary & Identity Tests ---

    test('FinalFn: Ensure mean() uses the IDENTITY logic correctly', () => {
        const view = arr1D.slice([0, 4]); // [0, 1, 2, 3]
        // Check if finalFn is actually called (sum/4)
        expect(view.mean()).toBe(1.5);
    });

    test('Identity: sum() vs Global reduce', () => {
        const view = arr2D.slice(null, [null, null, -1]); // Reverse columns
        expect(view.sum()).toBe(arr2D.sum());
    });
});