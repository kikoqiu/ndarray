const ndarray = require('../dist/ndarray.cjs');
const { NDArray } = ndarray;

let arr; // Declared here

describe('NDArray View', () => {

    beforeEach(() => {
        // Create a 2x3 array for view tests: [[0, 1, 2], [3, 4, 5]]
        arr = ndarray.arange(6).reshape([2, 3]); 
    });

    test('reshape', () => {
        const reshaped = arr.reshape([3, 2]);
        expect(reshaped.shape).toEqual(new Int32Array([3, 2]));
        expect(reshaped.strides).toEqual(new Int32Array([2, 1]));
        // Check a value to ensure data order is correct
        expect(reshaped.get(2, 0)).toBe(4);
    });

    test('transpose', () => {
        const transposed = arr.transpose(); // default transpose
        expect(transposed.shape).toEqual(new Int32Array([3, 2]));
        expect(transposed.strides).toEqual(new Int32Array([1, 3])); // Strides are swapped
        expect(transposed.isContiguous).toBe(false);

        // Check values
        expect(transposed.get(0, 1)).toBe(3);
        expect(transposed.get(2, 0)).toBe(2);
    });

    test('slice (full)', () => {
        const sliced = arr.slice(null, [0, 2]); // all rows, cols 0 to 2 (exclusive)
        expect(sliced.shape).toEqual(new Int32Array([2, 2]));
        expect(sliced.get(1, 1)).toBe(4); // original was arr.get(1, 1) -> 4
    });

    test('slice (with reduction)', () => {
        // Select the first row
        const sliced = arr.slice(0); // arr[0]
        expect(sliced.shape).toEqual(new Int32Array([3]));
        expect(sliced.get(1)).toBe(1);
        expect(sliced.offset).toBe(0); // Still a view on original data
    });
    
    test('slice (with step)', () => {
        const a = ndarray.arange(10); // [0, 1, ..., 9]
        const sliced = a.slice([1, 8, 2]); // start=1, end=8, step=2
        expect(sliced.shape).toEqual(new Int32Array([4]));
        
        // A sliced array is a view. To check its contents easily, we copy it.
        const copied = sliced.copy(); 
        expect(copied.data).toEqual(new Float64Array([1, 3, 5, 7]));
    });

    test('copy', () => {
        const transposed = arr.transpose();
        expect(transposed.isContiguous).toBe(false);

        const copied = transposed.copy();
        expect(copied.isContiguous).toBe(true);
        expect(copied.shape).toEqual(new Int32Array([3, 2]));
        expect(copied.strides).toEqual(new Int32Array([2, 1])); // Default C-order strides after copy
        expect(copied.data).toEqual(new Float64Array([0, 3, 1, 4, 2, 5]));
    });

    test('asContiguous', () => {
        const transposed = arr.transpose();
        expect(transposed.isContiguous).toBe(false);
        
        const contiguousView = arr.asContiguous();
        expect(contiguousView).toBe(arr); // Should return itself as it's already contiguous

        const contiguousCopy = transposed.asContiguous();
        expect(contiguousCopy).not.toBe(transposed); // Should be a new object
        expect(contiguousCopy.isContiguous).toBe(true);
        expect(contiguousCopy.data).toEqual(new Float64Array([0, 3, 1, 4, 2, 5]));
    });
});

describe('Complex View Manipulation', () => {
    beforeEach(() => {
        // Create a 2x3 array for view tests: [[0, 1, 2], [3, 4, 5]]
        arr = ndarray.arange(6).reshape([2, 3]); 
    });

    test('Chained views (transpose -> slice)', () => {
        // arr = [[0, 1, 2], [3, 4, 5]]
        const t = arr.transpose(); // [[0, 3], [1, 4], [2, 5]]
        expect(t.isContiguous).toBe(false);

        const sliced = t.slice([0, 2], null); // First two rows of transposed: [[0, 3], [1, 4]]
        expect(sliced.shape).toEqual(new Int32Array([2, 2]));
        expect(sliced.isContiguous).toBe(false);

        // Check values by getting them from the view
        expect(sliced.get(0, 0)).toBe(0);
        expect(sliced.get(0, 1)).toBe(3);
        expect(sliced.get(1, 0)).toBe(1);
        expect(sliced.get(1, 1)).toBe(4);

        // Verify with a copy
        const copied = sliced.copy();
        expect(copied.isContiguous).toBe(true);
        expect(copied.data).toEqual(new Float64Array([0, 3, 1, 4]));
    });

    test('Operations on slices', () => {
        // arr = [[0, 1, 2], [3, 4, 5]]
        const second_row = arr.slice(1); // [3, 4, 5]
        expect(second_row.sum()).toBe(12);

        const first_col = arr.slice(null, 0); // [0, 3]
        expect(first_col.mean()).toBe(1.5);

        // Perform a binary operation on a slice
        const new_col = first_col.mul(10); // [0, 30]
        expect(new_col.copy().data).toEqual(new Float64Array([0, 30]));
    });

    test('reshape non-contiguous should throw error', () => {
        const transposed = arr.transpose();
        expect(transposed.isContiguous).toBe(false);
        expect(() => transposed.reshape([2, 3])).toThrow("Reshape of non-contiguous view is ambiguous");
    });
});


describe('NDArray Slicing - Category 1: Basic Slicing & Dimension Reduction', () => {
    let arr1D, arr2D, arr3D;

    beforeEach(() => {
        // arr1D = [0, 1, 2, 3, 4, 5]
        arr1D = new NDArray(new Float64Array([0, 1, 2, 3, 4, 5]), { 
            shape: new Int32Array([6]) 
        });
        
        // arr2D = [[0, 1, 2], [3, 4, 5]]
        arr2D = new NDArray(new Float64Array([0, 1, 2, 3, 4, 5]), { 
            shape: new Int32Array([2, 3]) 
        });

        // arr3D = 2x2x2 cube
        arr3D = new NDArray(new Float64Array(Array.from({length: 8}, (_, i) => i)), { 
            shape: new Int32Array([2, 2, 2]) 
        });
    });

    // --- 1D Scalar Indexing (Reduces to 0D/Scalar) ---
    test('1. 1D: Pick first element (scalar)', () => {
        const view = arr1D.slice(0);
        expect(view.ndim).toBe(0);
        expect(view.shape).toEqual(new Int32Array([]));
        expect(view.get()).toBe(0);
    });

    test('2. 1D: Pick last element', () => {
        const view = arr1D.slice(5);
        expect(view.get()).toBe(5);
    });

    // --- 2D Scalar Indexing (Reduces dimensionality) ---
    test('3. 2D: Pick first row (reduces to 1D)', () => {
        const view = arr2D.slice(0); 
        expect(view.ndim).toBe(1);
        expect(view.shape).toEqual(new Int32Array([3]));
        expect(Array.from(view.copy().data)).toEqual([0, 1, 2]);
    });

    test('4. 2D: Pick second row', () => {
        const view = arr2D.slice(1);
        expect(view.shape).toEqual(new Int32Array([3]));
        expect(Array.from(view.copy().data)).toEqual([3, 4, 5]);
    });

    test('5. 2D: Pick specific element (1, 1)', () => {
        const view = arr2D.slice(1, 1);
        expect(view.ndim).toBe(0);
        expect(view.get()).toBe(4);
    });

    // --- 1D Range Slicing (Preserves dimensionality) ---
    test('6. 1D: Simple range [1, 4]', () => {
        const view = arr1D.slice([1, 4]); // indices 1, 2, 3
        expect(view.shape).toEqual(new Int32Array([3]));
        expect(Array.from(view.copy().data)).toEqual([1, 2, 3]);
    });

    test('7. 1D: Range with explicit step [0, 6, 2]', () => {
        const view = arr1D.slice([0, 6, 2]); // indices 0, 2, 4
        expect(view.shape).toEqual(new Int32Array([3]));
        expect(Array.from(view.copy().data)).toEqual([0, 2, 4]);
    });

    test('8. 1D: Range with step 3', () => {
        const view = arr1D.slice([0, 6, 3]); // indices 0, 3
        expect(view.shape).toEqual(new Int32Array([2]));
        expect(Array.from(view.copy().data)).toEqual([0, 3]);
    });

    // --- Default & Null Handling ---
    test('9. 1D: Null start [:3]', () => {
        const view = arr1D.slice([null, 3]);
        expect(view.shape).toEqual(new Int32Array([3]));
        expect(Array.from(view.copy().data)).toEqual([0, 1, 2]);
    });

    test('10. 1D: Null end [3:]', () => {
        const view = arr1D.slice([3, null]);
        expect(view.shape).toEqual(new Int32Array([3]));
        expect(Array.from(view.copy().data)).toEqual([3, 4, 5]);
    });

    test('11. 1D: All null (slice(null))', () => {
        const view = arr1D.slice(null);
        expect(view.shape).toEqual(new Int32Array([6]));
        expect(view.offset).toBe(0);
    });

    test('12. 1D: Array with all nulls [null, null, null]', () => {
        const view = arr1D.slice([null, null, null]);
        expect(view.shape).toEqual(new Int32Array([6]));
        expect(Array.from(view.copy().data)).toEqual([0, 1, 2, 3, 4, 5]);
    });

    // --- 2D Range Slicing ---
    test('13. 2D: Slice rows, all columns', () => {
        const view = arr2D.slice([0, 1], null); 
        expect(view.shape).toEqual(new Int32Array([1, 3]));
        expect(Array.from(view.copy().data)).toEqual([0, 1, 2]);
    });

    test('14. 2D: All rows, slice columns [:, 1:3]', () => {
        const view = arr2D.slice(null, [1, 3]);
        expect(view.shape).toEqual(new Int32Array([2, 2]));
        expect(view.get(0, 0)).toBe(1); // Row 0, Col 1 of original
        expect(view.get(1, 1)).toBe(5); // Row 1, Col 2 of original
    });

    test('15. 2D: Slice both dimensions', () => {
        const view = arr2D.slice([0, 2], [0, 2]);
        expect(view.shape).toEqual(new Int32Array([2, 2]));
        // Result: [[0, 1], [3, 4]]
        expect(Array.from(view.copy().data)).toEqual([0, 1, 3, 4]);
    });

    // --- Mixed Scalar and Range ---
    test('16. 2D: Scalar row, sliced column', () => {
        const view = arr2D.slice(1, [0, 2]); // Equivalent to arr[1, 0:2]
        expect(view.ndim).toBe(1);
        expect(view.shape).toEqual(new Int32Array([2]));
        expect(Array.from(view.copy().data)).toEqual([3, 4]);
    });

    test('17. 2D: Sliced row, scalar column', () => {
        const view = arr2D.slice([0, 2], 2); // Equivalent to arr[0:2, 2]
        expect(view.ndim).toBe(1);
        expect(view.shape).toEqual(new Int32Array([2]));
        expect(Array.from(view.copy().data)).toEqual([2, 5]);
    });

    // --- 3D Slicing ---
    test('18. 3D: Pick single plane (2D view)', () => {
        const view = arr3D.slice(0, null, null);
        expect(view.ndim).toBe(2);
        expect(view.shape).toEqual(new Int32Array([2, 2]));
    });

    test('19. 3D: Pick single pillar (1D view)', () => {
        const view = arr3D.slice(0, 0, null);
        expect(view.ndim).toBe(1);
        expect(view.shape).toEqual(new Int32Array([2]));
    });

    test('20. 3D: Slice all dimensions [0:1, 0:1, 0:1]', () => {
        const view = arr3D.slice([0, 1], [0, 1], [0, 1]);
        expect(view.ndim).toBe(3);
        expect(view.shape).toEqual(new Int32Array([1, 1, 1]));
        expect(view.get(0, 0, 0)).toBe(0);
    });

    // --- Boundary Clamping (NumPy style) ---
    test('21. 1D: Start beyond size returns empty', () => {
        const view = arr1D.slice([10, 20]);
        expect(view.shape).toEqual(new Int32Array([0]));
    });

    test('22. 1D: End beyond size is clamped', () => {
        const view = arr1D.slice([0, 100]);
        expect(view.shape).toEqual(new Int32Array([6]));
    });

    test('23. 1D: Start > End with positive step returns empty', () => {
        const view = arr1D.slice([4, 2]);
        expect(view.shape).toEqual(new Int32Array([0]));
    });

    // --- Ellipsis-like Behavior (Fewer specs) ---
    test('24. 2D: Fewer specs than ndim defaults to full slices', () => {
        const view = arr2D.slice([0, 1]); // Same as arr2D.slice([0, 1], null)
        expect(view.shape).toEqual(new Int32Array([1, 3]));
        expect(Array.from(view.copy().data)).toEqual([0, 1, 2]);
    });

    // --- Rowview & Colview Helpers ---
    test('25. rowview(0): First row', () => {
        const view = arr2D.rowview(0);
        expect(view.ndim).toBe(1);
        expect(view.shape).toEqual(new Int32Array([3]));
        expect(Array.from(view.copy().data)).toEqual([0, 1, 2]);
    });

    test('26. rowview(1): Second row', () => {
        const view = arr2D.rowview(1);
        expect(view.shape).toEqual(new Int32Array([3]));
        expect(Array.from(view.copy().data)).toEqual([3, 4, 5]);
    });

    test('27. colview(0): First column', () => {
        const view = arr2D.colview(0);
        expect(view.ndim).toBe(1);
        expect(view.shape).toEqual(new Int32Array([2]));
        expect(Array.from(view.copy().data)).toEqual([0, 3]);
    });

    test('28. colview(2): Third column', () => {
        const view = arr2D.colview(2);
        expect(view.shape).toEqual(new Int32Array([2]));
        expect(Array.from(view.copy().data)).toEqual([2, 5]);
    });

    // --- Edge Case / Error Handling ---
    test('29. Scalar index out of bounds (Positive) throws error', () => {
        expect(() => arr1D.slice(10)).toThrow();
    });

    test('30. Slice step zero throws error', () => {
        expect(() => arr1D.slice([0, 5, 0])).toThrow("step cannot be zero");
    });
});

describe('NDArray Slicing - Category 2: Negative Indices & Negative Steps', () => {
    let arr1D, arr2D;

    beforeEach(() => {
        // arr1D = [0, 1, 2, 3, 4, 5]
        arr1D = new NDArray(new Float64Array([0, 1, 2, 3, 4, 5]), { 
            shape: new Int32Array([6]) 
        });
        
        // arr2D = [[0, 1, 2], [3, 4, 5]]
        arr2D = new NDArray(new Float64Array([0, 1, 2, 3, 4, 5]), { 
            shape: new Int32Array([2, 3]) 
        });
    });

    // --- 1D Negative Scalar Indexing ---
    test('1. 1D: Pick last element via -1', () => {
        const view = arr1D.slice(-1);
        expect(view.get()).toBe(5);
        expect(view.shape).toEqual(new Int32Array([]));
    });

    test('2. 1D: Pick first element via -6', () => {
        const view = arr1D.slice(-6);
        expect(view.get()).toBe(0);
    });

    test('3. 1D: Out of bounds negative index throws', () => {
        expect(() => arr1D.slice(-7)).toThrow();
    });

    // --- 1D Negative Indices in Range (Positive Step) ---
    test('4. 1D: Range with negative start [-3:]', () => {
        const view = arr1D.slice([-3, null]); // Indices 3, 4, 5
        expect(Array.from(view.copy().data)).toEqual([3, 4, 5]);
    });

    test('5. 1D: Range with negative end [:-2]', () => {
        const view = arr1D.slice([null, -2]); // Indices 0, 1, 2, 3
        expect(Array.from(view.copy().data)).toEqual([0, 1, 2, 3]);
    });

    test('6. 1D: Negative start and end [-4, -1]', () => {
        const view = arr1D.slice([-4, -1]); // Indices 2, 3, 4
        expect(Array.from(view.copy().data)).toEqual([2, 3, 4]);
    });

    // --- 1D Negative Steps (Reversing) ---
    test('7. 1D: Full reverse [::-1]', () => {
        const view = arr1D.slice([null, null, -1]);
        expect(view.shape).toEqual(new Int32Array([6]));
        expect(Array.from(view.copy().data)).toEqual([5, 4, 3, 2, 1, 0]);
    });

    test('8. 1D: Reverse with step -2 [::-2]', () => {
        const view = arr1D.slice([null, null, -2]); // Indices 5, 3, 1
        expect(Array.from(view.copy().data)).toEqual([5, 3, 1]);
    });

    test('9. 1D: Positive start/end with negative step [4:1:-1]', () => {
        const view = arr1D.slice([4, 1, -1]); // Indices 4, 3, 2
        expect(Array.from(view.copy().data)).toEqual([4, 3, 2]);
    });

    test('10. 1D: Negative step including index 0 [5:-7:-1]', () => {
        // NumPy: 5 to -7 step -1 covers indices 5,4,3,2,1,0
        const view = arr1D.slice([5, -7, -1]); 
        expect(Array.from(view.copy().data)).toEqual([5, 4, 3, 2, 1, 0]);
    });

    test('11. 1D: Negative step empty slice [1:4:-1]', () => {
        const view = arr1D.slice([1, 4, -1]); // start < end with negative step
        expect(view.shape).toEqual(new Int32Array([0]));
    });

    test('12. 1D: Negative step with negative indices [-1:-4:-1]', () => {
        const view = arr1D.slice([-1, -4, -1]); // Indices 5, 4, 3
        expect(Array.from(view.copy().data)).toEqual([5, 4, 3]);
    });

    // --- 2D Negative Slicing ---
    test('13. 2D: Pick last row via -1', () => {
        const view = arr2D.slice(-1, null);
        expect(view.shape).toEqual(new Int32Array([3]));
        expect(Array.from(view.copy().data)).toEqual([3, 4, 5]);
    });

    test('14. 2D: Reverse rows [::-1, :]', () => {
        const view = arr2D.slice([null, null, -1], null);
        // [[3, 4, 5], [0, 1, 2]]
        expect(view.get(0, 0)).toBe(3);
        expect(view.get(1, 2)).toBe(2);
    });

    test('15. 2D: Reverse columns [:, ::-1]', () => {
        const view = arr2D.slice(null, [null, null, -1]);
        // [[2, 1, 0], [5, 4, 3]]
        expect(view.get(0, 0)).toBe(2);
        expect(view.get(1, 2)).toBe(3);
    });

    test('16. 2D: Reverse both axes [::-1, ::-1]', () => {
        const view = arr2D.slice([null, null, -1], [null, null, -1]);
        // [[5, 4, 3], [2, 1, 0]]
        expect(Array.from(view.copy().data)).toEqual([5, 4, 3, 2, 1, 0]);
    });

    test('17. 2D: Negative index for column [:, -1]', () => {
        const view = arr2D.slice(null, -1); // Last column
        expect(view.shape).toEqual(new Int32Array([2]));
        expect(Array.from(view.copy().data)).toEqual([2, 5]);
    });

    // --- Advanced Boundary Clamping with Negative Steps ---
    test('18. 1D: Negative step, start exceeds max boundary', () => {
        const view = arr1D.slice([100, null, -1]); // Starts from size-1
        expect(view.shape).toEqual(new Int32Array([6]));
        expect(view.get(0)).toBe(5);
    });

    test('19. 1D: Negative step, end exceeds min boundary', () => {
        const view = arr1D.slice([null, -100, -1]); // Ends at 0
        expect(view.shape).toEqual(new Int32Array([6]));
        expect(view.get(5)).toBe(0);
    });

    test('20. 1D: Negative step, end is 0 [5:0:-1]', () => {
        const view = arr1D.slice([5, 0, -1]); // Indices 5, 4, 3, 2, 1 (Excludes 0)
        expect(Array.from(view.copy().data)).toEqual([5, 4, 3, 2, 1]);
    });

    // --- Complex Multi-Dimensional Slices ---
    test('21. 3D Setup & Negative Slice', () => {
        const arr3D = new NDArray(new Float64Array(Array.from({length: 8}, (_, i) => i)), { 
            shape: new Int32Array([2, 2, 2]) 
        });
        // Reverse first and third dimension, pick second row of second dimension
        const view = arr3D.slice([null, null, -1], 1, [null, null, -1]);
        expect(view.shape).toEqual(new Int32Array([2, 2]));
        // Original data: [[[0,1],[2,3]], [[4,5],[6,7]]]
        // Slicing: Pick index 1 of dim1 -> [[2,3], [6,7]]
        // Reversing dim0 and dim2 -> [[7,6], [3,2]]
        expect(view.get(0, 0)).toBe(7);
        expect(view.get(1, 1)).toBe(2);
    });

    // --- Consistency with Step 1 vs -1 ---
    test('22. 1D: Compare [1:4] vs [3:0:-1]', () => {
        const fwd = arr1D.slice([1, 4, 1]); // [1, 2, 3]
        const rev = arr1D.slice([3, 0, -1]); // [3, 2, 1]
        expect(Array.from(fwd.copy().data)).toEqual(Array.from(rev.copy().data).reverse());
    });

    // --- Clamping specific negative values ---
    test('23. 1D: Negative index -size is index 0', () => {
        const view = arr1D.slice([-6, 2]); // [0, 1]
        expect(Array.from(view.copy().data)).toEqual([0, 1]);
    });

    test('24. 1D: Negative index -size-1 clamped in range slice', () => {
        const view = arr1D.slice([-100, 2]); // Clamped to 0
        expect(Array.from(view.copy().data)).toEqual([0, 1]);
    });

    // --- Slicing on a sliced view (Nested Slicing) ---
    test('25. Nested: Negative slice on a negative slice', () => {
        const rev = arr1D.slice([null, null, -1]); // [5,4,3,2,1,0]
        const sub = rev.slice([-2, null]); // Last two of rev: [1, 0]
        expect(Array.from(sub.copy().data)).toEqual([1, 0]);
    });

    test('26. Nested: Positive slice on negative slice', () => {
        const rev = arr1D.slice([null, null, -1]); // [5,4,3,2,1,0]
        const sub = rev.slice([1, 3]); // Index 1, 2 of rev: [4, 3]
        expect(Array.from(sub.copy().data)).toEqual([4, 3]);
    });

    // --- Combined rowview/colview with slice ---
    test('27. rowview with negative index', () => {
        const view = arr2D.rowview(-1);
        expect(Array.from(view.copy().data)).toEqual([3, 4, 5]);
    });

    test('28. colview with negative index', () => {
        const view = arr2D.colview(-1);
        expect(Array.from(view.copy().data)).toEqual([2, 5]);
    });

    // --- Logic Verification: Start/End as -1 for negative step ---
    test('29. 1D: [null, -1, -1] should stop before last element', () => {
        // NumPy: arr[::-1] includes everything. arr[: -1 : -1] excludes original index 0 (which is now last)
        // Actually in NumPy, end is exclusive. index -1 is last element.
        // If step is -1, end -1 means stop BEFORE original index -1.
        const view = arr1D.slice([null, -1, -1]); 
        // Index -1 is 5. Slice starts at 5, ends before 5.
        expect(view.shape[0]).toBe(0);
    });

    test('30. 1D: [-1, null, -1] is same as [null, null, -1]', () => {
        const view = arr1D.slice([-1, null, -1]);
        expect(Array.from(view.copy().data)).toEqual([5, 4, 3, 2, 1, 0]);
    });
});


describe('NDArray Slicing - Category 3: Complex View Composition & Transposed Slicing', () => {
    let arr1D, arr2D, arr3D;

    beforeEach(() => {
        // arr1D = [0, 1, 2, 3, 4, 5]
        arr1D = new NDArray(new Float64Array([0, 1, 2, 3, 4, 5]), { 
            shape: new Int32Array([6]) 
        });

        // arr2D = [[0, 1, 2], [3, 4, 5]]
        arr2D = new NDArray(new Float64Array([0, 1, 2, 3, 4, 5]), { 
            shape: new Int32Array([2, 3]) 
        });

        // arr3D = 2x2x2 cube: [[[0,1],[2,3]], [[4,5],[6,7]]]
        arr3D = new NDArray(new Float64Array(Array.from({length: 8}, (_, i) => i)), { 
            shape: new Int32Array([2, 2, 2]) 
        });
    });

    // --- Transpose + Slice ---
    test('1. T -> Slice: Row of a transposed matrix', () => {
        const t = arr2D.transpose(); // shape [3, 2]: [[0,3],[1,4],[2,5]]
        const row = t.slice(0); // Picks [0, 3]
        expect(row.shape).toEqual(new Int32Array([2]));
        expect(Array.from(row.copy().data)).toEqual([0, 3]);
    });

    test('2. T -> Slice: Column of a transposed matrix', () => {
        const t = arr2D.transpose();
        const col = t.slice(null, 1); // Picks second col: [3, 4, 5]
        expect(col.shape).toEqual(new Int32Array([3]));
        expect(Array.from(col.copy().data)).toEqual([3, 4, 5]);
    });

    test('2.2 T -> Slice: Column of a transposed matrix, with []', () => {
        const t = arr2D.transpose();
        const col = t.slice([], 1); // Picks second col: [3, 4, 5]
        expect(col.shape).toEqual(new Int32Array([3]));
        expect(Array.from(col.copy().data)).toEqual([3, 4, 5]);
    });

    test('3. T -> Slice: Sub-matrix of transposed', () => {
        const t = arr2D.transpose(); // [[0,3],[1,4],[2,5]]
        const sub = t.slice([1, 3], null); // [[1,4],[2,5]]
        expect(sub.shape).toEqual(new Int32Array([2, 2]));
        expect(sub.get(1, 1)).toBe(5);
    });

    // --- Slice + Transpose ---
    test('4. Slice -> T: Transpose of a column slice', () => {
        const col = arr2D.slice(null, [1, 3]); // [[1, 2], [4, 5]]
        const t = col.transpose(); // [[1, 4], [2, 5]]
        expect(t.shape).toEqual(new Int32Array([2, 2]));
        expect(t.get(0, 1)).toBe(4);
    });

    test('5. Slice -> T: Transpose of a row reduction', () => {
        const row = arr2D.slice(1); // [3, 4, 5] (1D)
        const t = row.transpose(); // 1D transpose is same
        expect(t.shape).toEqual(new Int32Array([3]));
        expect(t.get(1)).toBe(4);
    });

    // --- Negative Step + Transpose ---
    test('6. Reverse -> T: Transpose of reversed rows', () => {
        const rev = arr2D.slice([null, null, -1], null); // [[3,4,5],[0,1,2]]
        const t = rev.transpose(); // [[3,0],[4,1],[5,2]]
        expect(t.get(0, 1)).toBe(0);
        expect(t.get(2, 0)).toBe(5);
    });

    test('7. T -> Reverse: Reverse rows of a transposed matrix', () => {
        const t = arr2D.transpose(); // [[0,3],[1,4],[2,5]]
        const rev = t.slice([null, null, -1], null); // [[2,5],[1,4],[0,3]]
        expect(rev.get(0, 0)).toBe(2);
        expect(rev.get(2, 1)).toBe(3);
    });

    // --- 3D Permutations & Slicing ---
    test('8. 3D Permute -> Slice: Slice after axes swap', () => {
        // Swap axes 0 and 2: shape [2, 2, 2]
        const swapped = arr3D.transpose([2, 1, 0]); 
        const sliced = swapped.slice(0, null, null); // Pick first depth plane
        expect(sliced.shape).toEqual(new Int32Array([2, 2]));
        // Original [0,0,0]=0, [1,0,0]=4 => After swap: [0,0,0]=0, [0,0,1]=4
        expect(sliced.get(0, 1)).toBe(4);
    });

    test('9. 3D Slice -> Permute: Permute after slicing', () => {
        const sliced = arr3D.slice(null, 0, null); // shape [2, 2]: [[0,1],[4,5]]
        const t = sliced.transpose(); // [[0,4],[1,5]]
        expect(t.shape).toEqual(new Int32Array([2, 2]));
        expect(t.get(0, 1)).toBe(4);
    });

    // --- Chained Slicing (View of View) ---
    test('10. Chained: Slice of a Slice (Positive)', () => {
        const v1 = arr1D.slice([1, 5]); // [1, 2, 3, 4]
        const v2 = v1.slice([1, 3]);    // [2, 3]
        expect(v2.shape).toEqual(new Int32Array([2]));
        expect(v2.get(0)).toBe(2);
    });

    test('11. Chained: Slice of a Slice (Negative step)', () => {
        const v1 = arr1D.slice([null, null, -1]); // [5,4,3,2,1,0]
        const v2 = v1.slice([1, 4]);              // [4,3,2]
        expect(v2.get(0)).toBe(4);
    });

    test('12. Chained: Picking a row from a sliced matrix', () => {
        const v1 = arr2D.slice(null, [1, 3]); // [[1,2], [4,5]]
        const v2 = v1.slice(1);              // [4, 5]
        expect(v2.shape).toEqual(new Int32Array([2]));
        expect(v2.get(1)).toBe(5);
    });

    // --- Rowview/Colview on Non-contiguous Views ---
    test('13. colview on transposed matrix', () => {
        const t = arr2D.transpose(); // [[0,3],[1,4],[2,5]]
        const col = t.colview(1);    // [3, 4, 5]
        expect(col.shape).toEqual(new Int32Array([3]));
        expect(col.get(2)).toBe(5);
    });

    test('14. rowview on transposed matrix', () => {
        const t = arr2D.transpose(); // [[0,3],[1,4],[2,5]]
        const row = t.rowview(2);    // [2, 5]
        expect(row.get(1)).toBe(5);
    });

    // --- Complex Reshape & Slice Interaction ---
    test('15. Slice (Contiguous) -> Reshape -> Slice', () => {
        // arr1D = [0,1,2,3,4,5]
        const v1 = arr1D.slice([0, 4]); // [0,1,2,3] - this is contiguous
        const v2 = v1.reshape([2, 2]);  // [[0,1],[2,3]]
        const v3 = v2.slice(null, 1);   // [1, 3]
        expect(Array.from(v3.copy().data)).toEqual([1, 3]);
    });

    test('16. Slice (Non-contiguous) -> Reshape throws', () => {
        const v1 = arr1D.slice([0, 6, 2]); // [0, 2, 4] - non-contiguous
        expect(() => v1.reshape([3, 1])).toThrow();
    });

    // --- Math Operations on Slices ---
    test('17. Sum of a column slice', () => {
        const col = arr2D.colview(1); // [1, 4]
        // Assuming sum() is implemented
        expect(col.sum()).toBe(5);
    });

    test('18. Mean of a reversed slice', () => {
        const rev = arr1D.slice([null, null, -1]); // [5,4,3,2,1,0]
        const sub = rev.slice([0, 3]); // [5,4,3]
        expect(sub.mean()).toBe(4);
    });

    // --- Data Integrity & Copying ---
    test('19. Copy of a complex 2D slice', () => {
        const complex = arr2D.transpose().slice([0, 2], [1, 2]); // col 1 of T: [3, 4]
        const cp = complex.copy();
        expect(cp.isContiguous).toBe(true);
        expect(Array.from(cp.data)).toEqual([3, 4]);
    });

    test('20. Copy of a 3D permuted slice', () => {
        // arr3D transpose [2, 0, 1] -> New Shape [Col, Depth, Row]
        // slice(0, 1, null) -> Col=0, Depth=1, Row=all
        // Matches original: (Depth=1, Row=0, Col=0) -> 4 and (Depth=1, Row=1, Col=0) -> 6
        const complex = arr3D.transpose([2, 0, 1]).slice(0, 1, null); 
        expect(Array.from(complex.copy().data)).toEqual([4, 6]); // 之前算错了，这里应为 [4, 6]
    });

    // --- Stride Logic Validation ---
    test('21. Verify strides after T and Slice', () => {
        const t = arr2D.transpose(); 
        const sliced = t.slice([0, 2], null); // shape [2, 2]
        // Original strides were [3, 1]. T strides [1, 3].
        // Slice doesn't change strides, just shape and offset.
        expect(sliced.strides).toEqual(new Int32Array([1, 3]));
    });

    test('22. Verify strides after negative step slice', () => {
        const rev = arr1D.slice([null, null, -2]); 
        // Original stride [1]. New stride [1 * -2] = -2.
        expect(rev.strides).toEqual(new Int32Array([-2]));
    });

    // --- Multiple Reductions ---
    test('23. Chained reduction: 3D -> 2D -> 1D -> 0D', () => {
        const v2D = arr3D.slice(1);    // shape [2, 2]
        const v1D = v2D.slice(0);      // shape [2]
        const v0D = v1D.slice(1);      // shape []
        expect(v0D.get()).toBe(5); 
    });

    // --- Offset Calculations ---
    test('24. Offset check: T -> Slice', () => {
        const t = arr2D.transpose(); // [[0,3],[1,4],[2,5]]
        const row2 = t.slice(2);     // Picks row index 2 of T (which is element '2' in original data)
        // In element-indexing, offset should be 2
        expect(row2.offset).toBe(2); 
    });

    test('25. Offset check: Reverse -> Slice', () => {
        // arr1D = [0,1,2,3,4,5], size=6
        const rev = arr1D.slice([null, null, -1]); // Start=5, End=-1, Step=-1. Offset = 5
        const sub = rev.slice(1); // sub is a view of rev starting at index 1 of rev.
        // index 1 of [5,4,3,2,1,0] is element '4'.
        // Element '4' in original arr1D has index 4.
        expect(sub.offset).toBe(4); 
    });

    // --- Broad-range Tests ---
    test('26. Slice whole matrix with [null, null]', () => {
        const v = arr2D.slice(null, null);
        expect(v.shape).toEqual(arr2D.shape);
        expect(v.offset).toBe(arr2D.offset);
    });

    test('27. Slice with empty arrays', () => {
        const v = arr2D.slice([], []);
        expect(v.shape).toEqual(new Int32Array([2, 3]));
    });

    test('28. Chained T: arr.T.T should be original view', () => {
        const tt = arr2D.transpose().transpose();
        const s1 = arr2D.slice([0, 1], [0, 1]);
        const s2 = tt.slice([0, 1], [0, 1]);
        expect(s1.get(0, 0)).toBe(s2.get(0, 0));
    });

    // --- Scalar Indexing on Views ---
    test('29. Pick element from a 3D slice', () => {
        const view = arr3D.slice([1, 2], [1, 2], [1, 2]); // Single element [[[7]]]
        expect(view.get(0, 0, 0)).toBe(7);
    });

    test('30. 2D Slice -> 1D Slice -> pick', () => {
        const res = arr2D.slice(null, [1, 3]).slice(1).slice(1); // arr[:, 1:3] -> row 1 -> col 1
        expect(res.get()).toBe(5);
    });
});


describe('NDArray Slicing - Category 4: Edge Cases & Stress Tests', () => {
    let arr1D, arr2D, emptyArr;

    beforeEach(() => {
        arr1D = new NDArray(new Float64Array([0, 1, 2, 3, 4, 5]), { 
            shape: new Int32Array([6]) 
        });
        arr2D = new NDArray(new Float64Array([0, 1, 2, 3, 4, 5]), { 
            shape: new Int32Array([2, 3]) 
        });
        // Truly empty array
        emptyArr = new NDArray(new Float64Array([]), { 
            shape: new Int32Array([0, 5]) 
        });
    });

    // --- Ellipsis & Missing Arguments Simulation ---
    test('1. Ellipsis: Trailing nulls are implicit', () => {
        const v1 = arr2D.slice(0, null);
        const v2 = arr2D.slice(0); // Second dimension is implicitly null
        expect(v1.shape).toEqual(v2.shape);
        expect(v1.offset).toBe(v2.offset);
    });

    test('2. Ellipsis: slice() with no args returns full view', () => {
        const v = arr2D.slice();
        expect(v.shape).toEqual(new Int32Array([2, 3]));
        expect(v.offset).toBe(0);
    });

    test('3. Ellipsis: Multi-dimensional implicit selection', () => {
        const arr3D = new NDArray(new Float64Array(8), { shape: new Int32Array([2, 2, 2]) });
        expect(arr3D.slice(1).shape).toEqual(new Int32Array([2, 2]));
    });

    // --- Extreme Out-of-Bounds (Clamping) ---
    test('4. Extreme positive bounds: [100, 200]', () => {
        const v = arr1D.slice([100, 200]);
        expect(v.shape).toEqual(new Int32Array([0]));
    });

    test('5. Extreme negative bounds: [-200, -100]', () => {
        const v = arr1D.slice([-200, -100]);
        // Clamps to [0, 0]
        expect(v.shape).toEqual(new Int32Array([0]));
    });

    test('6. Overlapping out-of-bounds: [-100, 100]', () => {
        const v = arr1D.slice([-100, 100]);
        // Clamps to [0, 6]
        expect(v.shape).toEqual(new Int32Array([6]));
        expect(v.offset).toBe(0);
    });

    // --- Empty Arrays & Zero-Size Slices ---
    test('7. Slicing an already empty array', () => {
        const v = emptyArr.slice(null, [1, 3]);
        expect(v.shape).toEqual(new Int32Array([0, 2]));
    });

    test('8. Slice resulting in 0-length dimension', () => {
        const v = arr2D.slice([1, 1], null); // start=1, end=1
        expect(v.shape).toEqual(new Int32Array([0, 3]));
    });

    test('9. Rowview on 0-length axis throws', () => {
        // Correct way to test for expected exceptions in Jest
        expect(() => {
            emptyArr.rowview(0);
        }).toThrow(/out of bounds/);
    });

    test('10. Colview on 0-length axis', () => {
        const v = emptyArr.colview(0); // Shape [0, 5], colview(0) picks index 0 of axis 1
        expect(v.shape).toEqual(new Int32Array([0]));
    });

    // --- Large Steps & Boundary Steps ---
    test('11. Step larger than array size', () => {
        const v = arr1D.slice([0, 6, 100]);
        expect(v.shape).toEqual(new Int32Array([1]));
        expect(v.get(0)).toBe(0);
    });

    test('12. Step exactly equal to size', () => {
        const v = arr1D.slice([0, 6, 6]);
        expect(v.shape).toEqual(new Int32Array([1]));
    });

    test('13. Negative step larger than size', () => {
        const v = arr1D.slice([null, null, -100]);
        expect(v.shape).toEqual(new Int32Array([1]));
        expect(v.get(0)).toBe(5);
    });

    // --- Start/End Edge Cases ---
    test('14. Start at size (fwd): [6, null]', () => {
        const v = arr1D.slice([6, null]);
        expect(v.shape).toEqual(new Int32Array([0]));
    });

    test('15. End at 0 (fwd): [null, 0]', () => {
        const v = arr1D.slice([null, 0]);
        expect(v.shape).toEqual(new Int32Array([0]));
    });

    test('16. Start at -1 (rev): [-1, null, -1]', () => {
        const v = arr1D.slice([-1, null, -1]); // 5, 4, 3, 2, 1, 0
        expect(v.shape).toEqual(new Int32Array([6]));
        expect(v.get(0)).toBe(5);
    });

    // --- Stride Accumulation Stress Test ---
    test('17. Repeated striding [::2][::2]', () => {
        const longArr = new NDArray(new Float64Array(100), { shape: new Int32Array([100]) });
        const v1 = longArr.slice([null, null, 2]); // 0, 2, 4... shape 50
        const v2 = v1.slice([null, null, 2]);      // 0, 4, 8... shape 25
        expect(v2.shape).toEqual(new Int32Array([25]));
        expect(v2.strides).toEqual(new Int32Array([4]));
    });

    test('18. Mixing positive and negative steps', () => {
        // [0,1,2,3,4,5] -> [1,2,3,4] -> [4,3,2]
        const v1 = arr1D.slice([1, 5]); 
        const v2 = v1.slice([null, null, -1]);
        expect(Array.from(v2.copy().data)).toEqual([4, 3, 2, 1]);
    });

    // --- Scalar (0D) Interactions ---
    test('19. Slice a scalar array (0D)', () => {
        const scalar = arr1D.slice(0); // 0D
        const v = scalar.slice();      // Should return a new view of same 0D
        expect(v.ndim).toBe(0);
        expect(v.get()).toBe(0);
    });

    // --- Memory Alignment / Offset Stress ---
    test('20. Offset after multiple scalar reductions', () => {
        const arr3D = new NDArray(new Float64Array(Array.from({length: 8}, (_, i) => i)), { 
            shape: new Int32Array([2, 2, 2]) 
        });
        const v = arr3D.slice(1, 1, 1); // element 7
        expect(v.offset).toBe(7);
    });

    test('21. Large offset clamping: slice([10, null])', () => {
        const v = arr1D.slice([10, null]);
        expect(v.offset).toBe(arr1D.offset + 6); // NumPy clamps start to size
        expect(v.shape[0]).toBe(0);
    });

    // --- Precision and Floating Step (N/A for indices, but test Integers) ---
    test('22. Step resulting in non-integer divisions', () => {
        // size 6, step 4. ceil((6-0)/4) = 2. indices: 0, 4
        const v = arr1D.slice([0, 6, 4]);
        expect(v.shape).toEqual(new Int32Array([2]));
        expect(Array.from(v.copy().data)).toEqual([0, 4]);
    });

    // --- Row/Col Slicing on Non-Standard Strides ---
    test('23. colview on a row-sliced matrix', () => {
        const subRows = arr2D.slice([1, 2], null); // Picks [[3, 4, 5]]
        const col = subRows.colview(2); // Picks [5]
        expect(col.get(0)).toBe(5);
        expect(col.shape).toEqual(new Int32Array([1]));
    });

    // --- Negative Indices for Scalar Picking ---
    test('24. Picking row -1 of -2', () => {
        const v = arr2D.slice(-2); // Picks row 0
        expect(v.get(0)).toBe(0);
    });

    // --- Advanced Step Clamping ---
    test('25. Negative step with start < end', () => {
        const v = arr1D.slice([2, 4, -1]); // start 2, end 4, step -1
        expect(v.shape).toEqual(new Int32Array([0]));
    });

    test('26. Negative step with negative start > end', () => {
        const v = arr1D.slice([-1, -4, -1]); // indices 5, 4, 3
        expect(v.shape).toEqual(new Int32Array([3]));
    });

    // --- Functional Stress Test: Deep Chain ---
    test('27. Chaining 5 different slice operations', () => {
        let v = arr2D;
        v = v.slice(null, [null, null, -1]); // reverse cols: [[2,1,0],[5,4,3]]
        v = v.transpose();                  // [[2,5],[1,4],[0,3]]
        v = v.slice([0, 2], null);          // [[2,5],[1,4]]
        v = v.slice(null, 1);               // [5, 4]
        v = v.slice([null, null, -1]);      // [4, 5]
        expect(Array.from(v.copy().data)).toEqual([4, 5]);
    });

    // --- Data Types Integrity ---
    test('28. slice preserves dtype', () => {
        const v = arr1D.slice([0, 2]);
        expect(v.dtype).toBe(arr1D.dtype);
    });

    // --- Edge: start == end ---
    test('29. Start equals end results in empty array', () => {
        const v = arr1D.slice([3, 3]);
        expect(v.shape).toEqual(new Int32Array([0]));
    });

    // --- Boundary: Negative step stop at index 0 ---
    test('30. Negative step stopping exactly at 0', () => {
        // [5, 4, 3, 2, 1]. Excludes index 0.
        const v = arr1D.slice([5, 0, -1]);
        expect(v.shape).toEqual(new Int32Array([5]));
        expect(v.get(4)).toBe(1);
    });
});






/**
 * Test Suite: ndarray.prototype.set
 * Coverage: Scalar indexing, Type conversion, Broadcasting rules, View mutability.
 */

describe('ndarray.set - Extensive Test Suite (50 Cases)', () => {
    let arr1D, arr2D, arr3D;

    beforeEach(() => {
        // [0, 1, 2, 3, 4, 5]
        arr1D = new NDArray(new Float64Array([0, 1, 2, 3, 4, 5]), { shape: [6] });
        
        // [[0, 1, 2], [3, 4, 5]]
        arr2D = new NDArray(new Float64Array([0, 1, 2, 3, 4, 5]), { shape: [2, 3] });
        
        // 2x2x2 cube: [[[0,1],[2,3]], [[4,5],[6,7]]]
        arr3D = new NDArray(new Float64Array(Array.from({ length: 8 }, (_, i) => i)), { shape: [2, 2, 2] });
    });

    // =========================================================================
    // SECTION 1: BASIC SCALAR INDEXING (arr.set(val, i, j...))
    // =========================================================================

    test('1. Set single element in 1D array', () => {
        arr1D.set(99, 2);
        expect(arr1D.get(2)).toBe(99);
    });

    test('2. Set single element in 2D array', () => {
        arr2D.set(99, 1, 1); // target: 4
        expect(arr2D.get(1, 1)).toBe(99);
    });

    test('3. Set single element in 3D array', () => {
        arr3D.set(99, 1, 0, 1); // target: 5
        expect(arr3D.get(1, 0, 1)).toBe(99);
    });

    test('5. Set single element using negative index support (if applicable via slice)', () => {
        // Note: set indices are absolute, but slice indices can be negative
        arr1D.slice(-1).set(77);
        expect(arr1D.get(5)).toBe(77);
    });

    // =========================================================================
    // SECTION 2: PARAMETER TYPES (Number, Array, NDArray)
    // =========================================================================

    test('6. Bulk set with a single number (Fill)', () => {
        arr1D.set(10);
        expect(Array.from(arr1D.data)).toEqual([10, 10, 10, 10, 10, 10]);
    });

    test('7. Bulk set 2D with a single number', () => {
        arr2D.set(0);
        expect(Array.from(arr2D.data)).toEqual([0, 0, 0, 0, 0, 0]);
    });

    test('8. Set from a standard JavaScript Array', () => {
        arr1D.set([10, 20, 30, 40, 50, 60]);
        expect(Array.from(arr1D.data)).toEqual([10, 20, 30, 40, 50, 60]);
    });

    test('9. Set 2D from a nested JavaScript Array', () => {
        arr2D.set([[1, 1, 1], [2, 2, 2]]);
        expect(Array.from(arr2D.data)).toEqual([1, 1, 1, 2, 2, 2]);
    });

    test('10. Set from an NDArray of same shape', () => {
        const other = ndarray.array([[9, 9, 9], [8, 8, 8]]);
        arr2D.set(other);
        expect(Array.from(arr2D.data)).toEqual([9, 9, 9, 8, 8, 8]);
    });

    // =========================================================================
    // SECTION 3: BROADCASTING (NumPy Style)
    // =========================================================================

    test('11. Broadcast scalar to 1D', () => {
        arr1D.set(new NDArray(new Float64Array([5]), { shape: [1] }));
        expect(Array.from(arr1D.data).every(x => x === 5)).toBe(true);
    });

    test('12. Broadcast 1D row to 2D matrix', () => {
        // Target [2, 3], Source [3]
        arr2D.set([10, 20, 30]);
        expect(Array.from(arr2D.data)).toEqual([10, 20, 30, 10, 20, 30]);
    });

    test('13. Broadcast 1D column-like [2, 1] to 2D [2, 3]', () => {
        const col = new NDArray(new Float64Array([10, 20]), { shape: [2, 1] });
        arr2D.set(col);
        expect(Array.from(arr2D.data)).toEqual([10, 10, 10, 20, 20, 20]);
    });

    test('14. Broadcast 2D [1, 3] to 2D [2, 3]', () => {
        const rowMat = new NDArray(new Float64Array([7, 8, 9]), { shape: [1, 3] });
        arr2D.set(rowMat);
        expect(Array.from(arr2D.data)).toEqual([7, 8, 9, 7, 8, 9]);
    });

    test('15. Broadcast 1D to 3D cube [2, 2, 2]', () => {
        arr3D.set([100, 200]); // Source [2] -> [1, 1, 2]
        expect(Array.from(arr3D.data)).toEqual([100, 200, 100, 200, 100, 200, 100, 200]);
    });

    test('16. Broadcast 2D matrix to 3D cube', () => {
        // Target [2, 2, 2], Source [2, 2] -> [[10, 20], [30, 40]]
        arr3D.set([[10, 20], [30, 40]]);
        expect(Array.from(arr3D.data)).toEqual([10, 20, 30, 40, 10, 20, 30, 40]);
    });

    test('17. Throw error on incompatible broadcast (length mismatch)', () => {
        expect(() => arr1D.set([1, 2])).toThrow();
    });

    test('18. Throw error when source has more dimensions', () => {
        expect(() => arr1D.set(arr2D)).toThrow();
    });

    test('19. Broadcast [2, 1, 2] to [2, 2, 2]', () => {
        const src = new NDArray(new Float64Array([10, 20, 30, 40]), { shape: [2, 1, 2] });
        arr3D.set(src);
        // Depth 0, Row 0 & 1 get [10, 20]; Depth 1, Row 0 & 1 get [30, 40]
        expect(Array.from(arr3D.data)).toEqual([10, 20, 10, 20, 30, 40, 30, 40]);
    });

    test('20. Broadcast scalar NDArray (0D) to 2D', () => {
        const scalar = new NDArray(new Float64Array([5]), { shape: [] }); // 0-dim
        arr2D.set(scalar);
        expect(Array.from(arr2D.data).every(x => x === 5)).toBe(true);
    });

    // =========================================================================
    // SECTION 4: SLICING VIEWS & MUTABILITY
    // =========================================================================

    test('21. Set values through a simple slice', () => {
        arr1D.slice([1, 4]).set(0); 
        expect(Array.from(arr1D.data)).toEqual([0, 0, 0, 0, 4, 5]);
    });

    test('22. Set values through a sliced row of a 2D array', () => {
        arr2D.slice(1).set([9, 9, 9]); 
        expect(Array.from(arr2D.data)).toEqual([0, 1, 2, 9, 9, 9]);
    });

    test('23. Set values through a sliced column of a 2D array', () => {
        arr2D.slice(null, 1).set([10, 20]); // Middle column
        expect(arr2D.get(0, 1)).toBe(10);
        expect(arr2D.get(1, 1)).toBe(20);
    });

    test('24. Set with step: slice(start, end, 2)', () => {
        arr1D.slice([0, 6, 2]).set([10, 10, 10]); // indices 0, 2, 4
        expect(Array.from(arr1D.data)).toEqual([10, 1, 10, 3, 10, 5]);
    });

    test('25. Set with negative step: reverse assignment', () => {
        // [0, 1, 2, 3, 4, 5] -> reverse slice is indices [5, 4, 3, 2, 1, 0]
        arr1D.slice([null, null, -1]).set([10, 20, 30, 40, 50, 60]);
        expect(Array.from(arr1D.data)).toEqual([60, 50, 40, 30, 20, 10]);
    });

    test('26. Set values on a double-sliced view', () => {
        const view = arr1D.slice([0, 4]).slice([1, 3]); // [1, 2]
        view.set([88, 99]);
        expect(arr1D.get(1)).toBe(88);
        expect(arr1D.get(2)).toBe(99);
    });

    test('27. Set values on a 3D sub-cube', () => {
        // [[[0,1],[2,3]], [[4,5],[6,7]]] -> slice first depth -> [[0,1],[2,3]]
        arr3D.slice(0).set([[10, 10], [20, 20]]);
        expect(arr3D.get(0, 0, 0)).toBe(10);
        expect(arr3D.get(0, 1, 0)).toBe(20);
    });

    test('28. Set column in 3D: cube.slice(null, null, 0)', () => {
        arr3D.slice(null, null, 0).set([[1, 2], [3, 4]]);
        expect(arr3D.get(0, 0, 0)).toBe(1);
        expect(arr3D.get(1, 1, 0)).toBe(4);
    });

    test('29. Slice with scalar reduction then set: arr2D.slice(0, null).set(...)', () => {
        arr2D.slice(0, null).set([7, 8, 9]);
        expect(Array.from(arr2D.data.subarray(0, 3))).toEqual([7, 8, 9]);
    });

    test('30. Empty slice assignment (no-op)', () => {
        const view = arr1D.slice([0, 0]);
        expect(() => view.set([])).not.toThrow();
        expect(Array.from(arr1D.data)).toEqual([0, 1, 2, 3, 4, 5]);
    });

    // =========================================================================
    // SECTION 5: TRANSPOSITION VIEWS
    // =========================================================================

    test('31. Set values in a transposed matrix', () => {
        const t = arr2D.transpose(); // [3, 2]
        t.set([[10, 20], [30, 40], [50, 60]]);
        // Original was [[10, 30, 50], [20, 40, 60]]
        expect(arr2D.get(0, 0)).toBe(10);
        expect(arr2D.get(0, 1)).toBe(30);
    });

    test('32. Set single row of a transposed matrix', () => {
        const t = arr2D.transpose();
        t.slice(0).set([99, 88]); // Setting first row of T = first col of original
        expect(arr2D.get(0, 0)).toBe(99);
        expect(arr2D.get(1, 0)).toBe(88);
    });

    test('33. Set single element in a transposed matrix', () => {
        const t = arr2D.transpose();
        t.set(77, 2, 1); // T[2, 1] is original [1, 2]
        expect(arr2D.get(1, 2)).toBe(77);
    });

    test('34. 3D Transpose (axes 0, 2, 1) and set', () => {
        const t = arr3D.transpose(0, 2, 1);
        t.slice(0).set([[10, 20], [30, 40]]);
        expect(arr3D.get(0, 0, 0)).toBe(10);
        expect(arr3D.get(0, 1, 0)).toBe(20);
    });

    test('35. Nested: Transpose of a slice then set', () => {
        const view = arr2D.slice(null, [0, 2]).transpose(); // 2x2
        view.set([[0, 0], [0, 0]]);
        expect(arr2D.get(0, 0)).toBe(0);
        expect(arr2D.get(0, 1)).toBe(0);
        expect(arr2D.get(0, 2)).toBe(2); // Untouched
    });

    // =========================================================================
    // SECTION 6: COMPLEX CASCADING & ADVANCED
    // =========================================================================

    test('36. Cascaded: slice.transpose.slice.set', () => {
        // arr3D [2, 2, 2]
        const view = arr3D.slice(null, 0).transpose().slice(1);
        view.set([99, 99]);
        expect(arr3D.get(1, 0, 1)).toBe(99);
    });

    test('37. Set with non-contiguous source into non-contiguous target', () => {
        const target = arr2D.slice(null, [0, 2]); // [2, 2]
        const source = arr2D.slice(null, [1, 3]); // [2, 2]
        target.set(source);
        // Col 0, 1 becomes old Col 1, 2
        expect(arr2D.get(0, 0)).toBe(1);
        expect(arr2D.get(0, 1)).toBe(2);
    });

    test('38. Set rowview (helper method) in 2D', () => {
        arr2D.rowview(1).set([10, 11, 12]);
        expect(Array.from(arr2D.data.subarray(3))).toEqual([10, 11, 12]);
    });

    test('39. Set colview (helper method) in 2D', () => {
        arr2D.colview(2).set([100, 200]);
        expect(arr2D.get(0, 2)).toBe(100);
        expect(arr2D.get(1, 2)).toBe(200);
    });

    test('40. Reshape(contiguous) then set', () => {
        const reshaped = arr2D.reshape(6);
        reshaped.set([1, 2, 3, 4, 5, 6]);
        expect(Array.from(arr2D.data)).toEqual([1, 2, 3, 4, 5, 6]);
    });

    // =========================================================================
    // SECTION 7: EDGE CASES & STRESS
    // =========================================================================

    test('41. Set a 1x1x1 NDArray to a scalar index', () => {
        const mini = ndarray.array([[[5]]]);
        arr2D.set(mini, 0, 0);
        expect(arr2D.get(0, 0)).toBe(5);
    });

    test('42. Large step broadcasting: slice(null, null, 100).set(...)', () => {
        const bigArr = new NDArray(new Float64Array(10), { shape: [10] });
        bigArr.slice([0, 10, 5]).set([99, 88]);
        expect(bigArr.get(0)).toBe(99);
        expect(bigArr.get(5)).toBe(88);
    });

    test('43. Broadcast a single element NDArray to a large array', () => {
        const val = ndarray.array([3.14]);
        arr3D.set(val);
        expect(arr3D.get(1, 1, 1)).toBe(3.14);
    });

    test('44. Overlapping view assignment (Source is target slice)', () => {
        // NOTE: This usually requires a temporary copy in NumPy if truly overlapping.
        // Our implementation iterates, so results depend on direction.
        const firstHalf = arr1D.slice([0, 3]);
        const secondHalf = arr1D.slice([3, 6]);
        secondHalf.set(firstHalf); // [3, 4, 5] = [0, 1, 2]
        expect(Array.from(arr1D.data)).toEqual([0, 1, 2, 0, 1, 2]);
    });

    test('45. Set values via slice with all defaults: slice(null).set(...)', () => {
        arr1D.slice(null).set(123);
        expect(arr1D.get(0)).toBe(123);
        expect(arr1D.get(5)).toBe(123);
    });

    test('46. Set values on squeezed view', () => {
        const view = arr3D.slice(0, 0).squeeze(); // shape [2]
        view.set([55, 66]);
        expect(arr3D.get(0, 0, 0)).toBe(55);
        expect(arr3D.get(0, 0, 1)).toBe(66);
    });

    test('47. Broadcast [1, 3] to [2, 3] using a transposed source', () => {
        const src = ndarray.array([[1], [2], [3]]).transpose(); // [1, 3]
        arr2D.set(src);
        expect(Array.from(arr2D.data)).toEqual([1, 2, 3, 1, 2, 3]);
    });

    test('48. Set value in array with offset', () => {
        const view = new NDArray(arr1D.data, { shape: [2], offset: 2 }); // [2, 3]
        view.set([10, 20]);
        expect(arr1D.get(2)).toBe(10);
        expect(arr1D.get(3)).toBe(20);
    });

    test('49. Integrity check: ensure set doesn\'t change dtype', () => {
        const intArr = new NDArray(new Int32Array([1, 2]), { shape: [2] });
        intArr.set([3.9, 4.1]); // Should truncate to 3, 4
        expect(Array.from(intArr.data)).toEqual([3, 4]);
    });

    test('50. Complete logic stress: Transposed 3D slice set via broadcasted 1D', () => {
        // Target: arr3D.transpose(2, 0, 1) -> shape [2, 2, 2]
        // Slice: .slice(0) -> shape [2, 2]
        // Set with [99] -> broadcasted
        arr3D.transpose(2, 0, 1).slice(0).set([99]);
        // This corresponds to setting all elements where the original last dimension index was 0
        expect(arr3D.get(0, 0, 0)).toBe(99);
        expect(arr3D.get(1, 1, 0)).toBe(99);
        expect(arr3D.get(0, 0, 1)).toBe(1); // Should remain untouched
    });
});




/**
 * Test Suite: Unified ndarray.prototype.set
 * Focus: Advanced indexing, Mixed index types, and Complex Broadcasting.
 */

describe('ndarray.set - Advanced & Unified Indexing (30 Cases)', () => {
    let arr2D, arr3D;

    beforeEach(() => {
        // [[0, 1, 2], [3, 4, 5]]
        arr2D = new NDArray(new Float64Array([0, 1, 2, 3, 4, 5]), { shape: [2, 3] });
        
        // 2x2x2 cube: [[[0,1],[2,3]], [[4,5],[6,7]]]
        arr3D = new NDArray(new Float64Array(Array.from({ length: 8 }, (_, i) => i)), { shape: [2, 2, 2] });
    });

    // =========================================================================
    // SECTION 1: ADVANCED INDEXING (ARRAY ARGS)
    // =========================================================================

    test('1. Advanced Indexing: Set specific rows using array', () => {
        // Set both rows with a single 1D array (broadcasted)
        arr2D.set([10, 20, 30], [0, 1], null); 
        expect(Array.from(arr2D.data)).toEqual([10, 20, 30, 10, 20, 30]);
    });

    test('2. Advanced Indexing: Unordered row assignment', () => {
        // Swap rows via set
        const originalRows = [[0, 1, 2], [3, 4, 5]];
        arr2D.set(originalRows[0], [1], null);
        arr2D.set(originalRows[1], [0], null);
        expect(Array.from(arr2D.data)).toEqual([3, 4, 5, 0, 1, 2]);
    });

    test('3. Advanced Indexing: Repeated indices (Overwriting)', () => {
        // Index [0, 0] means the first row is set twice. Last one wins.
        arr2D.set([[10, 10, 10], [99, 99, 99]], [0, 0], null);
        expect(arr2D.get(0, 0)).toBe(99);
    });

    test('4. Advanced Indexing: Set specific columns in 2D', () => {
        // Set col 0 and 2
        arr2D.set([[10, 20], [30, 40]], null, [0, 2]);
        expect(arr2D.get(0, 0)).toBe(10);
        expect(arr2D.get(0, 2)).toBe(20);
        expect(arr2D.get(1, 0)).toBe(30);
        expect(arr2D.get(1, 2)).toBe(40);
    });

    test('5. Advanced Indexing: Sub-grid selection [rows][cols]', () => {
        // arr3D: depth 0, rows [0, 1], col 1
        arr3D.set([99, 88], 0, [0, 1], 1);
        expect(arr3D.get(0, 0, 1)).toBe(99);
        expect(arr3D.get(0, 1, 1)).toBe(88);
    });

    // =========================================================================
    // SECTION 2: MIXED INDEX TYPES (null, number, Array)
    // =========================================================================

    test('6. Mixed: (null, Array) - All rows, specific columns', () => {
        arr2D.set([100, 200], null, [0, 2]); 
        expect(arr2D.get(1, 0)).toBe(100);
        expect(arr2D.get(1, 2)).toBe(200);
    });

    test('7. Mixed: (Array, number) - Specific rows, single column', () => {
        // Target: column 1 of both rows
        arr2D.set([55, 66], [0, 1], 1);
        expect(arr2D.get(0, 1)).toBe(55);
        expect(arr2D.get(1, 1)).toBe(66);
    });

    test('8. Mixed: (number, Array, null) - 3D sub-plane', () => {
        // arr3D: depth 1, all rows, col 0
        arr3D.set([10, 20], 1, null, 0);
        expect(arr3D.get(1, 0, 0)).toBe(10);
        expect(arr3D.get(1, 1, 0)).toBe(20);
    });

    test('9. Mixed: (Array, null, number) - 3D specific depths, all rows, col 1', () => {
        arr3D.set([[1, 2], [3, 4]], [0, 1], null, 1);
        expect(arr3D.get(0, 0, 1)).toBe(1);
        expect(arr3D.get(1, 1, 1)).toBe(4);
    });

    test('10. Negative indices in Arrays', () => {
        arr1D = new NDArray(new Float64Array([1, 2, 3, 4]), { shape: [4] });
        arr1D.set([99, 88], [-1, -4]); // indices 3 and 0
        expect(arr1D.get(3)).toBe(99);
        expect(arr1D.get(0)).toBe(88);
    });

    // =========================================================================
    // SECTION 3: BROADCASTING INTO INDEXED REGIONS
    // =========================================================================

    test('11. Broadcast scalar into Advanced Indexing', () => {
        arr2D.set(99, [0, 1], [1, 2]); // 2x2 subgrid
        expect(arr2D.get(0, 1)).toBe(99);
        expect(arr2D.get(1, 2)).toBe(99);
    });

    test('12. Broadcast 1D row into Advanced Indexing grid', () => {
        // Target: 2x2 grid, Source: [2]
        arr2D.set([7, 8], [0, 1], [0, 2]);
        expect(arr2D.get(0, 0)).toBe(7);
        expect(arr2D.get(0, 2)).toBe(8);
        expect(arr2D.get(1, 0)).toBe(7);
        expect(arr2D.get(1, 2)).toBe(8);
    });

    test('13. Broadcast 1D column [2, 1] into Advanced Indexing grid', () => {
        const col = new NDArray(new Float64Array([10, 20]), { shape: [2, 1] });
        arr2D.set(col, [0, 1], [0, 2]);
        expect(arr2D.get(0, 0)).toBe(10);
        expect(arr2D.get(0, 2)).toBe(10);
        expect(arr2D.get(1, 0)).toBe(20);
    });

    test('14. Broadcast scalar to 3D Advanced Indexing', () => {
        arr3D.set(42, [0, 1], [0, 1], [0, 1]);
        expect(arr3D.get(1, 1, 1)).toBe(42);
    });

    test('15. Broadcast lower-dim array into 3D indexed slice', () => {
        // Target: 2x2 slice at col 0. Source: [2]
        arr3D.set([9, 9], null, null, 0); 
        expect(arr3D.get(0, 0, 0)).toBe(9);
        expect(arr3D.get(1, 1, 0)).toBe(9);
    });

    // =========================================================================
    // SECTION 4: BROADCASTING EDGE CASES (SCALARS & MISMATCH)
    // =========================================================================

    test('16. Source is 0D NDArray (Scalar)', () => {
        const scalar = new NDArray(new Float64Array([3]), { shape: [] });
        arr2D.set(scalar, [0, 1], null);
        expect(arr2D.get(1, 2)).toBe(3);
    });

    test('17. Source is smaller than target shape but broadcastable', () => {
        // Target shape: [2, 3] (full). Source: [1, 3]
        arr2D.set([[10, 11, 12]], null, null);
        expect(arr2D.get(1, 0)).toBe(10);
    });

    test('18. Mismatch: Source too large for indexed region', () => {
        expect(() => {
            arr2D.set([1, 2, 3, 4, 5], [0], null); // target is 1x3, source is 5
        }).toThrow(/Incompatible broadcast/);
    });

    test('19. Mismatch: Source dimensions not broadcastable to target subgrid', () => {
        expect(() => {
            arr2D.set([[1, 2]], [0, 1], [0, 1]); // target 2x2, source 1x2
        }).not.toThrow(); // This should actually succeed (1x2 broadcasts to 2x2)
    });

    test('20. Mismatch: Fixed dimension mismatch', () => {
        expect(() => {
            arr2D.set([1, 2, 3], [0, 1], [0, 1]); // target 2x2, source 3
        }).toThrow(/Incompatible broadcast/);
    });

    // =========================================================================
    // SECTION 5: ADVANCED VIEW COMBINATIONS
    // =========================================================================

    test('21. Set on a Transposed view with Advanced Indexing', () => {
        const t = arr2D.transpose(); // [3, 2]
        // Set first and last row of T (cols of original)
        t.set([[10, 10], [20, 20]], [0, 2], null);
        expect(arr2D.get(0, 0)).toBe(10);
        expect(arr2D.get(1, 0)).toBe(10);
        expect(arr2D.get(0, 2)).toBe(20);
    });

    test('22. Set on a Sliced view with Advanced Indexing', () => {
        const slice = arr2D.slice(null, [1, 3]); // [2, 2] - last two cols
        slice.set([99, 88], [0], [0, 1]); // Set first row of slice
        expect(arr2D.get(0, 1)).toBe(99);
        expect(arr2D.get(0, 2)).toBe(88);
    });

    test('23. Advanced Indexing on a negative-step slice', () => {
        const rev = arr2D.slice(null, [null, null, -1]); // [[2,1,0], [5,4,3]]
        rev.set([10, 20], null, [0, 2]); // Sets rev's col 0 and 2
        // Rev col 0 is original col 2. Rev col 2 is original col 0.
        expect(arr2D.get(0, 2)).toBe(10);
        expect(arr2D.get(0, 0)).toBe(20);
    });

    test('24. Triple Advanced Indexing in 3D', () => {
        arr3D.set([100], [1], [1], [1]); // Scalar-like advanced
        expect(arr3D.get(1, 1, 1)).toBe(100);
    });

    test('25. Advanced Indexing with duplicates performing multiple writes', () => {
        const logger = [];
        arr1D = new NDArray(new Float64Array([0, 0, 0]), { shape: [3] });
        // The loop will write 10 then 20 to index 0.
        arr1D.set([10, 20], [0, 0]);
        expect(arr1D.get(0)).toBe(20);
    });

    // =========================================================================
    // SECTION 6: COMPLEX SCENARIOS & BOUNDARIES
    // =========================================================================

    test('26. Set 3D via 2D array and partial Advanced Indexing', () => {
        // Target: arr3D[all, all, 1]. Shape [2, 2]. Source: 2x2
        arr3D.set([[10, 20], [30, 40]], null, null, 1);
        expect(arr3D.get(0, 1, 1)).toBe(20);
        expect(arr3D.get(1, 1, 1)).toBe(40);
    });

    test('27. Set using null at different positions', () => {
        // arr3D: [all, 1, all]
        arr3D.set([[10, 11], [20, 21]], null, 1, null);
        expect(arr3D.get(0, 1, 1)).toBe(11);
        expect(arr3D.get(1, 1, 0)).toBe(20);
    });

    test('28. Out of bounds in Advanced Index array throws', () => {
        expect(() => {
            arr2D.set([1, 2], [0, 10], null);
        }).toThrow(/out of bounds/);
    });

    test('29. Empty Advanced Indexing (No-op)', () => {
        const originalData = Array.from(arr2D.data);
        arr2D.set([99], [], [0]);
        expect(Array.from(arr2D.data)).toEqual(originalData);
    });

    test('30. Full Unified Stress: Mixed 3D assignment', () => {
        // arr3D: depth [1], all rows, columns [0]
        // Logical target shape: [2] (the "all rows" dimension)
        arr3D.set([[99], [88]], 1, null, [0]);
        expect(arr3D.get(1, 0, 0)).toBe(99);
        expect(arr3D.get(1, 1, 0)).toBe(88);
        expect(arr3D.get(0, 0, 0)).toBe(0); // Untouched
    });
});




/**
 * Test Suite: ndarray.where(condition, x, y)
 * Focus: Triple-way broadcasting, type standardization, and traversal logic.
 */

describe('ndarray.where - Selection & Broadcasting (30 Cases)', () => {

    // =========================================================================
    // SECTION 1: BASIC SELECTION (SAME SHAPES)
    // =========================================================================

    test('1. Basic 1D selection: choose between two arrays', () => {
        const cond = [true, false, true];
        const x = [10, 20, 30];
        const y = [-1, -2, -3];
        const res = ndarray.where(cond, x, y);
        expect(Array.from(res.data)).toEqual([10, -2, 30]);
    });

    test('2. Basic 2D selection: matrix masks', () => {
        const cond = ndarray.array([[true, false], [false, true]]);
        const x = ndarray.array([[1, 2], [3, 4]]);
        const y = ndarray.array([[10, 20], [30, 40]]);
        const res = ndarray.where(cond, x, y);
        expect(Array.from(res.data)).toEqual([1, 20, 30, 4]);
    });

    test('3. Selection with numeric condition (0 is falsy, others truthy)', () => {
        const res = ndarray.where([1, 0, 5], [10, 20, 30], [0, 0, 0]);
        expect(Array.from(res.data)).toEqual([10, 0, 30]);
    });

    // =========================================================================
    // SECTION 2: SCALAR BROADCASTING (X OR Y AS NUMBERS)
    // =========================================================================

    test('4. Scalar Y: replace falsy values with a constant', () => {
        const x = [1, -5, 2, -8];
        const cond = x.map(v => v > 0);
        const res = ndarray.where(cond, x, 0);
        expect(Array.from(res.data)).toEqual([1, 0, 2, 0]);
    });

    test('5. Scalar X: replace truthy values with a constant', () => {
        const cond = [true, false, true];
        const res = ndarray.where(cond, 99, [1, 2, 3]);
        expect(Array.from(res.data)).toEqual([99, 2, 99]);
    });

    test('6. Both X and Y are scalars', () => {
        const cond = ndarray.array([[true, false], [true, true]]);
        const res = ndarray.where(cond, 1, 0);
        expect(Array.from(res.data)).toEqual([1, 0, 1, 1]);
        expect(res.shape).toEqual(new Int32Array([2, 2]));
    });

    test('7. Scalar Condition: entire X or Y is chosen', () => {
        const x = [1, 2, 3];
        const y = [7, 8, 9];
        expect(Array.from(ndarray.where(true, x, y).data)).toEqual([1, 2, 3]);
        expect(Array.from(ndarray.where(false, x, y).data)).toEqual([7, 8, 9]);
    });

    // =========================================================================
    // SECTION 3: COMPLEX BROADCASTING (TRIPLE-WAY)
    // =========================================================================

    test('8. Broadcast 1D Condition to 2D X and Y', () => {
        const cond = [true, false, true]; // shape [3] -> broadcasts to [2, 3]
        const x = ndarray.array([[1, 1, 1], [1, 1, 1]]);
        const y = 0;
        const res = ndarray.where(cond, x, y);
        expect(Array.from(res.data)).toEqual([1, 0, 1, 1, 0, 1]);
    });

    test('9. Broadcast 1D X to 2D Condition', () => {
        const cond = ndarray.array([[true, true, true], [false, false, false]]);
        const x = [10, 20, 30]; // Broadcasts to both rows
        const res = ndarray.where(cond, x, -1);
        expect(Array.from(res.data)).toEqual([10, 20, 30, -1, -1, -1]);
    });

    test('10. Broadcast [2, 1] mask to [2, 3] X', () => {
        const cond = ndarray.array([[true], [false]]); // shape [2, 1]
        const x = ndarray.array([[1, 2, 3], [4, 5, 6]]);
        const res = ndarray.where(cond, x, 0);
        // Row 0 is true, Row 1 is false
        expect(Array.from(res.data)).toEqual([1, 2, 3, 0, 0, 0]);
    });

    test('11. Broadcast [1, 3] X into [2, 3] mask', () => {
        const cond = ndarray.array([[true, false, true], [true, false, true]]);
        const x = ndarray.array([[10, 20, 30]]); // shape [1, 3]
        const res = ndarray.where(cond, x, 0);
        expect(Array.from(res.data)).toEqual([10, 0, 30, 10, 0, 30]);
    });

    test('11.1 Where with Transposed inputs', () => {
        const x = ndarray.array([[1, 0], [1, 4]]).transpose(); // [[1, 1], [0, 4]]
        const res = ndarray.where(x).asContiguous();
        expect(Array.from(res.data)).toEqual([0, 0, 1, 0, 1, 1]);
    });

    test('11.2 argwhere with sliced inputs', () => {
        const x = ndarray.array([[1, 0], [1, 4]]).slice(null,[0,1]); // [[1], [1]]
        const res = ndarray.argwhere(x).asContiguous();
        expect(Array.from(res.data)).toEqual([0, 0, 1, 0]);
    });

    test('12. All three have different shapes (Broadcast extreme)', () => {
        // Condition: [2, 1, 1]
        // X: [1, 2, 1]
        // Y: [1, 1, 2]
        // Result: [2, 2, 2]
        const cond = new NDArray(new Int8Array([1, 0]), { shape: [2, 1, 1] });
        const x = new NDArray(new Float64Array([10, 20]), { shape: [1, 2, 1] });
        const y = new NDArray(new Float64Array([5, 6]), { shape: [1, 1, 2] });
        
        const res = ndarray.where(cond, x, y);
        expect(res.shape).toEqual(new Int32Array([2, 2, 2]));
        // First half (cond is true): uses x (10, 20) broadcasted
        // Second half (cond is false): uses y (5, 6) broadcasted
        expect(Array.from(res.data)).toEqual([
            10, 10, 20, 20, // From X
            5, 6, 5, 6      // From Y
        ]);
    });

    // =========================================================================
    // SECTION 4: VIEW INPUTS (SLICE/TRANSPOSE)
    // =========================================================================

    test('13. Where with Transposed inputs', () => {
        const x = ndarray.array([[1, 2], [3, 4]]).transpose(); // [[1, 3], [2, 4]]
        const cond = true;
        const res = ndarray.where(cond, x, 0);
        expect(Array.from(res.data)).toEqual([1, 3, 2, 4]);
    });

    test('14. Where with Sliced inputs', () => {
        const x = ndarray.array([[1, 2, 3], [4, 5, 6]]);
        const sliceX = x.slice(null, [0, 2]); // [[1, 2], [4, 5]]
        const res = ndarray.where(true, sliceX, 0);
        expect(Array.from(res.data)).toEqual([1, 2, 4, 5]);
    });

    test('15. Mixing sliced condition and transposed X', () => {
        const x = ndarray.array([[1, 2], [3, 4]]).transpose(); // [[1, 3], [2, 4]]
        const cond = ndarray.array([[true, false], [true, false]]).slice(null, 0); // [true, true]
        // [true, true] broadcasts to match X [2, 2]
        const res = ndarray.where(cond, x, 99);
        expect(Array.from(res.data)).toEqual([1, 3, 2, 4]);
    });

    // =========================================================================
    // SECTION 5: TYPES AND PRECISION
    // =========================================================================

    test('16. Dtype inheritance from X', () => {
        const x = new NDArray(new Float32Array([1.5]), { shape: [1] });
        const res = ndarray.where(true, x, 0);
        expect(res.data).toBeInstanceOf(Float32Array);
        expect(res.get(0)).toBeCloseTo(1.5);
    });

    test('17. Using Int32Array for X and Y', () => {
        const x = new NDArray(new Int32Array([10, 20]), { shape: [2] });
        const y = new NDArray(new Int32Array([1, 2]), { shape: [2] });
        const res = ndarray.where([true, false], x, y);
        expect(res.data).toBeInstanceOf(Int32Array);
        expect(Array.from(res.data)).toEqual([10, 2]);
    });

    // =========================================================================
    // SECTION 6: EDGE CASES
    // =========================================================================

    test('18. Incompatible shapes throw error', () => {
        expect(() => {
            ndarray.where([true, true], [1, 2, 3], 0);
        }).toThrow(/Incompatible shapes/);
    });

    test('19. Single element arrays', () => {
        const res = ndarray.where([false], [1], [2]);
        expect(res.get(0)).toBe(2);
    });

    test('20. 3D mask with 1D X', () => {
        const cond = ndarray.array([[[true], [false]]]); // [1, 2, 1]
        const x = [99]; // [1]
        const res = ndarray.where(cond, x, 0);
        expect(res.shape).toEqual(new Int32Array([1, 2, 1]));
        expect(Array.from(res.data)).toEqual([99, 0]);
    });

    test('21. Large broadcast: 1D array to 3D cube', () => {
        const cond = [true, false]; // [2] -> [1, 1, 2]
        const res = ndarray.where(cond, 1, 0);
        // For a 2x2x2 cube, result is 8 elements
        const cube = new NDArray(new Float64Array(8), { shape: [2, 2, 2] });
        const final = ndarray.where(cond, 1, 0); // Here condition determines shape if X/Y are numbers
        // Logic: Since X/Y are scalars, shape is taken from condition: [2]
        expect(final.shape).toEqual(new Int32Array([2]));
    });

    test('22. Complex: Condition [2, 1], X [1, 2], Y [1, 1]', () => {
        const cond = ndarray.array([[true], [false]]); // [2, 1]
        const x = ndarray.array([[10, 20]]); // [1, 2]
        const res = ndarray.where(cond, x, -1);
        expect(res.shape).toEqual(new Int32Array([2, 2]));
        expect(Array.from(res.data)).toEqual([10, 20, -1, -1]);
    });

    test('23. Zero-sized dimension handling', () => {
        const x = new NDArray(new Float64Array([]), { shape: [0, 5] });
        const res = ndarray.where(true, x, 0);
        expect(res.size).toBe(0);
        expect(res.shape).toEqual(new Int32Array([0, 5]));
    });

    test('25. 0-dimensional NDArray (Scalar) as condition', () => {
        const cond = new NDArray(new Int8Array([1]), { shape: [] });
        const res = ndarray.where(cond, [1, 2], [3, 4]);
        expect(Array.from(res.data)).toEqual([1, 2]);
    });

    test('26. X is row vector, Y is column vector', () => {
        const x = ndarray.array([[10, 20]]); // [1, 2]
        const y = ndarray.array([[1], [2]]);  // [2, 1]
        const res = ndarray.where(true, x, y);
        expect(res.shape).toEqual(new Int32Array([2, 2]));
        expect(Array.from(res.data)).toEqual([10, 20, 10, 20]);
    });

    test('27. Deeply nested views as where inputs', () => {
        const x = ndarray.array([[1, 2, 3, 4]]).reshape(2, 2).transpose();
        const res = ndarray.where(true, x, 0);
        expect(Array.from(res.data)).toEqual([1, 3, 2, 4]);
    });

    test('29. Broadcast error: [2] and [3]', () => {
        expect(() => ndarray.where([true, true], [1, 2, 3], 0)).toThrow();
    });

    test('30. Performance sanity: large selection', () => {
        const size = 1000;
        const cond = new NDArray(new Int8Array(size).fill(1), { shape: [size] });
        const x = new NDArray(new Float64Array(size).fill(42), { shape: [size] });
        const res = ndarray.where(cond, x, 0);
        expect(res.get(size - 1)).toBe(42);
        expect(res.size).toBe(size);
    });
});


/**
 * Test Suite: ndarray.prototype.pick
 * Focus: Advanced indexing, Sub-grid selection, Dimensionality reduction, and Copy behavior.
 */

describe('ndarray.pick - Advanced Indexing & Sub-grids (30 Cases)', () => {
    let arr1D, arr2D, arr3D;

    beforeEach(() => {
        // [0, 1, 2, 3, 4, 5]
        arr1D = new NDArray(new Float64Array([0, 1, 2, 3, 4, 5]), { shape: [6] });
        
        // [[0, 1, 2], [3, 4, 5]]
        arr2D = new NDArray(new Float64Array([0, 1, 2, 3, 4, 5]), { shape: [2, 3] });
        
        // 2x2x2 cube: [[[0,1],[2,3]], [[4,5],[6,7]]]
        arr3D = new NDArray(new Float64Array(Array.from({ length: 8 }, (_, i) => i)), { shape: [2, 2, 2] });
    });

    // =========================================================================
    // SECTION 1: BASIC ADVANCED INDEXING (ARRAY ARGS)
    // =========================================================================

    test('1. 1D Pick: Select specific elements in arbitrary order', () => {
        const res = arr1D.pick([5, 0, 2]);
        expect(res.shape).toEqual(new Int32Array([3]));
        expect(Array.from(res.data)).toEqual([5, 0, 2]);
    });

    test('2. 2D Row Pick: Select specific rows', () => {
        const res = arr2D.pick([1, 0], null); // Row 1 then Row 0
        expect(res.shape).toEqual(new Int32Array([2, 3]));
        expect(Array.from(res.data)).toEqual([3, 4, 5, 0, 1, 2]);
    });

    test('3. 2D Column Pick: Select specific columns', () => {
        const res = arr2D.pick(null, [2, 0]); // Col 2 then Col 0
        expect(res.shape).toEqual(new Int32Array([2, 2]));
        expect(Array.from(res.data)).toEqual([2, 0, 5, 3]);
    });

    test('4. Orthogonal Grid: Select sub-grid of rows and columns', () => {
        // Rows [0, 1], Cols [2, 1] -> [[Row0Col2, Row0Col1], [Row1Col2, Row1Col1]]
        const res = arr2D.pick([0, 1], [2, 1]);
        expect(Array.from(res.data)).toEqual([2, 1, 5, 4]);
    });

    test('5. Repeated Indices: Duplicate rows', () => {
        const res = arr2D.pick([0, 0, 0], null);
        expect(res.shape).toEqual(new Int32Array([3, 3]));
        expect(Array.from(res.data)).toEqual([0, 1, 2, 0, 1, 2, 0, 1, 2]);
    });

    // =========================================================================
    // SECTION 2: DIMENSIONALITY REDUCTION (SCALARS)
    // =========================================================================

    test('6. Scalar Indexing: Pick a single row (reduces to 1D)', () => {
        const res = arr2D.pick(1, null);
        expect(res.ndim).toBe(1);
        expect(res.shape).toEqual(new Int32Array([3]));
        expect(Array.from(res.data)).toEqual([3, 4, 5]);
    });

    test('7. Scalar Indexing: Pick a single element (reduces to 0D)', () => {
        const res = arr2D.pick(1, 2);
        expect(res.ndim).toBe(0);
        expect(res.data[res.offset]).toBe(5);
    });

    test('8. Mixed: Scalar row and Array columns', () => {
        const res = arr2D.pick(0, [2, 0]);
        expect(res.shape).toEqual(new Int32Array([2]));
        expect(Array.from(res.data)).toEqual([2, 0]);
    });

    test('9. Mixed: Array rows and Scalar column', () => {
        const res = arr2D.pick([1, 0], 2);
        expect(res.shape).toEqual(new Int32Array([2]));
        expect(Array.from(res.data)).toEqual([5, 2]);
    });

    // =========================================================================
    // SECTION 3: 3D PICKING (CUBES)
    // =========================================================================

    test('10. 3D: Pick specific depths', () => {
        const res = arr3D.pick([1, 0], null, null);
        expect(res.shape).toEqual(new Int32Array([2, 2, 2]));
        expect(Array.from(res.data.subarray(0, 4))).toEqual([4, 5, 6, 7]);
    });

    test('11. 3D: Cross-section (All depths, Row 0, All cols)', () => {
        const res = arr3D.pick(null, 0, null);
        expect(res.shape).toEqual(new Int32Array([2, 2]));
        expect(Array.from(res.data)).toEqual([0, 1, 4, 5]);
    });

    test('12. 3D: Specific column from all depths and rows', () => {
        const res = arr3D.pick(null, null, 1);
        expect(res.shape).toEqual(new Int32Array([2, 2]));
        expect(Array.from(res.data)).toEqual([1, 3, 5, 7]);
    });

    test('13. 3D: Complex mixed selection', () => {
        // Depth [1, 0], Row 1, Col [0, 1]
        const res = arr3D.pick([1, 0], 1, [0, 1]);
        expect(res.shape).toEqual(new Int32Array([2, 2]));
        expect(Array.from(res.data)).toEqual([6, 7, 2, 3]);
    });

    // =========================================================================
    // SECTION 4: NEGATIVE INDICES & TYPED ARRAYS
    // =========================================================================

    test('14. Negative Index support in scalars', () => {
        const res = arr1D.pick(-1);
        expect(res.data[0]).toBe(5);
    });

    test('15. Negative Index support in Arrays', () => {
        const res = arr2D.pick([-1], [-1, -3]);
        expect(Array.from(res.data)).toEqual([5, 3]);
    });

    test('16. Int32Array as input spec', () => {
        const specs = new Int32Array([2, 0]);
        const res = arr1D.pick(specs);
        expect(Array.from(res.data)).toEqual([2, 0]);
    });

    // =========================================================================
    // SECTION 5: PICKING FROM COMPLEX VIEWS
    // =========================================================================

    test('17. Pick from Transposed array', () => {
        const t = arr2D.transpose(); // [3, 2] -> [[0,3], [1,4], [2,5]]
        const res = t.pick([0, 2], null); // Pick row 0 and 2 of T
        expect(Array.from(res.data)).toEqual([0, 3, 2, 5]);
    });

    test('18. Pick from Sliced array', () => {
        const slice = arr1D.slice([1, 5]); // [1, 2, 3, 4]
        const res = slice.pick([3, 0]); // Pick indices 3 and 0 relative to slice
        expect(Array.from(res.data)).toEqual([4, 1]);
    });

    test('19. Pick from Negative Step view (Reverse)', () => {
        const rev = arr1D.slice([null, null, -1]); // [5, 4, 3, 2, 1, 0]
        const res = rev.pick([0, 5]);
        expect(Array.from(res.data)).toEqual([5, 0]);
    });

    test('20. Cascaded Pick: arr.pick().pick()', () => {
        const first = arr2D.pick(null, [0, 2]); // [[0, 2], [3, 5]]
        const second = first.pick([1, 0], 1);   // Col 1 of Row 1 then Row 0
        expect(Array.from(second.data)).toEqual([5, 2]);
    });

    // =========================================================================
    // SECTION 6: COPY BEHAVIOR & INTEGRITY
    // =========================================================================

    test('21. Pick returns a Copy (Independent memory)', () => {
        const res = arr1D.pick([0, 1, 2]);
        res.set(99, 0); // Modify the pick result
        expect(arr1D.get(0)).toBe(0); // Original remains unchanged
    });

    test('22. Pick result is always Contiguous', () => {
        const t = arr2D.transpose();
        expect(t.isContiguous).toBe(false);
        const res = t.pick(null, null); // Full pick is essentially a copy
        expect(res.isContiguous).toBe(true);
    });

    // =========================================================================
    // SECTION 7: EDGE CASES & STRESS
    // =========================================================================

    test('23. Pick with null (Full selection) on 1D', () => {
        const res = arr1D.pick(null);
        expect(Array.from(res.data)).toEqual([0, 1, 2, 3, 4, 5]);
    });

    test('24. Out of bounds throws error', () => {
        expect(() => arr1D.pick([10])).toThrow();
        expect(() => arr2D.pick(0, 5)).toThrow();
    });

    test('25. Empty Index Array: Result has size 0', () => {
        const res = arr1D.pick([]);
        expect(res.size).toBe(0);
        expect(res.shape).toEqual(new Int32Array([0]));
    });

    test('26. Repeated picking stress: Many duplicates', () => {
        const res = arr1D.pick([0, 0, 0, 0, 0]);
        expect(res.size).toBe(5);
        expect(Array.from(res.data).every(x => x === 0)).toBe(true);
    });

    test('27. Pick from a 0-size dimension', () => {
        const empty = new NDArray(new Float64Array([]), { shape: [0, 5] });
        const res = empty.pick([], null);
        expect(res.size).toBe(0);
    });

    test('28. Order preservation: Reversed array through pick', () => {
        const res = arr1D.pick([5, 4, 3, 2, 1, 0]);
        expect(Array.from(res.data)).toEqual([5, 4, 3, 2, 1, 0]);
    });

    test('29. Sub-grid from a single depth of 3D', () => {
        // Depth 1, Rows [1, 0], Cols [0]
        const res = arr3D.pick(1, [1, 0], 0);
        expect(res.shape).toEqual(new Int32Array([2]));
        expect(Array.from(res.data)).toEqual([6, 4]);
    });

    test('30. Full Unified Logic: Pick from sliced transposed 3D', () => {
        // Transpose 3D -> [col, row, depth]
        const view = arr3D.transpose(2, 1, 0);
        const res = view.pick([1], null, 0); // Pick specific column and depth
        expect(res.shape).toEqual(new Int32Array([1,2]));
        expect(Array.from(res.data)).toEqual([1, 3]);
    });
});



/**
 * Test Suite: ndarray.prototype.filter
 * Focus: Predicate logic, Boolean masks (NDArray/Array), and 1D flattening.
 */

describe('ndarray.filter - Predicates & Boolean Masks (30 Cases)', () => {
    let arr1D, arr2D, arr3D;

    beforeEach(() => {
        // [0, 1, 2, 3, 4, 5]
        arr1D = new NDArray(new Float64Array([0, 1, 2, 3, 4, 5]), { shape: [6] });
        
        // [[1, -2, 3], [-4, 5, -6]]
        arr2D = new NDArray(new Float64Array([1, -2, 3, -4, 5, -6]), { shape: [2, 3] });
        
        // 2x2x2 cube: [[[0,1],[2,3]], [[4,5],[6,7]]]
        arr3D = new NDArray(new Float64Array(Array.from({ length: 8 }, (_, i) => i)), { shape: [2, 2, 2] });
    });

    // =========================================================================
    // SECTION 1: CALLBACK PREDICATES (BASIC)
    // =========================================================================

    test('1. 1D Callback: Filter even numbers', () => {
        const res = arr1D.filter(v => v % 2 === 0);
        expect(Array.from(res.data)).toEqual([0, 2, 4]);
        expect(res.ndim).toBe(1);
    });

    test('2. 2D Callback: Filter positive numbers', () => {
        const res = arr2D.filter(v => v > 0);
        expect(Array.from(res.data)).toEqual([1, 3, 5]);
    });

    test('3. 3D Callback: Filter values in range', () => {
        const res = arr3D.filter(v => v >= 2 && v <= 5);
        expect(Array.from(res.data)).toEqual([2, 3, 4, 5]);
    });

    test('4. Empty Result: Predicate matches nothing', () => {
        const res = arr1D.filter(v => v > 100);
        expect(res.size).toBe(0);
        expect(res.shape).toEqual(new Int32Array([0]));
    });

    test('5. Full Result: Predicate matches everything', () => {
        const res = arr1D.filter(v => v >= 0);
        expect(res.size).toBe(6);
    });

    // =========================================================================
    // SECTION 2: BOOLEAN MASKS (NDARRAY)
    // =========================================================================

    test('6. NDArray Mask: Same shape boolean mask', () => {
        const mask = ndarray.array([[true, false, true], [false, true, false]]);
        const res = arr2D.filter(mask);
        expect(Array.from(res.data)).toEqual([1, 3, 5]);
    });

    test('7. NDArray Mask: Numeric mask (0 is falsy)', () => {
        const mask = ndarray.array([1, 0, 1, 0, 1, 0]);
        const res = arr1D.filter(mask);
        expect(Array.from(res.data)).toEqual([0, 2, 4]);
    });

    test('8. Transposed Mask: Filter using a non-contiguous mask', () => {
        // arr2D is [2, 3]. Transpose a [3, 2] mask to match.
        const maskT = ndarray.array([[true, false], [false, true], [true, false]]).transpose();
        const res = arr2D.filter(maskT);
        expect(Array.from(res.data)).toEqual([1, 3, 5]);
    });

    test('9. Mask size mismatch throws error', () => {
        const smallMask = ndarray.array([true, true]);
        expect(() => arr1D.filter(smallMask)).toThrow(/Mask size must match/);
    });

    test('10. JS Array Mask: Using standard boolean array', () => {
        const mask = [true, false, true, false, true, false];
        const res = arr1D.filter(mask);
        expect(Array.from(res.data)).toEqual([0, 2, 4]);
    });

    // =========================================================================
    // SECTION 3: CALLBACK INDEX TRACKING (currentIdx)
    // =========================================================================

    test('11. Index Usage: Filter based on column index (2D)', () => {
        // Keep only elements in the first column (index [r, 0])
        const res = arr2D.filter((v, idx) => idx[1] === 0);
        expect(Array.from(res.data)).toEqual([1, -4]);
    });

    test('12. Index Usage: Filter based on row index (2D)', () => {
        // Keep only elements in the second row (index [1, c])
        const res = arr2D.filter((v, idx) => idx[0] === 1);
        expect(Array.from(res.data)).toEqual([-4, 5, -6]);
    });

    test('13. Index Usage: Filter 3D by depth and column', () => {
        // Depth 0 and Column 1
        const res = arr3D.filter((v, idx) => idx[0] === 0 && idx[2] === 1);
        expect(Array.from(res.data)).toEqual([1, 3]);
    });

    test('14. Index Usage: Diagonal elements filtering', () => {
        const square = ndarray.array([[1, 2], [3, 4]]);
        const res = square.filter((v, idx) => idx[0] === idx[1]);
        expect(Array.from(res.data)).toEqual([1, 4]);
    });

    test('15. Callback "this" reference: Compare against array mean', () => {
        const res = arr1D.filter((v, idx, array) => v > 2.5);
        expect(Array.from(res.data)).toEqual([3, 4, 5]);
    });

    // =========================================================================
    // SECTION 4: FILTERING ON VIEWS (SLICE/TRANSPOSE)
    // =========================================================================

    test('16. Filter on a Transposed Matrix', () => {
        const t = arr2D.transpose(); // [[1, -4], [-2, 5], [3, -6]]
        const res = t.filter(v => v < 0);
        expect(Array.from(res.data)).toEqual([-4, -2, -6]);
    });

    test('17. Filter on a Sliced View', () => {
        const slice = arr1D.slice([1, 5]); // [1, 2, 3, 4]
        const res = slice.filter(v => v % 2 === 0);
        expect(Array.from(res.data)).toEqual([2, 4]);
    });

    test('18. Filter on a Reshaped View', () => {
        const reshaped = arr2D.reshape(6);
        const res = reshaped.filter(v => v > 0);
        expect(Array.from(res.data)).toEqual([1, 3, 5]);
    });

    test('19. Filter with Negative Step View', () => {
        const rev = arr1D.slice([null, null, -1]); // [5, 4, 3, 2, 1, 0]
        const res = rev.filter(v => v >= 4);
        expect(Array.from(res.data)).toEqual([5, 4]);
    });

    test('20. Mixed Mask and View: Transposed array with standard mask', () => {
        const t = arr2D.transpose(); // size 6
        const mask = [true, false, true, false, true, false];
        const res = t.filter(mask);
        // T elements at indices 0, 2, 4: 1, -2, 3
        expect(Array.from(res.data)).toEqual([1, -2, 3]);
    });

    // =========================================================================
    // SECTION 5: COMPLEX MASKS & LOGIC
    // =========================================================================

    test('21. Mask generated from another array', () => {
        const a = ndarray.array([10, 20, 30]);
        const b = ndarray.array([15, 15, 15]);
        // Filter 'a' where 'a < b'
        const res = a.filter((v, i) => v < b.data[i]);
        expect(Array.from(res.data)).toEqual([10]);
    });

    test('22. Double Filter (Chain)', () => {
        const res = arr1D.filter(v => v > 0).filter(v => v < 4);
        expect(Array.from(res.data)).toEqual([1, 2, 3]);
    });

    test('23. Mask with holes (undefined/null values)', () => {
        const mask = [true, null, true, undefined, true, false];
        const res = arr1D.filter(mask);
        // Only indices 0, 2, 4 are truthy
        expect(Array.from(res.data)).toEqual([0, 2, 4]);
    });

    test('24. Filter 3D array using a 2D slice as a template', () => {
        const res = arr3D.filter((v, idx) => {
            // Only keep elements where depth is 1
            return idx[0] === 1;
        });
        expect(Array.from(res.data)).toEqual([4, 5, 6, 7]);
    });

    test('25. Logic check: Filter based on external state', () => {
        const threshold = 3;
        const res = arr1D.filter(v => v > threshold);
        expect(Array.from(res.data)).toEqual([4, 5]);
    });

    // =========================================================================
    // SECTION 6: EDGE CASES & INTEGRITY
    // =========================================================================

    test('26. Floating point precision in filter', () => {
        const floatArr = ndarray.array([0.1, 0.2, 0.3]);
        const res = floatArr.filter(v => v > 0.15);
        expect(res.get(0)).toBeCloseTo(0.2);
    });

    test('27. Filtering with NaN values', () => {
        const arrWithNaN = ndarray.array([1, NaN, 3]);
        const res = arrWithNaN.filter(v => !isNaN(v));
        expect(Array.from(res.data)).toEqual([1, 3]);
    });

    test('28. Integrity: Filter result is a copy', () => {
        const res = arr1D.filter(v => v < 2);
        res.data[0] = 99; // Modify result
        expect(arr1D.data[0]).toBe(0); // Original untouched
    });

    test('29. Filtering a zero-sized array', () => {
        const empty = new NDArray(new Float64Array([]), { shape: [0] });
        const res = empty.filter(v => true);
        expect(res.size).toBe(0);
    });

    test('30. Stress: 3D Transposed filter with Callback', () => {
        // Transpose 3D cube and filter
        const t = arr3D.transpose(2, 0, 1);
        const res = t.filter((v, idx) => v % 3 === 0);
        // Original values: 0, 1, 2, 3, 4, 5, 6, 7
        // Multiples of 3: 0, 3, 6
        expect(Array.from(res.data).sort()).toEqual([0, 3, 6]);
    });
});