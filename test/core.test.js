const ndarray = require('../dist/ndarray.cjs');

describe('NDArray Core', () => {

    test('ndarray.zeros', () => {
        const shape = [2, 3];
        const arr = ndarray.zeros(shape);
        expect(arr.shape).toEqual(new Int32Array([2, 3]));
        expect(arr.size).toBe(6);
        expect(arr.dtype).toBe('float64');
        expect(arr.data).toEqual(new Float64Array([0, 0, 0, 0, 0, 0]));
    });

    test('ndarray.ones', () => {
        const shape = [2, 2];
        const arr = ndarray.ones(shape, 'int32');
        expect(arr.shape).toEqual(new Int32Array([2, 2]));
        expect(arr.size).toBe(4);
        expect(arr.dtype).toBe('int32');
        expect(arr.data).toEqual(new Int32Array([1, 1, 1, 1]));
    });

    test('ndarray.full', () => {
        const arr = ndarray.full([2, 2], 5);
        expect(arr.data).toEqual(new Float64Array([5, 5, 5, 5]));
    });

    test('ndarray.arange', () => {
        const arr = ndarray.arange(5);
        expect(arr.shape).toEqual(new Int32Array([5]));
        expect(arr.data).toEqual(new Float64Array([0, 1, 2, 3, 4]));

        const arr2 = ndarray.arange(2, 7, 2); // start, stop, step
        expect(arr2.data).toEqual(new Float64Array([2, 4, 6]));
    });

    test('ndarray.linspace', () => {
        const arr = ndarray.linspace(0, 10, 5); // start, stop, num
        expect(arr.shape).toEqual(new Int32Array([5]));
        expect(arr.data).toEqual(new Float64Array([0, 2.5, 5, 7.5, 10]));
    });

    test('ndarray.array (from nested array)', () => {
        const source = [[1, 2], [3, 4]];
        const arr = ndarray.array(source);
        expect(arr.shape).toEqual(new Int32Array([2, 2]));
        expect(arr.strides).toEqual(new Int32Array([2, 1]));
        expect(arr.data).toEqual(new Float64Array([1, 2, 3, 4]));
    });

    test('Contiguity check', () => {
        const arr = ndarray.zeros([2, 3, 4]);
        expect(arr.isContiguous).toBe(true);
    });
});



describe('NDArray.set - JIT Complex Scenarios Suite', () => {
    let arr1D, arr2D, arr3D;

    beforeEach(() => {
        // 1D: [0, 1, 2, 3, 4, 5]
        arr1D = ndarray.array([0, 1, 2, 3, 4, 5]);
        
        // 2D: [[0, 1, 2], [3, 4, 5]]
        arr2D = ndarray.array([0, 1, 2, 3, 4, 5]).reshape([2, 3]);
        
        // 3D 2x2x2 cube: [[[0,1],[2,3]], [[4,5],[6,7]]]
        // Flat data: [0, 1, 2, 3, 4, 5, 6, 7]
        arr3D = ndarray.array([0, 1, 2, 3, 4, 5, 6, 7]).reshape([2, 2, 2]);
    });

    test('1. 3D Mixed: Fancy index + Scalar + Wildcard', () => {
        // Target: arr3D[[0], 1, :] -> Level 0, Row 1, All columns -> elements [2, 3]
        arr3D.set([99, 88], [0], 1, null);
        // Entire data check: only index 2 and 3 should change
        expect(Array.from(arr3D.data)).toEqual([0, 1, 99, 88, 4, 5, 6, 7]);
    });

    test('2. 3D Broadcasting: Scalar value to a 2D sub-view', () => {
        // Target: arr3D[1, :, :] -> Entire second level (indices 4, 5, 6, 7)
        arr3D.set(10, 1, null, null);
        expect(Array.from(arr3D.data)).toEqual([0, 1, 2, 3, 10, 10, 10, 10]);
    });

    test('3. Transposed Source into 2D Sub-selection', () => {
        // arr2D is [[0, 1, 2], [3, 4, 5]]
        // src is [[10], [20]] (2x1) -> transpose -> [[10, 20]] (1x2)
        const src = ndarray.array([[10], [20]]).transpose(); 
        // Target: first row, columns 1 and 2
        arr2D.set(src, 0, [1, 2]); 
        expect(Array.from(arr2D.data)).toEqual([0, 10, 20, 3, 4, 5]);
    });

    test('4. Broadcasting 1D array to 3D selection grid', () => {
        // Target: arr3D[:, :, 0] -> selects (0,0,0), (0,1,0), (1,0,0), (1,1,0)
        // Logical target shape is [2, 2]. Source is [2].
        // Source [8, 9] broadcasts across the 2x2 grid rows.
        arr3D.set([8, 9], null, null, 0); 
        expect(Array.from(arr3D.data)).toEqual([8, 1, 9, 3, 8, 5, 9, 7]);
    });

    test('5. Fancy Indexing with Negative indices', () => {
        // arr1D: [0, 1, 2, 3, 4, 5]
        // Set at index 5 and 3 using negative notation
        arr1D.set([55, 33], [-1, -3]);
        expect(Array.from(arr1D.data)).toEqual([0, 1, 2, 33, 4, 55]);
    });

    test('6. Non-contiguous View as Source', () => {
        // Create a view with stride: [0, 2, 4] from arr1D
        const src = ndarray.array([0, 1, 2, 3, 4, 5]).slice([0, 6, 2]); 
        // Set arr2D first row (3 elements) using this view
        arr2D.set(src, 0);
        expect(Array.from(arr2D.data)).toEqual([0, 2, 4, 3, 4, 5]);
    });

    test('7. Set values on a Transposed Target View', () => {
        // arr2D (2x3) transposed -> (3x2 view)
        const view = arr2D.transpose();
        // Values: [[10,10], [20,20], [30,30]]
        view.set(ndarray.array([[10, 10], [20, 20], [30, 30]]));
        // Verify physical data: Row-major becomes Column-major in data
        expect(Array.from(arr2D.data)).toEqual([10, 20, 30, 10, 20, 30]);
    });

    test('8. 3D Deep Selection with Triple Scalar indices', () => {
        // arr3D[1, 1, 1] = 999 (Point selection)
        arr3D.set(999, 1, 1, 1);
        expect(Array.from(arr3D.data)).toEqual([0, 1, 2, 3, 4, 5, 6, 999]);
    });

    test('9. Broadcasting 0-dim Source (Scalar View)', () => {
        // Source is a view of a single element (shape [])
        const scalarView = ndarray.array([42]).reshape([]);
        // arr3D[:, 0, 0] = 42
        arr3D.set(scalarView, null, 0, 0);
        expect(Array.from(arr3D.data)).toEqual([42, 1, 2, 3, 42, 5, 6, 7]);
    });

    test('10. 2D Grid-style Fancy Indexing (Double Advanced)', () => {
        // targetShape = [pSet0.length, pSet1.length] = [2, 2]
        // Selected points: (0,0), (0,2), (1,0), (1,2)
        arr2D.set([[11, 22], [33, 44]], [0, 1], [0, 2]);
        expect(Array.from(arr2D.data)).toEqual([11, 1, 22, 33, 4, 44]);
    });

    test('11. Sliced view with Offset as Target', () => {
        // arr1D: [0, 1, 2, 3, 4, 5]
        // View starting at index 2, length 2: [2, 3]
        const view = arr1D.slice([2, 4]);
        view.set([77, 88]);
        expect(Array.from(arr1D.data)).toEqual([0, 1, 77, 88, 4, 5]);
    });

    test('12. Broadcast [1, 2] source into [2, 2] 3D sub-selection', () => {
        // Target: arr3D[:, :, 1] -> (0,0,1), (0,1,1), (1,0,1), (1,1,1) -> shape [2, 2]
        // Source: [[100, 200]] -> shape [1, 2]
        const src = ndarray.array([100, 200]).reshape([1, 2]);
        arr3D.set(src, null, null, 1);
        expect(Array.from(arr3D.data)).toEqual([0, 100, 2, 200, 4, 100, 6, 200]);
    });

    test('13. Broadcast single Scalar to whole 3D array', () => {
        arr3D.set(5);
        expect(Array.from(arr3D.data)).toEqual([5, 5, 5, 5, 5, 5, 5, 5]);
    });

    test('14. Overlapping view - Copy sub-slice within same array', () => {
        // Copy [0, 1] to [4, 5]
        const src = arr1D.slice([0, 2]);
        arr1D.set(src, [4, 5]);
        expect(Array.from(arr1D.data)).toEqual([0, 1, 2, 3, 0, 1]);
    });

    test('15. Fancy indexing with Duplicate indices (Last-write wins)', () => {
        // Indices: [0, 0, 0]. Final value 30 should be at index 0.
        arr1D.set([10, 20, 30], [0, 0, 0]);
        expect(arr1D.data[0]).toBe(30);
        expect(Array.from(arr1D.data)).toEqual([30, 1, 2, 3, 4, 5]);
    });

    test('16. Setting values on a Reshaped View', () => {
        // arr1D: [0..5] -> reshaped view [3, 2]
        const view = arr1D.reshape([3, 2]);
        // Set middle row
        view.set([88, 99], 1);
        expect(Array.from(arr1D.data)).toEqual([0, 1, 88, 99, 4, 5]);
    });

    test('17. Fancy selection with Squeezed source', () => {
        // Target: arr3D[[0, 1], 0, 0] -> selects (0,0,0) and (1,0,0)
        // Source: [50, 60]
        arr3D.set(ndarray.array([50, 60]), [0, 1], 0, 0);
        expect(Array.from(arr3D.data)).toEqual([50, 1, 2, 3, 60, 5, 6, 7]);
    });

    test('18. Large stride View (Step slicing emulation)', () => {
        // Select every 3rd element: [0, 3]
        const sparseView = arr1D.slice([0, 6, 3]); 
        sparseView.set([111, 222]);
        expect(Array.from(arr1D.data)).toEqual([111, 1, 2, 222, 4, 5]);
    });

    test('19. Complex broadcast: Source [2, 1] into [2, 3] selection', () => {
        // Source: [[10], [20]]
        const src = ndarray.array([[10], [20]]); 
        // Each row of arr2D gets the corresponding value from src
        arr2D.set(src); 
        expect(Array.from(arr2D.data)).toEqual([10, 10, 10, 20, 20, 20]);
    });

    test('20. Integrity: Setting values at boundaries of 3D', () => {
        // arr3D first and last element
        arr3D.set(100, 0, 0, 0);
        arr3D.set(200, 1, 1, 1);
        expect(Array.from(arr3D.data)).toEqual([100, 1, 2, 3, 4, 5, 6, 200]);
    });
});


describe('arg min max', () => {
    test('argmin', () => {
        const arr = ndarray.array([1, 2, 3, -1]);
        expect(arr.argmin()).toBe(3);
    });

    test('argmax', () => {
        const arr = ndarray.array([1, 2, 0, -1]);
        expect(arr.argmax()).toBe(1);
    });
});