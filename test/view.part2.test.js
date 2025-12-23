const ndarray = require('../dist/ndarray.cjs');
const { concat } = ndarray;

describe('concat (Comprehensive Tests)', () => {
    // Category 1: Basic Functionality
    test('1. Should concatenate two 1D arrays', () => {
        const a = ndarray.array([1, 2]);
        const b = ndarray.array([3, 4]);
        const c = concat([a, b]);
        expect(c.toArray()).toEqual([1, 2, 3, 4]);
        expect(c.shape).toEqual(new Int32Array([4]));
    });

    test('2. Should concatenate three 1D arrays', () => {
        const a = ndarray.array([1]);
        const b = ndarray.array([2, 3]);
        const c = ndarray.array([4, 5, 6]);
        const d = concat([a, b, c]);
        expect(d.toArray()).toEqual([1, 2, 3, 4, 5, 6]);
    });

    test('3. Should concatenate two 2D arrays on axis=0', () => {
        const a = ndarray.array([[1, 2], [3, 4]]);
        const b = ndarray.array([[5, 6]]);
        const c = concat([a, b], 0);
        expect(c.toArray()).toEqual([[1, 2], [3, 4], [5, 6]]);
        expect(c.shape).toEqual(new Int32Array([3, 2]));
    });

    test('4. Should concatenate two 2D arrays on axis=1', () => {
        const a = ndarray.array([[1, 2], [3, 4]]);
        const b = ndarray.array([[5], [6]]);
        const c = concat([a, b], 1);
        expect(c.toArray()).toEqual([[1, 2, 5], [3, 4, 6]]);
        expect(c.shape).toEqual(new Int32Array([2, 3]));
    });

    test('5. Should concatenate three 2D arrays on axis=0', () => {
        const a = ndarray.array([[1, 2]]);
        const b = ndarray.array([[3, 4], [5, 6]]);
        const c = ndarray.array([[7, 8]]);
        const d = concat([a, b, c], 0);
        expect(d.toArray()).toEqual([[1, 2], [3, 4], [5, 6], [7, 8]]);
    });

    test('6. Should concatenate three 2D arrays on axis=1', () => {
        const a = ndarray.array([[1], [2]]);
        const b = ndarray.array([[3, 4], [5, 6]]);
        const c = ndarray.array([[7], [8]]);
        const d = concat([a, b, c], 1);
        expect(d.toArray()).toEqual([[1, 3, 4, 7], [2, 5, 6, 8]]);
    });

    test('7. Should concatenate two 3D arrays on axis=0', () => {
        const a = ndarray.ones([1, 2, 2]);
        const b = ndarray.zeros([2, 2, 2]);
        const c = concat([a, b], 0);
        expect(c.shape).toEqual(new Int32Array([3, 2, 2]));
        expect(c.toArray()).toEqual([[[1, 1], [1, 1]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]);
    });

    test('8. Should concatenate two 3D arrays on axis=1', () => {
        const a = ndarray.ones([2, 1, 2]);
        const b = ndarray.zeros([2, 2, 2]);
        const c = concat([a, b], 1);
        expect(c.shape).toEqual(new Int32Array([2, 3, 2]));
        expect(c.toArray()).toEqual([[[1, 1], [0, 0], [0, 0]], [[1, 1], [0, 0], [0, 0]]]);
    });

    test('9. Should concatenate two 3D arrays on axis=2', () => {
        const a = ndarray.ones([2, 2, 1]);
        const b = ndarray.zeros([2, 2, 2]);
        const c = concat([a, b], 2);
        expect(c.shape).toEqual(new Int32Array([2, 2, 3]));
        expect(c.toArray()).toEqual([[[1, 0, 0], [1, 0, 0]], [[1, 0, 0], [1, 0, 0]]]);
    });

    // Category 2: Edge Cases
    test('10. Should concatenate an array with an empty array', () => {
        const a = ndarray.array([[1, 2]]);
        const b = ndarray.zeros([0, 2]);
        const c = concat([a, b], 0);
        expect(c.toArray()).toEqual([[1, 2]]);
    });

    test('11. Should handle concatenating only empty arrays', () => {
        const a = ndarray.zeros([0, 2, 3]);
        const b = ndarray.zeros([0, 2, 3]);
        const c = concat([a, b], 0);
        expect(c.shape).toEqual(new Int32Array([0, 2, 3]));
        expect(c.size).toBe(0);
    });

    test('12. Should concatenate arrays with zeros', () => {
        const a = ndarray.array([0, 1]);
        const b = ndarray.array([0, 2]);
        const c = concat([a, b]);
        expect(c.toArray()).toEqual([0, 1, 0, 2]);
    });

    test('13. Should concatenate arrays with negative numbers', () => {
        const a = ndarray.array([-1, -2]);
        const b = ndarray.array([3, -4]);
        const c = concat([a, b]);
        expect(c.toArray()).toEqual([-1, -2, 3, -4]);
    });

    test('14. Should concatenate arrays with floating point numbers', () => {
        const a = ndarray.array([1.1, 2.2]);
        const b = ndarray.array([3.3, 4.4]);
        const c = concat([a, b]);
        expect(c.toArray()).toEqual([1.1, 2.2, 3.3, 4.4]);
    });

    test('15. Should return a copy when concatenating a single array', () => {
        const a = ndarray.array([1, 2, 3]);
        const c = concat([a]);
        expect(c.toArray()).toEqual(a.toArray());
        expect(c).not.toBe(a); // Should be a new instance
        expect(c.data).not.toBe(a.data); // Data buffer should be different
    });
    
    // Category 3: Data Types
    test('16. Should concatenate int32 arrays', () => {
        const a = ndarray.array([1, 2], 'int32');
        const b = ndarray.array([3, 4], 'int32');
        const c = concat([a, b]);
        expect(c.dtype).toBe('int32');
        expect(c.data).toBeInstanceOf(Int32Array);
        expect(c.toArray()).toEqual([1, 2, 3, 4]);
    });

    test('17. Should concatenate uint8 arrays', () => {
        const a = ndarray.array([10, 20], 'uint8');
        const b = ndarray.array([30, 40], 'uint8');
        const c = concat([a, b]);
        expect(c.dtype).toBe('uint8');
        expect(c.data).toBeInstanceOf(Uint8Array);
        expect(c.toArray()).toEqual([10, 20, 30, 40]);
    });

    test('18. Should concatenate float32 arrays', () => {
        const a = ndarray.array([1.5, 2.5], 'float32');
        const b = ndarray.array([3.5, 4.5], 'float32');
        const c = concat([a, b]);
        expect(c.dtype).toBe('float32');
        expect(c.data).toBeInstanceOf(Float32Array);
        expect(c.get(0)).toBeCloseTo(1.5);
        expect(c.get(3)).toBeCloseTo(4.5);
    });
    
    // Category 4: Axis Variations
    test('19. Should work with negative axis (-1) on 2D array', () => {
        const a = ndarray.array([[1], [2]]);
        const b = ndarray.array([[3], [4]]);
        const c = concat([a, b], -1); // same as axis=1
        expect(c.shape).toEqual(new Int32Array([2, 2]));
        expect(c.toArray()).toEqual([[1, 3], [2, 4]]);
    });

    test('20. Should work with negative axis (-2) on 2D array', () => {
        const a = ndarray.array([[1, 2]]);
        const b = ndarray.array([[3, 4]]);
        const c = concat([a, b], -2); // same as axis=0
        expect(c.shape).toEqual(new Int32Array([2, 2]));
        expect(c.toArray()).toEqual([[1, 2], [3, 4]]);
    });

    test('21. Should work with negative axis (-1) on 3D array', () => {
        const a = ndarray.ones([2, 2, 1]);
        const b = ndarray.zeros([2, 2, 2]);
        const c = concat([a, b], -1); // same as axis=2
        expect(c.shape).toEqual(new Int32Array([2, 2, 3]));
        expect(c.toArray()).toEqual([[[1, 0, 0], [1, 0, 0]], [[1, 0, 0], [1, 0, 0]]]);
    });
    
    // Category 5: Non-Contiguous Arrays
    test('22. Should concatenate transposed 2D arrays on axis=0', () => {
        const a = ndarray.array([[1, 3], [2, 4]]).transpose(); // shape [2,2], strides [1,2]
        const b = ndarray.array([[5, 7], [6, 8]]).transpose();
        const c = concat([a, b], 0);
        expect(a.isContiguous).toBe(false);
        expect(c.shape).toEqual(new Int32Array([4, 2]));
        expect(c.toArray()).toEqual([[1, 2], [3, 4], [5, 6], [7, 8]]);
    });

    test('23. Should concatenate transposed 2D arrays on axis=1', () => {
        const a = ndarray.array([[1, 2], [5, 6]]);
        const b = ndarray.array([[3, 4], [7, 8]]).transpose(); // shape [2,2], but view of [[3,7],[4,8]]
        const c = concat([a, b], 1);
        expect(b.isContiguous).toBe(false);
        expect(c.shape).toEqual(new Int32Array([2, 4]));
        expect(c.toArray()).toEqual([[1, 2, 3, 7], [5, 6, 4, 8]]);
    });

    test('24. Should concatenate slices on axis=0', () => {
        const base = ndarray.arange(12).reshape(4, 3); // [[0,1,2],[3,4,5],[6,7,8],[9,10,11]]
        const a = base.slice([0, 2]); // [[0,1,2],[3,4,5]]
        const b = base.slice([3, 4]); // [[9,10,11]]
        const c = concat([a, b], 0);
        expect(c.shape).toEqual(new Int32Array([3, 3]));
        expect(c.toArray()).toEqual([[0, 1, 2], [3, 4, 5], [9, 10, 11]]);
    });
    
    test('25. Should concatenate slices on axis=1', () => {
        const base = ndarray.arange(12).reshape(3, 4); // [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
        const a = base.slice(null, [0, 2]); // [[0,1],[4,5],[8,9]]
        const b = base.slice(null, [3, 4]); // [[3],[7],[11]]
        const c = concat([a, b], 1);
        expect(c.shape).toEqual(new Int32Array([3, 3]));
        expect(c.toArray()).toEqual([[0, 1, 3], [4, 5, 7], [8, 9, 11]]);
    });
    
    test('26. Should concatenate a contiguous and a non-contiguous array', () => {
        const a = ndarray.array([[1, 2], [3, 4]]); // contiguous, shape [2, 2]
        const b = ndarray.arange(6).reshape(2, 3).transpose(); // shape [3, 2], not contiguous
        const c = concat([a, b], 0);
        expect(c.shape).toEqual(new Int32Array([5, 2]));
        expect(c.toArray()).toEqual([[1, 2], [3, 4], [0, 3], [1, 4], [2, 5]]);
    });

    // Category 6: Error Handling
    test('27. Should throw error for shape mismatch on non-concat axis', () => {
        const a = ndarray.array([[1, 2]]);
        const b = ndarray.array([[3, 4, 5]]);
        expect(() => concat([a, b], 0)).toThrow('Dimension mismatch on axis 1');
    });

    test('28. Should throw error for dimension (ndim) mismatch', () => {
        const a = ndarray.array([1, 2]);
        const b = ndarray.array([[3, 4]]);
        expect(() => concat([a, b])).toThrow('All arrays must have same number of dimensions');
    });

    test('29. Should throw error for dtype mismatch', () => {
        const a = ndarray.array([1, 2], 'float64');
        const b = ndarray.array([3, 4], 'int32');
        expect(() => concat([a, b])).toThrow('All arrays must have same dtype');
    });

    test('30. Should throw error for invalid axis', () => {
        const a = ndarray.array([1, 2]);
        const b = ndarray.array([3, 4]);
        expect(() => concat([a, b], 1)).toThrow('Axis 1 is out of bounds for array of dimension 1');
        expect(() => concat([a, b], -2)).toThrow('Axis -2 is out of bounds for array of dimension 1');
    });
});


describe('concat (Comprehensive Tests)', () => {
    // Category 1: Basic Functionality
    test('1. Should concatenate two 1D arrays', () => {
        const a = ndarray.array([1, 2]);
        const b = ndarray.array([3, 4]);
        const c = concat([a, b]);
        expect(c.toArray()).toEqual([1, 2, 3, 4]);
        expect(c.shape).toEqual(new Int32Array([4]));
    });

    test('2. Should concatenate three 1D arrays', () => {
        const a = ndarray.array([1]);
        const b = ndarray.array([2, 3]);
        const c = ndarray.array([4, 5, 6]);
        const d = concat([a, b, c]);
        expect(d.toArray()).toEqual([1, 2, 3, 4, 5, 6]);
    });

    test('3. Should concatenate two 2D arrays on axis=0', () => {
        const a = ndarray.array([[1, 2], [3, 4]]);
        const b = ndarray.array([[5, 6]]);
        const c = concat([a, b], 0);
        expect(c.toArray()).toEqual([[1, 2], [3, 4], [5, 6]]);
        expect(c.shape).toEqual(new Int32Array([3, 2]));
    });

    test('4. Should concatenate two 2D arrays on axis=1', () => {
        const a = ndarray.array([[1, 2], [3, 4]]);
        const b = ndarray.array([[5], [6]]);
        const c = concat([a, b], 1);
        expect(c.toArray()).toEqual([[1, 2, 5], [3, 4, 6]]);
        expect(c.shape).toEqual(new Int32Array([2, 3]));
    });

    test('5. Should concatenate three 2D arrays on axis=0', () => {
        const a = ndarray.array([[1, 2]]);
        const b = ndarray.array([[3, 4], [5, 6]]);
        const c = ndarray.array([[7, 8]]);
        const d = concat([a, b, c], 0);
        expect(d.toArray()).toEqual([[1, 2], [3, 4], [5, 6], [7, 8]]);
    });

    test('6. Should concatenate three 2D arrays on axis=1', () => {
        const a = ndarray.array([[1], [2]]);
        const b = ndarray.array([[3, 4], [5, 6]]);
        const c = ndarray.array([[7], [8]]);
        const d = concat([a, b, c], 1);
        expect(d.toArray()).toEqual([[1, 3, 4, 7], [2, 5, 6, 8]]);
    });

    test('7. Should concatenate two 3D arrays on axis=0', () => {
        const a = ndarray.ones([1, 2, 2]);
        const b = ndarray.zeros([2, 2, 2]);
        const c = concat([a, b], 0);
        expect(c.shape).toEqual(new Int32Array([3, 2, 2]));
        expect(c.toArray()).toEqual([[[1, 1], [1, 1]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]);
    });

    test('8. Should concatenate two 3D arrays on axis=1', () => {
        const a = ndarray.ones([2, 1, 2]);
        const b = ndarray.zeros([2, 2, 2]);
        const c = concat([a, b], 1);
        expect(c.shape).toEqual(new Int32Array([2, 3, 2]));
        expect(c.toArray()).toEqual([[[1, 1], [0, 0], [0, 0]], [[1, 1], [0, 0], [0, 0]]]);
    });

    test('9. Should concatenate two 3D arrays on axis=2', () => {
        const a = ndarray.ones([2, 2, 1]);
        const b = ndarray.zeros([2, 2, 2]);
        const c = concat([a, b], 2);
        expect(c.shape).toEqual(new Int32Array([2, 2, 3]));
        expect(c.toArray()).toEqual([[[1, 0, 0], [1, 0, 0]], [[1, 0, 0], [1, 0, 0]]]);
    });

    // Category 2: Edge Cases
    test('10. Should concatenate an array with an empty array', () => {
        const a = ndarray.array([[1, 2]]);
        const b = ndarray.zeros([0, 2]);
        const c = concat([a, b], 0);
        expect(c.toArray()).toEqual([[1, 2]]);
    });

    test('11. Should handle concatenating only empty arrays', () => {
        const a = ndarray.zeros([0, 2, 3]);
        const b = ndarray.zeros([0, 2, 3]);
        const c = concat([a, b], 0);
        expect(c.shape).toEqual(new Int32Array([0, 2, 3]));
        expect(c.size).toBe(0);
    });

    test('12. Should concatenate arrays with zeros', () => {
        const a = ndarray.array([0, 1]);
        const b = ndarray.array([0, 2]);
        const c = concat([a, b]);
        expect(c.toArray()).toEqual([0, 1, 0, 2]);
    });

    test('13. Should concatenate arrays with negative numbers', () => {
        const a = ndarray.array([-1, -2]);
        const b = ndarray.array([3, -4]);
        const c = concat([a, b]);
        expect(c.toArray()).toEqual([-1, -2, 3, -4]);
    });

    test('14. Should concatenate arrays with floating point numbers', () => {
        const a = ndarray.array([1.1, 2.2]);
        const b = ndarray.array([3.3, 4.4]);
        const c = concat([a, b]);
        expect(c.toArray()).toEqual([1.1, 2.2, 3.3, 4.4]);
    });

    test('15. Should return a copy when concatenating a single array', () => {
        const a = ndarray.array([1, 2, 3]);
        const c = concat([a]);
        expect(c.toArray()).toEqual(a.toArray());
        expect(c).not.toBe(a); // Should be a new instance
        expect(c.data).not.toBe(a.data); // Data buffer should be different
    });
    
    // Category 3: Data Types
    test('16. Should concatenate int32 arrays', () => {
        const a = ndarray.array([1, 2], 'int32');
        const b = ndarray.array([3, 4], 'int32');
        const c = concat([a, b]);
        expect(c.dtype).toBe('int32');
        expect(c.data).toBeInstanceOf(Int32Array);
        expect(c.toArray()).toEqual([1, 2, 3, 4]);
    });

    test('17. Should concatenate uint8 arrays', () => {
        const a = ndarray.array([10, 20], 'uint8');
        const b = ndarray.array([30, 40], 'uint8');
        const c = concat([a, b]);
        expect(c.dtype).toBe('uint8');
        expect(c.data).toBeInstanceOf(Uint8Array);
        expect(c.toArray()).toEqual([10, 20, 30, 40]);
    });

    test('18. Should concatenate float32 arrays', () => {
        const a = ndarray.array([1.5, 2.5], 'float32');
        const b = ndarray.array([3.5, 4.5], 'float32');
        const c = concat([a, b]);
        expect(c.dtype).toBe('float32');
        expect(c.data).toBeInstanceOf(Float32Array);
        expect(c.get(0)).toBeCloseTo(1.5);
        expect(c.get(3)).toBeCloseTo(4.5);
    });
    
    // Category 4: Axis Variations
    test('19. Should work with negative axis (-1) on 2D array', () => {
        const a = ndarray.array([[1], [2]]);
        const b = ndarray.array([[3], [4]]);
        const c = concat([a, b], -1); // same as axis=1
        expect(c.shape).toEqual(new Int32Array([2, 2]));
        expect(c.toArray()).toEqual([[1, 3], [2, 4]]);
    });

    test('20. Should work with negative axis (-2) on 2D array', () => {
        const a = ndarray.array([[1, 2]]);
        const b = ndarray.array([[3, 4]]);
        const c = concat([a, b], -2); // same as axis=0
        expect(c.shape).toEqual(new Int32Array([2, 2]));
        expect(c.toArray()).toEqual([[1, 2], [3, 4]]);
    });

    test('21. Should work with negative axis (-1) on 3D array', () => {
        const a = ndarray.ones([2, 2, 1]);
        const b = ndarray.zeros([2, 2, 2]);
        const c = concat([a, b], -1); // same as axis=2
        expect(c.shape).toEqual(new Int32Array([2, 2, 3]));
        expect(c.toArray()).toEqual([[[1, 0, 0], [1, 0, 0]], [[1, 0, 0], [1, 0, 0]]]);
    });
    
    // Category 5: Non-Contiguous Arrays
    test('22. Should concatenate transposed 2D arrays on axis=0', () => {
        const a = ndarray.array([[1, 3], [2, 4]]).transpose(); // shape [2,2], strides [1,2]
        const b = ndarray.array([[5, 7], [6, 8]]).transpose();
        const c = concat([a, b], 0);
        expect(a.isContiguous).toBe(false);
        expect(c.shape).toEqual(new Int32Array([4, 2]));
        expect(c.toArray()).toEqual([[1, 2], [3, 4], [5, 6], [7, 8]]);
    });

    test('23. Should concatenate transposed 2D arrays on axis=1', () => {
        const a = ndarray.array([[1, 2], [5, 6]]);
        const b = ndarray.array([[3, 4], [7, 8]]).transpose(); // shape [2,2], but view of [[3,7],[4,8]]
        const c = concat([a, b], 1);
        expect(b.isContiguous).toBe(false);
        expect(c.shape).toEqual(new Int32Array([2, 4]));
        expect(c.toArray()).toEqual([[1, 2, 3, 7], [5, 6, 4, 8]]);
    });

    test('24. Should concatenate slices on axis=0', () => {
        const base = ndarray.arange(12).reshape(4, 3); // [[0,1,2],[3,4,5],[6,7,8],[9,10,11]]
        const a = base.slice([0, 2]); // [[0,1,2],[3,4,5]]
        const b = base.slice([3, 4]); // [[9,10,11]]
        const c = concat([a, b], 0);
        expect(c.shape).toEqual(new Int32Array([3, 3]));
        expect(c.toArray()).toEqual([[0, 1, 2], [3, 4, 5], [9, 10, 11]]);
    });
    
    test('25. Should concatenate slices on axis=1', () => {
        const base = ndarray.arange(12).reshape(3, 4); // [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
        const a = base.slice(null, [0, 2]); // [[0,1],[4,5],[8,9]]
        const b = base.slice(null, [3, 4]); // [[3],[7],[11]]
        const c = concat([a, b], 1);
        expect(c.shape).toEqual(new Int32Array([3, 3]));
        expect(c.toArray()).toEqual([[0, 1, 3], [4, 5, 7], [8, 9, 11]]);
    });
    
    test('26. Should concatenate a contiguous and a non-contiguous array', () => {
        const a = ndarray.array([[1, 2], [3, 4]]); // contiguous, shape [2, 2]
        const b = ndarray.arange(6).reshape(2, 3).transpose(); // shape [3, 2], not contiguous
        const c = concat([a, b], 0);
        expect(c.shape).toEqual(new Int32Array([5, 2]));
        expect(c.toArray()).toEqual([[1, 2], [3, 4], [0, 3], [1, 4], [2, 5]]);
    });

    // Category 6: Error Handling
    test('27. Should throw error for shape mismatch on non-concat axis', () => {
        const a = ndarray.array([[1, 2]]);
        const b = ndarray.array([[3, 4, 5]]);
        expect(() => concat([a, b], 0)).toThrow('Dimension mismatch on axis 1');
    });

    test('28. Should throw error for dimension (ndim) mismatch', () => {
        const a = ndarray.array([1, 2]);
        const b = ndarray.array([[3, 4]]);
        expect(() => concat([a, b])).toThrow('All arrays must have same number of dimensions');
    });

    test('29. Should throw error for dtype mismatch', () => {
        const a = ndarray.array([1, 2], 'float64');
        const b = ndarray.array([3, 4], 'int32');
        expect(() => concat([a, b])).toThrow('All arrays must have same dtype');
    });

    test('30. Should throw error for invalid axis', () => {
        const a = ndarray.array([1, 2]);
        const b = ndarray.array([3, 4]);
        expect(() => concat([a, b], 1)).toThrow('Axis 1 is out of bounds for array of dimension 1');
        expect(() => concat([a, b], -2)).toThrow('Axis -2 is out of bounds for array of dimension 1');
    });
});