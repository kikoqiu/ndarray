const ndarray = require('../dist/ndarray.cjs');

describe('NDArray Operations', () => {

    test('add (scalar)', () => {
        const a = ndarray.arange(4).reshape([2, 2]); // [[0, 1], [2, 3]]
        const b = a.add(10);
        expect(b.data).toEqual(new Float64Array([10, 11, 12, 13]));
    });

    test('add (array)', () => {
        const a = ndarray.array([[1, 2], [3, 4]]);
        const b = ndarray.array([[10, 20], [30, 40]]);
        const c = a.add(b);
        expect(c.data).toEqual(new Float64Array([11, 22, 33, 44]));
    });

    test('add (array) slice', () => {
        const a = ndarray.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        const b = ndarray.array([[10, 20],[30, 40]]);
        const c=a.slice([0,2],[1,3]);
        const d = b.add(c);
        expect(d.data).toEqual(new Float64Array([12, 23, 35, 46]));
    });

    test('add (broadcast)', () => {
        const a = ndarray.array([[1, 2, 3], [4, 5, 6]]); // shape [2, 3]
        const b = ndarray.array([10, 20, 30]); // shape [3]
        const c = a.add(b);
        expect(c.shape).toEqual(new Int32Array([2, 3]));
        // Use copy() to get a new flat array for simple comparison
        expect(c.copy().data).toEqual(new Float64Array([11, 22, 33, 14, 25, 36]));
    });

    test('iadd (in-place add)', () => {
        const a = ndarray.ones([2, 2]);
        a.iadd(5);
        expect(a.data).toEqual(new Float64Array([6, 6, 6, 6]));
    });

    test('mul (element-wise multiplication)', () => {
        const a = ndarray.array([[1, 2], [3, 4]]);
        const b = a.mul(a);
        expect(b.data).toEqual(new Float64Array([1, 4, 9, 16]));
    });
    
    test('sum (global)', () => {
        const a = ndarray.arange(6); // [0, 1, 2, 3, 4, 5]
        const sum = a.sum();
        expect(sum).toBe(15);
    });

    test('sum (axis)', () => {
        const a = ndarray.arange(6).reshape([2, 3]); // [[0, 1, 2], [3, 4, 5]]
        
        const sum0 = a.sum(0); // sum along columns
        expect(sum0.shape).toEqual(new Int32Array([3]));
        expect(sum0.copy().data).toEqual(new Float64Array([3, 5, 7]));
        
        const sum1 = a.sum(1); // sum along rows
        expect(sum1.shape).toEqual(new Int32Array([2]));
        expect(sum1.copy().data).toEqual(new Float64Array([3, 12]));
    });

    test('mean', () => {
        const a = ndarray.arange(4).reshape([2, 2]); // [[0, 1], [2, 3]]
        const mean = a.mean();
        expect(mean).toBe(1.5);

        const mean1 = a.mean(1); // mean along rows
        expect(mean1.copy().data).toEqual(new Float64Array([0.5, 2.5]));
    });

});

describe('Broadcasting Edge Cases', () => {
    test('1D to 3D', () => {
        const a = ndarray.zeros([2, 3, 4]);
        const b = ndarray.arange(4); // shape [4]
        const c = a.add(b);

        expect(c.shape).toEqual(new Int32Array([2, 3, 4]));
        // Every 4-element row should be [0, 1, 2, 3]
        const expectedRow = new Float64Array([0, 1, 2, 3]);
        const cContiguous = c.copy().data;
        for (let i = 0; i < 6; i++) {
            const row = cContiguous.subarray(i * 4, i * 4 + 4);
            expect(row).toEqual(expectedRow);
        }
    });

    test('Broadcast with inner dimension of 1', () => {
        const a = ndarray.arange(4).reshape([2, 1, 2]); // [[[0,1]], [[2,3]]]
        const b = ndarray.array([[10, 20], [30, 40]]); // shape [2, 2]
        
        // broadcast `a` from [2, 1, 2] to [2, 2, 2]
        // broadcast `b` from [   2, 2] to [2, 2, 2]
        const c = a.add(b);

        const expected = ndarray.array([
            [[10, 21], [30, 41]], // from [[0,1]] + [[10,20],[30,40]]
            [[12, 23], [32, 43]]  // from [[2,3]] + [[10,20],[30,40]]
        ]);

        expect(c.shape).toEqual(new Int32Array([2, 2, 2]));
        expect(c.copy().data).toEqual(expected.copy().data);
    });

    test('Broadcast should fail on incompatible shapes', () => {
        const a = ndarray.zeros([2, 3]);
        const b = ndarray.zeros([4, 5]);
        const c = ndarray.zeros([3, 2]);

        expect(() => a.add(b)).toThrow("Incompatible shapes");
        expect(() => a.add(c)).toThrow("Incompatible shapes");
    });
});

describe('Reduction Edge Cases', () => {
    test('Sum on a non-contiguous (transposed) array', () => {
        const a = ndarray.array([[1, 2, 3], [4, 5, 6]]).transpose(); // [[1,4], [2,5], [3,6]], shape [3,2]
        expect(a.isContiguous).toBe(false);

        // Global sum should still work
        expect(a.sum()).toBe(21);

        // Axis sum
        const sum0 = a.sum(0); // sum along columns: [1+2+3, 4+5+6] = [6, 15]
        expect(sum0.copy().data).toEqual(new Float64Array([6, 15]));

        const sum1 = a.sum(1); // sum along rows: [1+4, 2+5, 3+6] = [5, 7, 9]
        expect(sum1.copy().data).toEqual(new Float64Array([5, 7, 9]));
    });

    test('Mean on an array with a dimension of size 1', () => {
        const a = ndarray.array([[10], [20], [30]]); // shape [3, 1]
        
        const mean0 = a.mean(0); // reduce along axis 0
        expect(mean0.shape).toEqual(new Int32Array([1]));
        expect(mean0.get(0)).toBe(20);

        const mean1 = a.mean(1); // reduce along axis 1
        expect(mean1.shape).toEqual(new Int32Array([3]));
        expect(mean1.copy().data).toEqual(new Float64Array([10, 20, 30]));
    });
});
