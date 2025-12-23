const ndarray = require('../dist/ndarray.cjs');

describe('Logical and Comparison Operations', () => {
    const a = ndarray.array([[1, 5, 3], [4, 2, 6]]);
    const b = ndarray.array([[1, 2, 9], [0, 2, 8]]);

    describe('Comparison Operators', () => {
        const comp_ops = [
            { name: 'eq', op: (x, y) => x === y },
            { name: 'neq', op: (x, y) => x !== y },
            { name: 'gt', op: (x, y) => x > y },
            { name: 'gte', op: (x, y) => x >= y },
            { name: 'lt', op: (x, y) => x < y },
            { name: 'lte', op: (x, y) => x <= y },
        ];

        comp_ops.forEach(({ name, op }) => {
            describe(`${name}()`, () => {
                test(`scalar comparison`, () => {
                    const scalar = 3;
                    const result = a[name](scalar);
                    const expected = a.copy().data.map(v => op(v, scalar) ? 1 : 0);
                    
                    expect(result.dtype).toBe('uint8');
                    expect(result.copy().data).toEqual(new Uint8Array(expected));
                });

                test(`element-wise comparison`, () => {
                    const result = a[name](b);
                    const expected = a.copy().data.map((v, i) => op(v, b.copy().data[i]) ? 1 : 0);

                    expect(result.shape).toEqual(a.shape);
                    expect(result.dtype).toBe('uint8');
                    expect(result.copy().data).toEqual(new Uint8Array(expected));
                });

                test('broadcast comparison', () => {
                    const v = ndarray.array([4, 2, 7]);
                    const result = a[name](v);
                    const expected = a.copy().data.map((val, i) => op(val, v.data[i % 3]) ? 1 : 0);
                    expect(result.copy().data).toEqual(new Uint8Array(expected));
                });
            });
        });
    });

    describe('Logical Functions', () => {
        const bool_a = ndarray.array([[1, 0, 1], [0, 0, 1]], 'uint8');
        const bool_b = ndarray.array([[1, 1, 0], [0, 1, 0]], 'uint8');

        test('logical_and', () => {
            const result = bool_a.logical_and(bool_b);
            expect(result.copy().data).toEqual(new Uint8Array([1, 0, 0, 0, 0, 0]));
            expect(result.dtype).toBe('uint8');
        });

        test('logical_or', () => {
            const result = bool_a.logical_or(bool_b);
            expect(result.copy().data).toEqual(new Uint8Array([1, 1, 1, 0, 1, 1]));
            expect(result.dtype).toBe('uint8');
        });

        test('logical_not', () => {
            const result = bool_a.logical_not();
            expect(result.copy().data).toEqual(new Uint8Array([0, 1, 0, 1, 1, 0]));
            expect(result.dtype).toBe('uint8');
        });
    });

    describe('all() and any()', () => {
        test('all() is true when all elements are non-zero', () => {
            const arr = ndarray.array([1, 2, 3, -1]);
            expect(arr.all()).toBe(true);
        });

        test('all() is false when one element is zero', () => {
            const arr = ndarray.array([1, 2, 0, -1]);
            expect(arr.all()).toBe(false);
        });

        test('any() is true when at least one element is non-zero', () => {
            const arr = ndarray.array([0, 0, 5, 0]);
            expect(arr.any()).toBe(true);
        });

        test('any() is false when all elements are zero', () => {
            const arr = ndarray.zeros([2, 3]);
            expect(arr.any()).toBe(false);
        });
        
        test('any() works on a non-contiguous view', () => {
            const arr = ndarray.array([[0, 1], [0, 0]]).transpose(); // [[0, 0], [1, 0]]
            expect(arr.any()).toBe(true);
        });
    });
});
