const ndarray = require('../dist/ndarray.cjs');


describe('Binary Operations', () => {

    let a, b;
    beforeEach(() => {
        // Re-initialize arrays before each test to ensure independence
        a = ndarray.array([[1, 2], [3, 4]]);
        b = ndarray.array([[10, 20], [30, 40]]);
    });

    // Helper for validation of 2D array results
    const validate2DResult = (result, expectedShape, validationFn) => {
        expect(result.shape).toEqual(new Int32Array(expectedShape));
        const [rows, cols] = expectedShape;
        const resContiguous = result.copy(); // Work on a contiguous copy for easy iteration
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const value = resContiguous.get(i, j);
                const expected = validationFn(i, j);
                expect(value).toBeCloseTo(expected);
            }
        }
    };
    
    const ops = [
        { name: 'add', op: (x, y) => x + y },
        { name: 'sub', op: (x, y) => x - y },
        { name: 'mul', op: (x, y) => x * y },
        { name: 'div', op: (x, y) => x / y },
        { name: 'pow', op: (x, y) => x ** y },
        { name: 'mod', op: (x, y) => x % y },
    ];

    ops.forEach(({ name, op }) => {
        const iop = `i${name}`; // e.g., 'iadd'

        describe(`${name}()`, () => {
            test(`scalar ${name}`, () => {
                const scalar = 10;
                const result = a[name](scalar);
                validate2DResult(result, [2, 2], (i, j) => op(a.get(i, j), scalar));
            });

            test(`element-wise ${name}`, () => {
                const result = a[name](b);
                validate2DResult(result, [2, 2], (i, j) => op(a.get(i, j), b.get(i, j)));
            });

            test(`broadcast ${name}`, () => {
                const v = ndarray.array([100, 200]); // vector with shape [2]
                const result = a[name](v); // [[1,2],[3,4]] op [100, 200]
                validate2DResult(result, [2, 2], (i, j) => op(a.get(i, j), v.get(j)));
            });

            test(`in-place scalar ${name} (${iop})`, () => {
                const a_copy = a.copy();
                const scalar = 5;
                a_copy[iop](scalar); // Modify in-place
                validate2DResult(a_copy, [2, 2], (i, j) => op(a.get(i, j), scalar));
            });
    
            test(`in-place element-wise ${name} (${iop})`, () => {
                const a_copy = a.copy();
                a_copy[iop](b); // Modify in-place
                validate2DResult(a_copy, [2, 2], (i, j) => op(a.get(i, j), b.get(i, j)));
            });
        });
    });

    // Special case for broadcasting a column vector
    test('broadcast column vector', () => {
        const col = ndarray.array([[10], [20]]); // shape [2, 1]
        const result = a.add(col); // [[1,2],[3,4]] + [[10],[20]]
        
        const expected = ndarray.array([[11, 12], [23, 24]]);
        expect(result.copy().data).toEqual(expected.data);
    });
});
