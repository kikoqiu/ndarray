const ndarray = require('../dist/ndarray.cjs');

// These tests can be slow, so they are in a separate file.
describe('Large Data Validation', () => {

    const size = 1000 * 1000; // 1 million elements
    
    // Increase timeout for this suite to 30 seconds
    beforeAll(() => {
        jest.setTimeout(30000);
    });

    test('sum() on a large array', () => {
        const arr = ndarray.arange(size);
        // The sum of numbers from 0 to n-1 is n * (n-1) / 2
        const expectedSum = size * (size - 1) / 2;
        expect(arr.sum()).toBe(expectedSum);
    });

    test('mean() on a large array', () => {
        const arr = ndarray.arange(size);
        // The mean of numbers from 0 to n-1 is (n-1) / 2
        const expectedMean = (size - 1) / 2;
        expect(arr.mean()).toBeCloseTo(expectedMean);
    });

    test('element-wise add on large arrays', () => {
        const a = ndarray.arange(size);
        const b = ndarray.full([size], 5);
        const c = a.add(b);

        // Don't check every element. Check a few pseudo-randomly spaced ones.
        for (let i = 0; i < size; i += Math.floor(size / 10)) {
            const index = Math.floor(i);
            const expected = index + 5;
            expect(c.get(index)).toBe(expected);
        }
    });

    test('broadcast add on large arrays', () => {
        const rows = 1000;
        const cols = size / rows;
        const a = ndarray.arange(size).reshape([rows, cols]); 
        const b = ndarray.full([cols], 10); // 1D array to broadcast
        const c = a.add(b);

        // Check the first element of a few rows
        for (let i = 0; i < rows; i += Math.floor(rows / 10)) {
            const index = Math.floor(i);
            const expected = (index * cols) + 10; // (row * num_cols) + broadcast_val_at_col_0
            expect(c.get(index, 0)).toBe(expected);
        }
    });
});
