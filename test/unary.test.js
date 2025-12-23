const ndarray = require('../dist/ndarray.cjs');
const { NDArray } = ndarray;

/**
 * Helper function to validate a unary operation against an expected array of values.
 * @param {NDArray} result - The result from the NDArray operation.
 * @param {Array<number>} expectedData - An array of the expected numerical results.
 */
const validateElementwise = (result, expectedData) => {
    const resData = result.copy().data;
    expect(resData.length).toBe(expectedData.length);
    resData.forEach((val, i) => {
        // Use toBeCloseTo for floating point comparisons
        expect(val).toBeCloseTo(expectedData[i]);
    });
};

describe('Unary Operations', () => {
    // A standard array with mixed float/integer, positive/negative values
    const baseData = [-3.8, -1, 0, 1.2, 5];
    const baseArr = ndarray.array(baseData);
    // A non-contiguous view of the same data
    const baseView = ndarray.array([baseData, baseData]).transpose().slice(null, 0);

    // --- Arithmetic Unary Ops ---

    describe('neg()', () => {
        const expected = baseData.map(v => -v);
        test('calculates the negative value of each element', () => {
            validateElementwise(baseArr.neg(), expected);
        });
        test('works correctly on a non-contiguous view', () => {
            validateElementwise(baseView.neg(), expected);
        });
    });

    describe('abs()', () => {
        const expected = baseData.map(v => Math.abs(v));
        test('calculates the absolute value of each element', () => {
            validateElementwise(baseArr.abs(), expected);
        });
        test('works correctly on a non-contiguous view', () => {
            validateElementwise(baseView.abs(), expected);
        });
    });

    // --- Exponential and Logarithmic Ops ---

    describe('exp()', () => {
        const expected = baseData.map(v => Math.exp(v));
        test('calculates the exponential of each element', () => {
            validateElementwise(baseArr.exp(), expected);
        });
        test('works correctly on a non-contiguous view', () => {
            validateElementwise(baseView.exp(), expected);
        });
    });

    describe('log()', () => {
        const logData = [0.1, 1, Math.E, 10, 100];
        const logArr = ndarray.array(logData);
        const expected = logData.map(v => Math.log(v));
        test('calculates the natural log of each element', () => {
            validateElementwise(logArr.log(), expected);
        });
        test('results in NaN for non-positive inputs', () => {
            const negArr = ndarray.array([-1, 0]);
            const result = negArr.log().copy().data;
            expect(result[0]).toBeNaN(); // log(-1)
            expect(result[1]).toBe(-Infinity); // log(0)
        });
    });

    describe('sqrt()', () => {
        const sqrtData = [0, 1, 4, 9.5, 16];
        const sqrtArr = ndarray.array(sqrtData);
        const expected = sqrtData.map(v => Math.sqrt(v));
        test('calculates the square root of each element', () => {
            validateElementwise(sqrtArr.sqrt(), expected);
        });
        test('results in NaN for negative inputs', () => {
            const negArr = ndarray.array([-1, -4]);
            const result = negArr.sqrt().copy().data;
            expect(result[0]).toBeNaN();
            expect(result[1]).toBeNaN();
        });
    });
    
    // --- Rounding Ops ---

    describe('ceil()', () => {
        const expected = baseData.map(v => Math.ceil(v));
        test('calculates the ceiling of each element', () => {
            validateElementwise(baseArr.ceil(), expected);
        });
        test('works correctly on a non-contiguous view', () => {
            validateElementwise(baseView.ceil(), expected);
        });
    });

    describe('floor()', () => {
        const expected = baseData.map(v => Math.floor(v));
        test('calculates the floor of each element', () => {
            validateElementwise(baseArr.floor(), expected);
        });
        test('works correctly on a non-contiguous view', () => {
            validateElementwise(baseView.floor(), expected);
        });
    });

    describe('round()', () => {
        const expected = baseData.map(v => Math.round(v));
        test('rounds each element to the nearest integer', () => {
            validateElementwise(baseArr.round(), expected);
        });
        test('works correctly on a non-contiguous view', () => {
            validateElementwise(baseView.round(), expected);
        });
    });

    // --- Trigonometric Ops ---

    describe('sin()', () => {
        const trigData = [-Math.PI, -Math.PI / 2, 0, Math.PI / 2, Math.PI];
        const trigArr = ndarray.array(trigData);
        const expected = trigData.map(v => Math.sin(v));
        test('calculates the sine of each element', () => {
            validateElementwise(trigArr.sin(), expected);
        });
    });

    describe('cos()', () => {
        const trigData = [-Math.PI, -Math.PI / 2, 0, Math.PI / 2, Math.PI];
        const trigArr = ndarray.array(trigData);
        const expected = trigData.map(v => Math.cos(v));
        test('calculates the cosine of each element', () => {
            validateElementwise(trigArr.cos(), expected);
        });
    });

    describe('tan()', () => {
        const trigData = [-Math.PI / 4, 0, Math.PI / 4];
        const trigArr = ndarray.array(trigData);
        const expected = trigData.map(v => Math.tan(v));
        test('calculates the tangent of each element', () => {
            validateElementwise(trigArr.tan(), expected);
        });
    });

});
