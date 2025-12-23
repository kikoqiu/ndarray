const ndarray = require('../dist/ndarray.cjs');

describe('Bitwise Functions (Fixed Types)', () => {
    
    describe('bitwise_and', () => {
        test('uint8 array and scalar', () => {
            const a = ndarray.array([255, 170, 85], 'uint8'); // [11111111, 10101010, 01010101]
            const result = a.bitwise_and(15);                 // 00001111
            expect(result.copy().data).toEqual(new Uint8Array([15, 10, 5]));
            expect(result.dtype).toBe('uint8');
        });

        test('uint32 array and array', () => {
            const a = ndarray.array([0xFFFFFFFF, 0x0000FFFF], 'uint32');
            const b = ndarray.array([0xFFFF0000, 0x0000FFFF], 'uint32');
            const result = a.bitwise_and(b);
            expect(result.copy().data).toEqual(new Uint32Array([0xFFFF0000, 0x0000FFFF]));
            expect(result.dtype).toBe('uint32');
        });

        test('uint8 broadcast', () => {
            const a = ndarray.array([[240, 15], [240, 15]], 'uint8');
            const b = ndarray.array([240, 240], 'uint8');
            const result = a.bitwise_and(b);
            expect(result.copy().data).toEqual(new Uint8Array([240, 0, 240, 0]));
        });

        test('uint32 large values', () => {
            const a = ndarray.array([0x12345678], 'uint32');
            const result = a.bitwise_and(0xFF00FF00);
            expect(result.copy().data[0]).toBe(0x12005600);
        });

        test('uint8 slice operation', () => {
            const a = ndarray.array([255, 255, 255, 255], 'uint8').reshape([2, 2]);
            const b = ndarray.array([1, 2], 'uint8');
            const result = a.slice([0, 1], [0, 2]).bitwise_and(b);
            expect(result.copy().data).toEqual(new Uint8Array([1, 2]));
        });
    });

    describe('bitwise_or', () => {
        test('uint8 scalar', () => {
            const a = ndarray.array([128, 64, 32], 'uint8');
            const result = a.bitwise_or(1);
            expect(result.copy().data).toEqual(new Uint8Array([129, 65, 33]));
            expect(result.dtype).toBe('uint8');
        });

        test('uint32 array', () => {
            const a = ndarray.array([0xAAAA0000], 'uint32');
            const b = ndarray.array([0x00005555], 'uint32');
            const result = a.bitwise_or(b);
            expect(result.copy().data[0]).toBe(0xAAAA5555);
        });

        test('uint32 with sign bit (0x80000000)', () => {
            const a = ndarray.array([0x7FFFFFFF], 'uint32');
            const result = a.bitwise_or(0x80000000);
            expect(result.copy().data[0]).toBe(0xFFFFFFFF);
        });

        test('uint8 broadcast row', () => {
            const a = ndarray.array([[0], [1], [2]], 'uint8');
            const b = ndarray.array([128], 'uint8');
            const result = a.bitwise_or(b);
            expect(result.copy().data).toEqual(new Uint8Array([128, 129, 130]));
        });

        test('uint32 identity', () => {
            const a = ndarray.array([123456], 'uint32');
            const result = a.bitwise_or(0);
            expect(result.copy().data[0]).toBe(123456);
        });
    });

    describe('bitwise_xor', () => {
        test('uint8 toggle', () => {
            const a = ndarray.array([255, 0, 170], 'uint8');
            const result = a.bitwise_xor(255);
            expect(result.copy().data).toEqual(new Uint8Array([0, 255, 85]));
        });

        test('uint32 self xor', () => {
            const a = ndarray.array([0xDEADBEEF, 0xCAFEBABE], 'uint32');
            const result = a.bitwise_xor(a);
            expect(result.copy().data).toEqual(new Uint32Array([0, 0]));
        });

        test('uint8 symmetry', () => {
            const a = ndarray.array([1, 2, 3], 'uint8');
            const b = ndarray.array([4, 5, 6], 'uint8');
            const res1 = a.bitwise_xor(b);
            const res2 = b.bitwise_xor(a);
            expect(res1.copy().data).toEqual(res2.copy().data);
        });

        test('uint32 large broadcast', () => {
            const a = ndarray.array([[0x1], [0x2]], 'uint32');
            const b = ndarray.array([0xFFFFFFFF], 'uint32');
            const result = a.bitwise_xor(b);
            // 0xFFFFFFFF ^ 1 = 0xFFFFFFFE
            expect(result.copy().data).toEqual(new Uint32Array([0xFFFFFFFE, 0xFFFFFFFD]));
        });

        test('uint8 dtype persistence', () => {
            const a = ndarray.array([1], 'uint8');
            expect(a.bitwise_xor(1).dtype).toBe('uint8');
        });
    });

    describe('bitwise_lshift', () => {
        test('uint8 shift', () => {
            const a = ndarray.array([1, 2, 4, 8], 'uint8');
            const result = a.bitwise_lshift(1);
            expect(result.copy().data).toEqual(new Uint8Array([2, 4, 8, 16]));
        });

        test('uint8 overflow (wrap)', () => {
            const a = ndarray.array([128, 192], 'uint8'); // [10000000, 11000000]
            const result = a.bitwise_lshift(1);          // 128 << 1 = 256 -> 0
            expect(result.copy().data).toEqual(new Uint8Array([0, 128]));
        });

        test('uint32 large shift', () => {
            const a = ndarray.array([0x0000FFFF], 'uint32');
            const result = a.bitwise_lshift(16);
            expect(result.copy().data[0]).toBe(0xFFFF0000);
        });

        test('uint32 shift by array', () => {
            const a = ndarray.array([1, 1, 1], 'uint32');
            const b = ndarray.array([10, 20, 30], 'uint32');
            const result = a.bitwise_lshift(b);
            expect(result.copy().data).toEqual(new Uint32Array([1024, 1048576, 1073741824]));
        });

        test('uint8 zero shift', () => {
            const a = ndarray.array([255], 'uint8');
            expect(a.bitwise_lshift(0).copy().data[0]).toBe(255);
        });
    });

    describe('bitwise_rshift', () => {
        test('uint8 positive shift', () => {
            const a = ndarray.array([128, 64, 32], 'uint8');
            const result = a.bitwise_rshift(1);
            expect(result.copy().data).toEqual(new Uint8Array([64, 32, 16]));
        });

        test('uint32 shift', () => {
            const a = ndarray.array([0x80000000], 'uint32');
            const result = a.bitwise_rshift(1);
            expect(result.copy().data[0]).toBe(0x40000000); 
        });

        test('uint32 broadcast shift', () => {
            const a = ndarray.array([1024, 2048], 'uint32');
            const b = ndarray.array([2], 'uint32');
            const result = a.bitwise_rshift(b);
            expect(result.copy().data).toEqual(new Uint32Array([256, 512]));
        });

        test('uint8 small values', () => {
            const a = ndarray.array([1, 0], 'uint8');
            const result = a.bitwise_rshift(1);
            expect(result.copy().data).toEqual(new Uint8Array([0, 0]));
        });

        test('uint32 max shift', () => {
            const a = ndarray.array([0xFFFFFFFF], 'uint32');
            const result = a.bitwise_rshift(31);
            expect(result.copy().data[0]).toBe(1);
        });
    });

    describe('bitwise_not', () => {
        test('uint8 inversion', () => {
            const a = ndarray.array([0, 255, 170], 'uint8');
            const result = a.bitwise_not();
            // ~0 = -1 (Uint8: 255)
            // ~255 = -256 (Uint8: 0)
            // ~170 (10101010) = -171 (Uint8: 01010101 = 85)
            expect(result.copy().data).toEqual(new Uint8Array([255, 0, 85]));
            expect(result.dtype).toBe('uint8');
        });

        test('uint32 full inversion', () => {
            const a = ndarray.array([0x00000000, 0xFFFFFFFF], 'uint32');
            const result = a.bitwise_not();
            expect(result.copy().data).toEqual(new Uint32Array([0xFFFFFFFF, 0x00000000]));
            expect(result.dtype).toBe('uint32');
        });

        test('uint32 pattern inversion', () => {
            const a = ndarray.array([0xAAAAAAAA], 'uint32');
            const result = a.bitwise_not();
            expect(result.copy().data[0]).toBe(0x55555555);
        });

        test('uint8 double negation', () => {
            const a = ndarray.array([1, 2, 3], 'uint8');
            const result = a.bitwise_not().bitwise_not();
            expect(result.copy().data).toEqual(new Uint8Array([1, 2, 3]));
        });

        test('uint32 slice not', () => {
            const a = ndarray.array([0, 0, 0, 0], 'uint32').reshape([2, 2]);
            const result = a.slice([0, 1], [0, 2]).bitwise_not();
            expect(result.copy().data).toEqual(new Uint32Array([0xFFFFFFFF, 0xFFFFFFFF]));
        });
    });
});