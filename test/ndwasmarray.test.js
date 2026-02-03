const ndarray = require('../dist/ndarray.cjs');
const { NDArray, NDWasm, NDWasmArray, WasmRuntime } = ndarray;


describe('NDWasmArray Comprehensive Test Suite', () => {
    
    beforeAll(async () => {
        const runtime = new WasmRuntime();
        await runtime.init({
            execUrl: 'dist/wasm_exec.js',
            wasmUrl: 'dist/ndarray_plugin.wasm'
        });
        NDWasm.bind(runtime);
    }, 30000);

    // --- 1-10: LIFECYCLE & DISPOSAL ---
    describe('Lifecycle Management', () => {
        test('1. fromArray (JS Array) -> pull(true)', () => {
            const wa = NDWasmArray.fromArray([[1, 2], [3, 4]]);
            expect(wa.buffer).not.toBeNull();
            const res = wa.pull(true);
            expect(res.data).toEqual(new Float64Array([1, 2, 3, 4]));
            expect(wa.buffer).toBeNull();
        });

        test('2. fromArray (Number) -> shape [1]', () => {
            const wa = NDWasmArray.fromArray(42);
            expect(wa.shape).toEqual(new Int32Array([1]));
            wa.dispose();
        });

        test('3. fromArray (NDArray) -> delegating to push()', () => {
            const nd = ndarray.array([[5, 6]]);
            const wa = NDWasmArray.fromArray(nd);
            expect(wa.shape).toEqual(new Int32Array([1, 2]));
            wa.dispose();
        });

        test('4. Double dispose is safe', () => {
            const wa = NDWasmArray.fromArray([1]);
            wa.dispose();
            expect(() => wa.dispose()).not.toThrow();
        });

        test('5. Exception: Use after dispose (pull)', () => {
            const wa = NDWasmArray.fromArray([1]);
            wa.dispose();
            expect(() => wa.pull()).toThrow(/disposed/);
        });

        test('6. Exception: Use after dispose (matMul caller)', () => {
            const wa = NDWasmArray.fromArray([[1]]);
            wa.dispose();
            const r=NDWasmArray.fromArray([[1]]);
            const res=NDWasmArray.newArray([1,1]);
            expect(() => r.matMul(wa, res)).toThrow(/Cannot read properties of null/);
            r.dispose();
            res.dispose();
        });

        test('7. Exception: Use after dispose (matMul operand)', () => {
            const a = NDWasmArray.fromArray([[1]]);
            const b = NDWasmArray.fromArray([[1]]);
            b.dispose();
            const res = NDWasmArray.newArray([1,1]);
            expect(() => a.matMul(b, res)).toThrow(/Cannot read properties of null/);
            a.dispose();
            res.dispose();
        });

        test('8. pull(false) preserves buffer', () => {
            const wa = NDWasmArray.fromArray([1, 2, 3]);
            wa.pull(false);
            expect(wa.buffer).not.toBeNull();
            wa.dispose();
        });

        test('9. size calculation check', () => {
            const wa = NDWasmArray.fromArray([[1, 2, 3], [4, 5, 6]]);
            expect(wa.size).toBe(6);
            wa.dispose();
        });

        test('10. dtype propagation', () => {
            const wa = NDWasmArray.fromArray([1, 2], 'float32');
            expect(wa.dtype).toBe('float32');
            wa.dispose();
        });
    });

    // --- 11-20: SHAPE INFERENCE & DATA INTEGRITY ---
    describe('Data and Shapes', () => {
        test('11. 1D Array', () => {
            const wa = NDWasmArray.fromArray([1, 2, 3]);
            expect(wa.shape).toEqual(new Int32Array([3]));
            wa.dispose();
        });

        test('12. 3D Array nesting', () => {
            const wa = NDWasmArray.fromArray([[[1, 2]], [[3, 4]]]);
            expect(wa.shape).toEqual(new Int32Array([2, 1, 2]));
            wa.dispose();
        });

        test('13. Empty 1D array', () => {
            const wa = NDWasmArray.fromArray([]);
            expect(wa.size).toBe(0);
            wa.dispose();
        });

        test('14. Mixed nesting fill integrity', () => {
            const wa = NDWasmArray.fromArray([[1], [2], [3]]);
            const res = wa.pull(true);
            expect(res.shape).toEqual(new Int32Array([3, 1]));
            expect(res.data).toEqual(new Float64Array([1, 2, 3]));
        });

        test('15. Large array direct fill', () => {
            const arr = Array.from({length: 1000}, (_, i) => i);
            const wa = NDWasmArray.fromArray(arr);
            expect(wa.size).toBe(1000);
            wa.dispose();
        });

        test('16. Higher dimension (4D)', () => {
            const wa = NDWasmArray.fromArray([[[[1]]]]);
            expect(wa.ndim).toBe(4);
            wa.dispose();
        });

        test('17. Float precision check', () => {
            const wa = NDWasmArray.fromArray([0.3333333333333333]);
            expect(wa.pull(true).data[0]).toBe(1/3);
        });

        test('18. Error: fromArray string', () => {
            expect(() => NDWasmArray.fromArray("fail")).toThrow();
        });

        test('19. Float32 boundary', () => {
            const wa = NDWasmArray.fromArray([1.1], 'float32');
            expect(wa.pull(true).data[0]).toBeCloseTo(1.1, 5);
        });

        test('20. ndarray.push() integration', () => {
            const nd = ndarray.zeros([2, 5]);
            const wa = nd.push();
            expect(wa.shape).toEqual(new Int32Array([2, 5]));
            wa.dispose();
        });
    });

    // --- 21-40: MATMUL & OPERANDS ---
    describe('MatMul Operations', () => {

        test('23. Chained MatMul (A * B * C)', () => {
            const a = NDWasmArray.fromArray([[2, 0], [0, 2]]);
            const b = NDWasmArray.fromArray([[0.5, 0], [0, 0.5]]); // Inverse of a
            const c = NDWasmArray.fromArray([[1, 2], [3, 4]]);
            const tmp = NDWasmArray.newArray([2, 2]);

            a.matMul(b, tmp); // result is identity
            tmp.matMul(c, b); // result should be c
            
            expect(b.pull(true).data).toEqual(new Float64Array([1, 2, 3, 4]));
            a.dispose();
            b.dispose();
            c.dispose();
            tmp.dispose();            
        });

        test('24. Dimension mismatch throws', () => {
            const a = NDWasmArray.fromArray([[1, 2, 3]]); // [1, 3]
            const b = NDWasmArray.fromArray([[1, 2], [3, 4]]); // [2, 2]
            const res = NDWasmArray.newArray([1,2]);
            expect(() => a.matMul(b, res)).toThrow(/mismatch/);
            a.dispose();
            b.dispose();
            res.dispose();
        });

        test('25. 1x3 * 3x1 = 1x1 result', () => {
            const a = NDWasmArray.fromArray([[1, 2, 3]]);
            const b = NDWasmArray.fromArray([[1], [1], [1]]);
            const res = NDWasmArray.newArray([1, 1]);
            a.matMul(b, res);
            const pulled = res.pull(true);
            expect(pulled.shape).toEqual(new Int32Array([1, 1]));
            expect(pulled.data[0]).toBe(6);
            a.dispose();
            b.dispose();
            res.dispose();
        });

        test('26. Zero Matrix', () => {
            const a = NDWasmArray.fromArray([[1, 2], [3, 4]]);
            const b = NDWasmArray.fromArray([[0, 0], [0, 0]]);
            const res = NDWasmArray.newArray([2, 2]);
            a.matMul(b, res);
            const pulled = res.pull(true);
            expect(pulled.data).toEqual(new Float64Array([0, 0, 0, 0]));
            a.dispose();
            b.dispose();
            res.dispose();
        });

        test('27. Multiplication with negatives', () => {
            const a = NDWasmArray.fromArray([[-1, 2]]);
            const b = NDWasmArray.fromArray([[-3], [4]]);
            const res = NDWasmArray.newArray([1, 1]);
            a.matMul(b, res);
            const pulled = res.pull(true); // (-1*-3) + (2*4) = 11
            expect(pulled.data[0]).toBe(11);
            a.dispose();
            b.dispose();
            res.dispose();
        });

        test('28. Vector scaling via matrix', () => {
            const a = NDWasmArray.fromArray([[10, 20]]);
            const b = NDWasmArray.fromArray([[2, 0], [0, 2]]);
            const res = NDWasmArray.newArray([1, 2]);
            a.matMul(b, res);
            expect(res.pull(true).data).toEqual(new Float64Array([20, 40]));
            a.dispose();
            b.dispose();
            res.dispose();
        });

        test('29. Large chain (WASM only)', () => {
            let res = NDWasmArray.fromArray([[1, 1], [0, 1]]);
            let step = NDWasmArray.fromArray([[1, 1], [0, 1]]);
            let tmp = NDWasmArray.newArray([2, 2]);
            for(let i = 0; i < 4; i++) {
                res.matMul(step, tmp);
                let t=res; res=tmp; tmp=t; // Swap
            }
            // Matrix [[1, 1], [0, 1]]^5 = [[1, 5], [0, 1]]
            const data = res.pull(true).data;
            expect(data[1]).toBe(5);
            res.dispose();
            step.dispose();
            tmp.dispose();
        });

        test('30. Float32 MatMul', () => {
            const a = NDWasmArray.fromArray([[1, 2]], 'float32');
            const b = NDWasmArray.fromArray([[2], [2]], 'float32');
            const res = NDWasmArray.newArray([1, 1],"float32");
            a.matMul(b, res);
            expect(res.dtype).toBe('float32');
            expect(res.pull(true).data[0]).toBe(6);
            a.dispose();
            b.dispose();
            res.dispose();
        });

        test('31. Rectangular 3x2 * 2x1', () => {
            const a = NDWasmArray.fromArray([[1, 1], [2, 2], [3, 3]]);
            const b = NDWasmArray.fromArray([[10], [10]]);
            const res = NDWasmArray.newArray([3, 1]);
            a.matMul(b, res);
            expect(res.shape).toEqual(new Int32Array([3, 1]));
            expect(res.pull(true).data).toEqual(new Float64Array([20, 40, 60]));
            a.dispose();
            b.dispose();
            res.dispose();
        });

        test('32. matMul with itself (square)', () => {
            const a = NDWasmArray.fromArray([[1, 2], [3, 4]]);
            const res = NDWasmArray.newArray([2, 2]);
            a.matMul(a, res);
            expect(res.pull(true).data).toEqual(new Float64Array([7, 10, 15, 22]));
            a.dispose();
            res.dispose();
        });

        test('33. matMul(NDArray) non-contiguous', () => {
            const a = NDWasmArray.fromArray([[1, 0], [0, 1]]);
            let b = new NDArray(new Float64Array([1, 2, 3, 4]), { shape: [2, 2], strides: [1, 2] }); // Transposed view
            b = b.push(); // Push to WASM
            const res = NDWasmArray.newArray([2, 2]);
            a.matMul(b, res);
            expect(res.pull(true).data).toEqual(new Float64Array([1, 3, 2, 4]));
            a.dispose();
            b.dispose();
            res.dispose();
        });

        test('34. Error: MatMul rank 1', () => {
            const a = NDWasmArray.fromArray([1, 2, 3]);
            const b = NDWasmArray.fromArray([1, 2, 3]);
            const res = NDWasmArray.newArray([1, 1]);
            expect(() => a.matMul(b, res)).toThrow();
            a.dispose();
            b.dispose();
            res.dispose();
        });

        test('35. Very small floats', () => {
            const a = NDWasmArray.fromArray([[1e-8]]);
            const b = NDWasmArray.fromArray([[1e-8]]);
            const res = NDWasmArray.newArray([1, 1]);
            a.matMul(b, res);
            expect(res.pull(true).data[0]).toBeCloseTo(1e-16, 20);
            a.dispose();
            b.dispose();
            res.dispose();
        });

        test('36. Identity 100x100 performance', () => {
            const identity = Array.from({length: 100}, (_, i) => 
                Array.from({length: 100}, (_, j) => i === j ? 1 : 0)
            );
            const a = NDWasmArray.fromArray(identity);
            const res = NDWasmArray.newArray([100, 100]);
            a.matMul(a, res);
            expect(res.pull(true).size).toBe(10000);
            a.dispose();
            res.dispose();
        });

        test('37. Mixed Types: push() then matMul(JS)', () => {
            const a = ndarray.array([[1, 2], [3, 4]]).push();
            const b=NDWasmArray.fromArray([[1, 1], [1, 1]]);
            const res = NDWasmArray.newArray([2, 2]);
            a.matMul(b, res);
            expect(res.pull(true).data).toEqual(new Float64Array([3, 3, 7, 7]));
            a.dispose();
            b.dispose();
            res.dispose();
        });


        test('39. Rectangular 1x1 * 1x10', () => {
            const a = NDWasmArray.fromArray([[2]]);
            const b = NDWasmArray.fromArray([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]);
            const res = NDWasmArray.newArray([1, 10]);
            a.matMul(b, res);
            expect(res.pull(false).shape).toEqual(new Int32Array([1, 10]));
            expect(res.pull(true).data[9]).toBe(20);
            a.dispose();
            b.dispose();
            res.dispose();
        });

        test('40. Deeply nested chained memory', () => {
            const a = NDWasmArray.fromArray([[1, 0], [0, 1]]);
            const br = NDWasmArray.fromArray([[2, 0], [0, 2]]);
            const cr = NDWasmArray.fromArray([[3, 0], [0, 3]]);
            const dr = NDWasmArray.fromArray([[4, 0], [0, 4]]);
            const b = NDWasmArray.newArray([2, 2]);
            const c = NDWasmArray.newArray([2, 2]);
            const d = NDWasmArray.newArray([2, 2]);
            a.matMul(br, b);
            b.matMul(cr, c);
            c.matMul(dr, d);
            expect(d.pull(true).data[0]).toBe(24);
            [a, b, c, d, br, cr, dr].forEach(x => x.dispose());
        });
    });

    // --- 41-50: BATCH MATMUL & STRESS ---
    describe('Batch MatMul', () => {
        test('41. Batch [2, 2, 2] Identity', () => {
            const a = NDWasmArray.fromArray([[[1, 0], [0, 1]], [[1, 0], [0, 1]]]);
            const b = NDWasmArray.fromArray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
            const res = NDWasmArray.newArray([2, 2, 2]);
            a.matMulBatch(b, res);
            expect(res.pull(true).data).toEqual(new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]));
            a.dispose();
            b.dispose();
            res.dispose();
        });

        test('42. Batch scaling [2, 2, 2]', () => {
            const a = NDWasmArray.fromArray([[[2, 0], [0, 2]], [[3, 0], [0, 3]]]);
            const b = NDWasmArray.fromArray([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]);
            const res = NDWasmArray.newArray([2, 2, 2]);
            a.matMulBatch(b, res);
            expect(res.pull(true).data).toEqual(new Float64Array([2, 2, 2, 2, 3, 3, 3, 3]));
            a.dispose();
            res.dispose();
        });

        test('43. Batch size mismatch', () => {
            const a = NDWasmArray.fromArray([0,0,0,0,0,0,0,0]); // shape [8]
            // We need proper 3D to even enter the logic
            const a3d = NDWasmArray.fromArray([[[1]]]); // [1, 1, 1]
            const b3d = NDWasmArray.fromArray([[[1]], [[1]]]); // [2, 1, 1]
            const res = NDWasmArray.newArray([2, 1, 1]);
            expect(() => a3d.matMulBatch(b3d, res)).toThrow(/dimensions mismatch/);
            a3d.dispose();
            b3d.dispose();
            res.dispose();
        });

        test('44. Batch MatMul rank check', () => {
            const a = NDWasmArray.fromArray([[1, 2], [3, 4]]);
            const res = NDWasmArray.newArray([2, 2]);
            expect(() => a.matMulBatch(a, res)).toThrow(/mismatch/);
            a.dispose();
            res.dispose();
        });

        test('45. [2, 1, 3] * [2, 3, 2]', () => {
            const a = NDWasmArray.fromArray([ [[1, 1, 1]], [[2, 2, 2]] ]);
            const b = NDWasmArray.fromArray([ [[1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1], [1, 1]] ]);
            const res = NDWasmArray.newArray([2, 1, 2]);
            a.matMulBatch(b, res);
            expect(res.pull(false).shape).toEqual(new Int32Array([2, 1, 2]));
            expect(res.pull(true).data).toEqual(new Float64Array([3, 3, 6, 6]));
            a.dispose();
            b.dispose();
            res.dispose();
        });

        test('46. Batch result disposal', () => {
            const a = NDWasmArray.fromArray([[[1]]]);
            const res = NDWasmArray.newArray([1, 1, 1]);
            a.matMulBatch(a, res);
            res.dispose();
            expect(res.buffer).toBeNull();
            a.dispose();
        });

        test('47. Mixed Batch operands (NDWasm + NDArray)', () => {
            const a = NDWasmArray.fromArray([[[1, 0], [0, 1]]]);
            const b = NDWasmArray.fromArray(ndarray.zeros([1, 2, 2]));
            const res = NDWasmArray.newArray([1, 2, 2]);
            a.matMulBatch(b, res);
            expect(res.pull(true).data[0]).toBe(0);
            a.dispose();
            b.dispose();
            res.dispose();
        });

        test('48. Large batch stress (100 batches)', () => {
            const data = Array.from({length: 100}, () => [[1, 0], [0, 1]]);
            const a = NDWasmArray.fromArray(data);
            const b= NDWasmArray.fromArray(data);
            const res = NDWasmArray.newArray([100, 2, 2]);
            a.matMulBatch(b, res);
            expect(res.pull(true).shape[0]).toBe(100);
            a.dispose();
            b.dispose();
            res.dispose();
        });

        test('49. Stress: Rapid Allocate/Dispose', () => {
            for(let i=0; i<50; i++) {
                const wa = NDWasmArray.fromArray([[i]]);
                wa.dispose();
            }
            expect(true).toBe(true);
        });

        test('50. Chained Batch MatMul', () => {
            const a = NDWasmArray.fromArray([[[2]]]);            
            const b = NDWasmArray.newArray([1, 1, 1]);
            a.matMulBatch(a, b);
            const c = NDWasmArray.newArray([1, 1, 1]);
            b.matMulBatch(a, c);
            expect(c.pull(true).data[0]).toBe(8);
            a.dispose(); b.dispose(); c.dispose();
        });
    });
});