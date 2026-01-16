const ndarray = require('../dist/ndarray.cjs');
const { NDWasm, WasmRuntime } = ndarray;

// This is not a typical test file. It runs benchmarks and logs them to the console.
// It will always "pass" as long as it doesn't throw an error.
describe('Performance Benchmarks', () => {
    
    const size = 1000;
    const largeArr = ndarray.arange(size * size).reshape([size, size]);
    const transposedArr = largeArr.transpose();

    // JS implementation for comparison
    const naiveMatMul = (a, b) => {
        const [rowsA, colsA] = a.shape;
        const [rowsB, colsB] = b.shape;
        const result = ndarray.zeros([rowsA, colsB]);
        for (let i = 0; i < rowsA; i++) {
            for (let j = 0; j < colsB; j++) {
                let sum = 0;
                for (let k = 0; k < colsA; k++) {
                    sum += a.get(i, k) * b.get(k, j);
                }
                result.set(sum, i, j);
            }
        }
        return result;
    };
    
    // This 'test' is a container for all benchmark runs.
    test('run all benchmarks', async () => {
        console.log('\n--- Performance Benchmarks ---');

        // --- Contiguous vs Non-contiguous ---
        console.log('\n1. Contiguous vs. Non-Contiguous Operations:');
        
        console.time('add() on contiguous array');
        largeArr.add(1);
        console.timeEnd('add() on contiguous array');

        console.time('add() on non-contiguous array (view)');
        transposedArr.add(1);
        console.timeEnd('add() on non-contiguous array (view)');
        
        console.time('sum(axis=1) on contiguous array');
        largeArr.sum(1);
        console.timeEnd('sum(axis=1) on contiguous array');

        console.time('sum(axis=1) on non-contiguous array (view)');
        transposedArr.sum(1);
        console.timeEnd('sum(axis=1) on non-contiguous array (view)');

        // --- WASM vs JS ---
        console.log('\n2. WASM vs. Naive JS Matrix Multiplication (100x100):');
        const matSize = 100;
        const matA = ndarray.arange(matSize * matSize).reshape([matSize, matSize]);
        const matB = matA.transpose();
        
        // Init WASM
        const runtime = new WasmRuntime();
        await runtime.init({ execUrl: 'dist/wasm_exec.js', wasmUrl: 'dist/ndarray_plugin.wasm' });
        NDWasm.bind(runtime);

        console.time('matMul() (WASM)');
        matA.matMul(matB);
        console.timeEnd('matMul() (WASM)');
        
        console.time('matMul() (naive JS)');
        naiveMatMul(matA, matB);
        console.timeEnd('matMul() (naive JS)');

        console.log('\n--------------------------------\n');
        
        // This is just a dummy assertion to make Jest happy.
        expect(true).toBe(true);
    }, 60000); // Long timeout for benchmarks
});
