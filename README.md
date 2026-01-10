# ndarray

A fast, NumPy-like multi-dimensional array library for JavaScript, powered by WebAssembly for high-performance computations.

## About The Project

This library provides an `NDArray` class that mimics the rich, expressive API of Python's NumPy library, enabling developers to perform complex mathematical and linear algebra operations with the speed of WebAssembly and the convenience of JavaScript.

It supports a wide range of operations, from basic element-wise arithmetic to advanced linear algebra decompositions, signal processing, and statistical analysis.

## Features

-   **Familiar API**: Designed to be intuitive for developers coming from NumPy.
-   **WebAssembly Core**: Critical, performance-sensitive operations are executed in a WASM core written in Go for near-native speed.
-   **Broadcasting**: Sophisticated element-wise operations on arrays of different shapes.
-   **Advanced Indexing**: Powerful slicing, dicing, and "fancy indexing" capabilities.
-   **Rich Functionality**: Includes modules for BLAS, LAPACK (decompositions, solvers), signal processing (FFT, convolutions), and statistical analysis.
-   **Built-in Documentation**: Explore the API at runtime using the `help()` and `helphtml()` functions.

## Architecture

This library employs a hybrid architecture to balance ease of use with maximum performance:

1.  **High-Level JavaScript API**: The primary interface is a pure JavaScript `NDArray` class. It manages the array's metadata (shape, strides, dtype) and provides a rich set of methods that mimic the NumPy API. All high-level logic, such as broadcasting calculations and view manipulations (slicing, reshaping), happens at this layer.

2.  **JIT-Compiled JavaScript Kernels**: For element-wise operations (`add`, `mul`, `map`, `reduce`, etc.), the library acts as a mini-compiler. It dynamically generates highly optimized, unrolled JavaScript loop functions (kernels) tailored to the specific shapes and memory layouts of the input arrays. These kernels use direct pointer arithmetic on the underlying `TypedArray` data buffers, avoiding the overhead of multi-dimensional index lookups in loops. Kernels are cached, so the compilation cost is paid only once per unique array configuration.

3.  **Go-based WebAssembly Core**: Computationally intensive tasks that cannot be easily parallelized or optimized in JavaScript are delegated to a pre-compiled WebAssembly module written in Go. This includes operations for:
    -   **BLAS** (Basic Linear Algebra Subprograms): Matrix multiplication, vector multiplication, etc.
    -   **LAPACK** (Linear Algebra PACKage): Decompositions (LU, SVD, QR), linear equation solvers, and eigenvalue computations.
    -   **Signal Processing**: Fast Fourier Transforms (FFT), convolutions, and correlations.
    -   **Analysis**: Sorting (`argsort`), statistical functions (`cov`), and more.
    -   **Image Processing**: Image processing functions.
    -   **Optimize**: Optimization module for linear programming, non-linear minimization, and linear regression.

This tiered approach ensures that you get the expressiveness of a dynamic language and the raw speed of compiled code where it matters most.

## The NDArray Data Structure

At the heart of the library is the `NDArray` class, which represents a multi-dimensional array. Unlike a nested JavaScript array, an NDArray stores its data in a single, flat `TypedArray`. The multi-dimensional structure is achieved by mapping logical indices to physical memory locations using metadata.

An `NDArray` instance has the following key properties:

-   `data`: The underlying `TypedArray` (e.g., `Float64Array`, `Int32Array`) that holds the raw data in a contiguous block of memory.
-   `shape`: An `Int32Array` describing the size of each dimension. For example, a 2x3 matrix has a shape of `[2, 3]`.
-   `strides`: An `Int32Array` that dictates how many steps to take in memory to advance one index along a given dimension. For a standard C-style (row-major) 2x3 matrix, the strides would be `[3, 1]`. To move to the next row (dimension 0), you jump 3 elements; to move to the next column (dimension 1), you jump 1 element.
-   `offset`: The starting position of the view within the `data` buffer. This allows multiple NDArrays (views) to share the same underlying data without copying it.
-   `dtype`: A string specifying the data type (e.g., `'float64'`, `'int32'`).
-   `isContiguous`: A boolean flag that is `true` if the elements of the array are laid out in standard C-style (row-major) order in memory. This is a critical optimization flag; contiguous arrays can be processed with much faster, simpler loops. Operations like `transpose` or certain slices create non-contiguous views.

This structure allows for powerful, memory-efficient "views" of data. For example, slicing an array does not create a copy of the data; it creates a new `NDArray` instance with a different shape, strides, and offset that points to the same underlying `data` buffer.

## Computation Methods

The library intelligently delegates computations to the most appropriate engine:

### Element-wise Operations & Broadcasting

Standard arithmetic (`add`, `sub`, `mul`, `div`), comparison (`eq`, `gt`), and logical (`logical_and`) operators are implemented in JavaScript. They fully support NumPy-style broadcasting, allowing you to perform element-wise operations on arrays of different-but-compatible shapes.

```javascript
const a = ndarray.arange(0, 4).reshape(2, 2); // [[0, 1], [2, 3]]
const b = ndarray.array([10, 20]);              // A 1D array

// Broadcast `b` across the rows of `a`
const c = a.add(b);
// => array([[10, 21],
//           [12, 23]], dtype=float64)
```

### Just-in-Time (JIT) JavaScript Kernels

The performance of element-wise operations is supercharged by a Just-in-Time compilation engine. When you call a method like `add`, `map`, or `sum` for the first time with a specific combination of array shapes, strides, and dtypes, the library generates a custom JavaScript function (a "kernel").

This kernel contains unrolled, nested loops that iterate over the data buffers using simple pointer arithmetic. This avoids all the overhead of bounds checking and multi-dimensional index calculation on a per-element basis. The generated kernel is then cached and reused for subsequent calls with the same array geometry.

### WebAssembly Functions

For algorithms where raw computational power is paramount, the library calls its Go-based WebAssembly module. Before execution, the JavaScript `NDArray` data is copied into the WASM's linear memory. The exported Go function runs at near-native speed, and the result is copied back into a new `NDArray`.

This approach is reserved for complex operations where the overhead of the memory copy is small compared to the computational savings. Key WASM-powered modules include:

-   `NDArray.blas`: Matrix multiplication (`matmul`), matrix-vector products (`matVecMul`), rank-updates (`syrk`), and more.
-   `NDArray.decomp`: Linear equation solving (`solve`), matrix inversion (`inv`), and decompositions (`svd`, `qr`, `lu`, `cholesky`, `eigen`).
-   `NDArray.signal`: Fourier analysis (`fft`, `ifft`), convolutions (`conv2d`), and correlations.
-   `NDArray.analysis`: Sorting (`argsort`), statistical functions (`cov`), and more.
-   `NDArray.image`: Image processing functions.
-   `NDArray.optimize`: Optimization module for linear programming, non-linear minimization, and linear regression.


```javascript
// Perform matrix multiplication in WebAssembly
const m1 = ndarray.arange(0, 4).reshape(2, 2);
const m2 = ndarray.eye(2);
const m3 = m1.matmul(m2); // Dispatches to WASM
```

## Installation

Install the library using npm:

```bash
npm install ndarray
```

## How to Use

First, make sure you have initialized the WebAssembly module, especially if you plan to use linear algebra or signal processing functions.

```javascript
import * as ndarray from 'ndarray';

// It's best practice to initialize the WASM runtime once when your app starts.
// The `init()` function will fetch and instantiate the wasm module.
await ndarray.NDWasm.init();
```

### Creating Arrays

You can create `NDArray` instances from nested JavaScript arrays or use built-in factory functions.

```javascript
// From a nested array
const a = ndarray.array([[1, 2], [3, 4]]);

// A 3x3 matrix of all zeros
const z = ndarray.zeros([3, 3]);

// A 1D array from 0 up to (but not including) 10
const r = ndarray.arange(0, 10);

// A 2x2 identity matrix
const i = ndarray.eye(2);

// A 2x2 array of random numbers
const rand = ndarray.NDArray.random.rand([2, 2]);
```

### Indexing & Slicing

The library supports powerful and expressive indexing capabilities to select and modify data.

```javascript
const m = ndarray.arange(0, 9).reshape(3, 3);
// => array([[0, 1, 2],
//           [3, 4, 5],
//           [6, 7, 8]], dtype=float64)

// Get a single element (scalar)
const el = m.get(0, 1); // => 1

// Get the first row (1D view)
const row0 = m.slice(0, null); // => array([0, 1, 2], dtype=float64)

// Get the last column (1D view)
const col2 = m.slice(null, 2); // => array([2, 5, 8], dtype=float64)

// Get a 2x2 sub-matrix from the top-right corner (2D view)
const sub = m.slice([0, 2], [1, 3]); // => array([[1, 2], [4, 5]], dtype=float64)

// Set a value
m.set(99, 0, 0);

// Set an entire row using broadcasting
m.slice(1, null).set(ndarray.array([10, 11, 12]));
```

### Views vs. Copies

One of the most powerful features is the use of memory-efficient "views". Slicing an array does **not** create a new copy of the data. It creates a new `NDArray` object that points to the same underlying data buffer with a different shape, stride, or offset.

```javascript
const base = ndarray.arange(0, 4); // array([0, 1, 2, 3])
const view = base.slice([1, 3]);   // array([1, 2])

// Modifying the view also modifies the original array's data
view.set(99, 0); // Sets the first element of the view

console.log(base.toString());
// => array([0, 99, 2, 3], dtype=float64)
```

To create a new array with its own data, use the `.copy()` method.

```javascript
const copy = base.copy();
copy.set(0, 0); // Does NOT affect `base`
```

### Using WebAssembly for High-Performance Computations

For heavy-duty tasks like matrix multiplication, you can use the methods that are backed by the WebAssembly core.

```javascript
const m1 = ndarray.array([[1, 2], [3, 4]]);
const m2 = ndarray.array([[5, 6], [7, 8]]);

// This operation is dispatched to the Go WASM core
const result = m1.matmul(m2);

console.log(result.toString());
// => array([[19, 22],
//           [43, 50]], dtype=float64)

// You can also compute the SVD (Singular Value Decomposition)
const { u, s, v } = m1.svd();
```

For more advanced and comprehensive usage examples, please refer to the files in the `test/` directory of this project.

## API Documentation

A live, searchable version of the API documentation can be found here:
[**Full API Documentation**](https://kikoqiu.github.io/ndarray/)

This library is also self-documenting. You can get help on any class or function at runtime.

### Help

Use the `help()` function to get formatted documentation. You can pass either a string name or the function/class object itself.

```javascript
import { zeros, help } from 'ndarray';

// Get help by name
help.helpdoc('NDArray.prototype.matmul');

// Get help by object reference
const a = zeros([2, 2]);
help.helpdoc(a.matmul);
```

### HTML Help

Use the `helphtml()` function to get a formatted HTML string, which can be injected into a web page. CSS classes are provided for easy styling.

```javascript
import { help } from 'ndarray';

const htmlDoc = help.helphtml('NDArray.prototype.svd');
// document.getElementById('help-content').innerHTML = htmlDoc;
```

A full, searchable HTML documentation site can also be generated. See the "Building from Source" section.

## Building from Source

To build the project from its source code, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:kikoqiu/ndarray.git
    cd ndarray
    ```

2.  **Install dependencies:**
    ```bash
    npm install
    ```

3.  **Run the build script:**
    This command will lint the code, generate the `docs.json` for the help system, and bundle the library into `dist/` in multiple formats (MJS, CJS, IIFE).
    ```bash
    npm run build
    ```

4.  **Generate HTML Documentation (Optional):**
    To generate a full HTML documentation site into the `docs/` directory, run:
    ```bash
    npm run doc:html
    ```
