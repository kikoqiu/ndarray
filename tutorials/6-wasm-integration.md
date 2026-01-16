# Tutorial 6: High-Performance Computing with WebAssembly

For computationally intensive tasks, `ndarray` leverages a pre-compiled WebAssembly (WASM) module written in Go. This provides near-native speed for complex algorithms that benefit from low-level optimizations, such as linear algebra, image processing, and numerical optimization.

## Initializing the WASM Module

Before you can use any WASM-powered function, you **must** initialize the runtime. This is an asynchronous operation that fetches and prepares the WASM module. It's best to do this once when your application starts.

```javascript
import * as ndarray from 'ndarray';

// This will fetch 'ndarray_plugin.wasm' and 'wasm_exec.js'
// from the same directory as your script.
await ndarray.init();
console.log("WASM runtime initialized.");

// You can also specify a different base directory if your files are elsewhere.
// await ndarray.init('./path/to/wasm_files/');
```

## Using WASM-Powered Functions

Once initialized, you can call WASM-powered functions directly from the `ndarray` object. The library handles the necessary memory management (copying data to and from WASM) for you.

### Linear Algebra with BLAS

The `ndarray.blas` module (Basic Linear Algebra Subprograms) provides highly optimized routines for matrix and vector operations.

```javascript
// Example: Matrix Multiplication (matMul)
const m1 = ndarray.array([[1, 2], [3, 4]]);
const m2 = ndarray.array([[5, 6], [7, 8]]);

// The .matMul() method (among others like .matVecMul())
// automatically uses the underlying WASM BLAS implementation.
const result = m1.matMul(m2);

console.log(result.toString());
// => array([[19, 22],
//           [43, 50]], dtype=float64)
```

Other linear algebra functions, including decompositions (LU, SVD, QR) and solvers, are available via the `ndarray.decomp` module.

### Image Processing

The `ndarray.image` module provides functions for decoding and encoding various image formats, enabling efficient image manipulation.

#### Decoding an Image

You can decode binary image data (e.g., from a PNG or JPEG file) directly into an `NDArray`. The result is typically a `[height, width, 4]` array with `uint8c` dtype (RGBA channels).

```javascript
// Example: Decoding an image (assuming 'imageBytes' is a Uint8Array of image data)
// In a browser, you might fetch it:
// const imageBlob = await fetch('./my-image.png').then(res => res.blob());
// const imageBytes = new Uint8Array(await imageBlob.arrayBuffer());

// For demonstration, create dummy image bytes (simplified)
const dummyPngBytes = new Uint8Array([
    137, 80, 78, 71, 13, 10, 26, 10, // PNG signature
    0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0, 0, 0, 31, 21, 106, 137, // IHDR (1x1 RGBA)
    0, 0, 0, 12, 73, 68, 65, 84, 120, 156, 99, 0, 1, 0, 0, 5, 0, 1, 2, 78, 120, 4, // IDAT (pixel data: red)
    0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130 // IEND
]);
const imageArr = ndarray.image.decode(dummyPngBytes);

if (imageArr) {
    console.log(`Decoded image shape: ${imageArr.shape}`); // e.g., [1, 1, 4]
    console.log(`Pixel data: ${imageArr.get(0, 0, 0)}, ${imageArr.get(0, 0, 1)}, ${imageArr.get(0, 0, 2)}, ${imageArr.get(0, 0, 3)}`);
    // Output for red pixel: 255, 0, 0, 255
}

// Example: Encoding an NDArray back to PNG
const newPngBytes = ndarray.image.encodePng(imageArr);
// `newPngBytes` is now a Uint8Array containing the PNG image data.

// Utility to convert to a Data URL for display in HTML (browser only)
// const dataUrl = ndarray.image.convertUint8ArrrayToDataurl(newPngBytes, 'image/png');
// document.getElementById('myImage').src = dataUrl;
```

The WASM integration ensures that even the most demanding computational tasks can be handled efficiently, bridging the gap between JavaScript's flexibility and Go's performance.
