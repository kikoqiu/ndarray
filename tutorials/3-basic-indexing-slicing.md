# Tutorial 3: Basic Indexing and Slicing

This tutorial introduces the fundamental ways to access and manipulate `NDArray` elements using `.slice()` and `.get()`. Slicing operations create "views" into your data, offering a powerful and memory-efficient way to work with sub-arrays without copying large amounts of data.

The primary method for slicing is `ndarray.slice(...specs)`, where `...specs` are parameters for each dimension.

## Scalar Indexing (Reducing Dimensions)

Providing a single number for a dimension's spec will select that specific index along the axis. This reduces the dimensionality of the resulting `NDArray` view by one.

```javascript
import * as ndarray from 'ndarray';

// Create a 3x4 matrix
const m = ndarray.arange(12).reshape(3, 4);
// => array([[0, 1, 2, 3],
//           [4, 5, 6, 7],
//           [8, 9, 10, 11]], dtype=float64)

// Access a single element (returns a 0-dimensional NDArray)
const val = m.slice(1, 2); // Get element at row 1, column 2
console.log(val.toString()); // => array(6, dtype=float64)
console.log(val.get());     // Get raw value: 6

// Get the second row (index 1) of the matrix
// This reduces the 2D matrix view to a 1D NDArray
const row1 = m.slice(1); // Shorthand for m.slice(1, null)
console.log(row1.toString()); // => array([4, 5, 6, 7], dtype=float64)

// Get the last column (index 3) of the matrix
// This reduces the 2D matrix view to a 1D NDArray
const col3 = m.slice(null, 3); // Shorthand for selecting all rows, then column 3
console.log(col3.toString()); // => array([3, 7, 11], dtype=float64)
```

## Range Slicing (Preserving Dimensions)

To select a range of elements along an axis, you provide a tuple `[start, end, step]`. This preserves the number of dimensions in the resulting view.

-   `start`: The starting index (inclusive). If `null`, defaults to 0.
-   `end`: The ending index (exclusive). If `null`, defaults to the end of the dimension.
-   `step`: The increment between indices. Defaults to 1. If negative, it reverses the order.

```javascript
const m = ndarray.arange(12).reshape(3, 4);

// Get the first two rows and columns 1-2
// Rows: [0, 2] (indices 0, 1)
// Cols: [1, 3] (indices 1, 2)
const sub_matrix = m.slice([0, 2], [1, 3]);
// => array([[1, 2],
//           [5, 6]], dtype=float64)

// Get all rows, but only every other column
// Cols: start=0, end=4, step=2 => indices 0, 2
const alternate_cols = m.slice(null, [0, 4, 2]);
// => array([[0, 2],
//           [4, 6],
//           [8, 10]], dtype=float64)
```

## Negative Indexing and Steps

You can use negative indices to count from the end of a dimension. A negative `step` value will reverse the order of elements along that axis.

```javascript
const a = ndarray.arange(5);
// => array([0, 1, 2, 3, 4], dtype=float64)

// Get the last element using a negative scalar index
const last_el = a.slice(-1); // => array(4, dtype=float64)

// Get the last 3 elements using a negative start index
const last_three = a.slice([-3, null]);
// => array([2, 3, 4], dtype=float64)

// Reverse the entire array using a negative step
const reversed_a = a.slice([null, null, -1]);
// => array([4, 3, 2, 1, 0], dtype=float64)

// Reverse the rows of a 2D array
const m_rev_rows = ndarray.arange(4).reshape(2, 2).slice([null, null, -1], null);
// => array([[2, 3],
//           [0, 1]], dtype=float64)
```

## Helper Views: `rowview()` and `colview()`

For 2D arrays, `rowview()` and `colview()` provide convenient shortcuts for extracting rows and columns.

```javascript
const matrix = ndarray.array([[1, 2, 3], [4, 5, 6]]);

// Get the first row
const firstRow = matrix.rowview(0); // Equivalent to matrix.slice(0, null)
// => array([1, 2, 3], dtype=float64)

// Get the second column
const secondCol = matrix.colview(1); // Equivalent to matrix.slice(null, 1)
// => array([2, 5], dtype=float64)
```
