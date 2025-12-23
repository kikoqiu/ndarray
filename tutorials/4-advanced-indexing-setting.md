# Tutorial 4: Advanced Indexing with `pick()` and Complex `set()` Operations

This tutorial delves into `ndarray`'s advanced indexing capabilities using the `.pick()` and `.set()` methods. These methods allow for highly flexible, non-contiguous data selection and modification, often mirroring powerful NumPy patterns.

## `pick()`: Extracting Data into a New Array

The `.pick()` method is used for "fancy indexing," where you specify element positions using arrays of indices. It always returns a **new `NDArray` (a copy)** with its own data, not a view. This is useful for creating new arrays from arbitrarily selected elements.

The signature is `ndarray.pick(...specs)`, where `...specs` can include numbers (for dimension reduction) or arrays of indices.

```javascript
import * as ndarray from 'ndarray';

const m = ndarray.arange(9).reshape(3, 3);
// => array([[0, 1, 2],
//           [3, 4, 5],
//           [6, 7, 8]], dtype=float64)

// Pick specific elements from a 1D array
const a = ndarray.arange(10); // [0, 1, ..., 9]
const picked_1d = a.pick([5, 2, 8, 0]);
// => array([5, 2, 8, 0], dtype=float64)

// Pick specific rows in a custom order
const picked_rows = m.pick([2, 0], null); // Get row 2, then row 0
// => array([[6, 7, 8],
//           [0, 1, 2]], dtype=float64)

// Pick specific columns in a custom order
const picked_cols = m.pick(null, [2, 0, 1]); // Get col 2, then col 0, then col 1
// => array([[2, 0, 1],
//           [5, 3, 4],
//           [8, 6, 7]], dtype=float64)

// Orthogonal (grid) picking: combine row and column index arrays
// Selects elements at (row 0, col 1), (row 0, col 2), (row 2, col 1), (row 2, col 2)
const sub_grid = m.pick([0, 2], [1, 2]);
// => array([[1, 2],
//           [7, 8]], dtype=float64)
```

## Complex `set()` Operations

The `.set()` method is used to assign values to parts of an array. It is highly versatile and supports:
-   Scalar assignment.
-   Assignment from another `NDArray` or JavaScript array.
-   Broadcasting of the assigned value to the target region.
-   **Advanced indexing**: using arrays of indices to specify target locations.

The signature is `ndarray.set(value, ...indices)`.

### Setting into a `slice` (View Mutation)

As seen in previous tutorials, `slice` returns a view. Modifying this view with `set` changes the original array's data.

```javascript
const m = ndarray.arange(9).reshape(3, 3);
// => array([[0, 1, 2],
//           [3, 4, 5],
//           [6, 7, 8]], dtype=float64)

// Get a view of the second row
const row_view = m.slice(1); // [3, 4, 5]

// Set the entire row view to all 99s
row_view.set(99);
// 'm' is now:
// => array([[0, 1, 2],
//           [99, 99, 99],
//           [6, 7, 8]], dtype=float64)

// Set only the first element of the first column view to 0
m.slice(null, 0).set(0, 0); // Modifies m.get(0,0)
// 'm' is now:
// => array([[0, 1, 2],
//           [99, 99, 99],
//           [6, 7, 8]], dtype=float64)
```

### Advanced Setting with Index Arrays and Broadcasting

You can pass arrays of indices to `set()` to target specific elements. The `value` to be set will be broadcast against the target region.

```javascript
const grid = ndarray.zeros([4, 4]);

// Set the elements at (0,0), (1,1), (2,2), (3,3) to 1
grid.set(1, [0, 1, 2, 3], [0, 1, 2, 3]);

// Set all elements in columns 1 and 3 to 5
grid.set(5, null, [1, 3]);

console.log(grid.toString());
// => array([[1, 5, 0, 5],
//           [0, 1, 0, 5],
//           [0, 0, 1, 5],
//           [0, 0, 0, 1]], dtype=float64)
```

#### Complex Example: Setting a Sub-grid

You can even assign a smaller `NDArray` to a specific, non-contiguous sub-grid selected by index arrays, leveraging broadcasting.

```javascript
const m = ndarray.arange(16).reshape(4, 4);
// Original m:
// [[ 0,  1,  2,  3],
//  [ 4,  5,  6,  7],
//  [ 8,  9, 10, 11],
//  [12, 13, 14, 15]]

const values_to_set = ndarray.array([[100, 101], [200, 201]]);

// Target rows: 1 and 3
// Target columns: 0 and 2
// The 'values_to_set' (2x2) will be assigned to these four specific locations:
// (1,0), (1,2)
// (3,0), (3,2)
m.set(values_to_set, [1, 3], [0, 2]);

console.log(m.toString());
// Resulting m:
// [[  0,  1,  2,  3],
//  [100,  5, 101,  7], // Row 1 modified
//  [  8,  9, 10, 11],
//  [200, 13, 201, 15]] // Row 3 modified
```

### Chained `slice()` and `set()` for Powerful Transformations

You can chain `slice()` calls to pinpoint regions, and then use `set()` to modify them. The JIT engine optimizes this to be very efficient.

```javascript
const arr = ndarray.arange(6).reshape(2, 3);
// => array([[0, 1, 2],
//           [3, 4, 5]], dtype=float64)

// Get a view of the first two columns (rows: all, cols: [0, 2])
// Then transpose this view (shape becomes [2, 2])
// Then get the first row of the transposed view (shape becomes [2])
const target_view = arr.slice(null, [0, 2]).transpose().slice(0);
// This view corresponds to original elements at (0,0) and (1,0), which are [0, 3]

// Set these two elements to new values
target_view.set([99, 88]);

console.log(arr.toString());
// => array([[99, 1, 2],
//           [88, 4, 5]], dtype=float64)
```
These advanced combinations of `slice`, `pick`, and `set` provide unparalleled flexibility and control over your array data, allowing you to perform complex data manipulations concisely and efficiently.
