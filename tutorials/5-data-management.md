# Tutorial 5: Data Management: Views, Copies, and Contiguity

Understanding how `ndarray` manages its data in memory is crucial for writing efficient and correct code. This tutorial clarifies the distinction between views and copies, and introduces the concept of contiguity.

## Views: Efficient Data Access

Many `ndarray` operations, such as `.slice()`, `.transpose()`, and `.reshape()` (when the array is contiguous), create **views**. A view is a new `NDArray` object that refers to the **same underlying data buffer** as the original array. This makes view operations extremely fast and memory-efficient as no data is duplicated.

### Modifying Views Affects the Original

Because views share data, any changes made through a view will be reflected in the original array.

```javascript
import * as ndarray from 'ndarray';

const original = ndarray.arange(5);
// => array([0, 1, 2, 3, 4], dtype=float64)

// Create a slice (which is a view)
const middle_view = original.slice([1, 4]); // [1, 2, 3]

// Modify an element in the view
middle_view.set(99, 1); // Set the element at index 1 of the view (which is original[2])

console.log(original.toString());
// => array([0, 1, 99, 3, 4], dtype=float64)

// Transpose also creates a view
const m = ndarray.array([[1, 2], [3, 4]]);
const t = m.transpose(); // t is a view of m
t.set(100, 0, 1); // Sets element t[0,1] which is m[1,0]
console.log(m.toString());
// => array([[1, 2],
//           [100, 4]], dtype=float64)
```

## Copies: Independent Data

Sometimes you need an entirely independent copy of an array, where modifications will not affect the original.

### The `.copy()` Method

The `.copy()` method creates a new `NDArray` with its own, separate data buffer.

```javascript
const original = ndarray.arange(5);
// => array([0, 1, 2, 3, 4], dtype=float64)

const independent_copy = original.copy();

// Modify the copy
independent_copy.set(99, 0);

// The original array remains unchanged
console.log(original.get(0));       // => 0
console.log(independent_copy.get(0)); // => 99
```

**Note:** The `.pick()` method (for advanced indexing) also returns a new copy, not a view.

## Contiguity: Optimizing for Performance

An array's "contiguity" refers to whether its elements are stored sequentially in memory in a standard, row-major (C-style) order. This property is crucial for performance, especially when interfacing with WebAssembly or other optimized libraries.

### `.isContiguous`

You can check if an `NDArray` is contiguous using the `.isContiguous` boolean property.

-   Newly created arrays (`ndarray.array()`, `ndarray.zeros()`, etc.) are always contiguous.
-   Simple slices that maintain row-major order are often contiguous.
-   `transpose()` and many complex slices create non-contiguous views.

```javascript
const m = ndarray.arange(9).reshape(3, 3);
console.log(m.isContiguous); // => true

const t = m.transpose();
console.log(t.isContiguous); // => false (strides are swapped)

const sliced = m.slice(null, [null, null, -1]); // Reversed columns
console.log(sliced.isContiguous); // => false (stride is negative)
```

### `.asContiguous()`

When a function requires a contiguous array, you can use `.asContiguous()`. This method intelligently:
-   Returns the array itself if it's already contiguous (no copy needed).
-   Returns a new, contiguous copy of the array if it's non-contiguous (equivalent to `.copy()`).

```javascript
const non_contiguous_view = ndarray.arange(6).reshape(2, 3).transpose();
console.log(non_contiguous_view.isContiguous); // => false

// Ensure we have a contiguous version
const contiguous_array = non_contiguous_view.asContiguous();
console.log(contiguous_array.isContiguous); // => true
console.log(contiguous_array.toString());
// => array([[0, 3],
//           [1, 4],
//           [2, 5]], dtype=float64)
```
Using `.asContiguous()` is a best practice before passing arrays to external libraries or high-performance routines that expect a specific memory layout.
