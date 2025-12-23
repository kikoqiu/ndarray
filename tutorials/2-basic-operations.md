# Tutorial 2: Basic Operations, Reductions, and Broadcasting

This tutorial covers common array operations, including element-wise arithmetic, reductions along axes, and the powerful concept of broadcasting.

## Element-wise Operations

You can perform standard arithmetic operations directly on `NDArray` objects. These operations are applied element by element. If the arrays have different shapes, broadcasting rules apply.

```javascript
import * as ndarray from 'ndarray';

const a = ndarray.array([[1, 2], [3, 4]]);
const b = ndarray.array([[10, 20], [30, 40]]);

// Element-wise addition
const sum = a.add(b);
// => array([[11, 22],
//           [33, 44]], dtype=float64)

// Element-wise multiplication
const product = a.mul(b);
// => array([[10, 40],
//           [90, 160]], dtype=float64)

// You can also use in-place operations (e.g., iadd, imul)
a.iadd(b);
// 'a' is now array([[11, 22],
//                   [33, 44]], dtype=float64)
```

You can apply any custom function to each element using the `map()` method. This is highly optimized by the JIT compiler.

```javascript
const a = ndarray.array([1, 4, 9, 16]);

// Calculate the square root of each element
const sqrt_a = a.map('Math.sqrt(${val})');
// => array([1, 2, 3, 4], dtype=float64)

// Apply a custom lambda function
const cubed = a.map(val => val * val * val);
// => array([1, 64, 729, 4096], dtype=float64)
```

## Broadcasting

Broadcasting is a mechanism that allows `NDArray` to perform operations on arrays of different shapes. It implicitly "stretches" the smaller array to match the larger one without actually copying data, making operations very efficient.

The rules are:
1.  If dimensions differ, the smaller array is padded with ones on its left side.
2.  Arrays are compatible if, for each dimension, the sizes are equal, or one of them is 1.

```javascript
const m = ndarray.arange(4).reshape(2, 2);
// => array([[0, 1],
//           [2, 3]], dtype=float64)

const v = ndarray.array([10, 20]); // A 1D array

// 'v' (shape [2]) is broadcast across the rows of 'm' (shape [2, 2]).
// It effectively becomes [[10, 20], [10, 20]] for the operation.
const result = m.add(v);
// => array([[10, 21],
//           [12, 23]], dtype=float64)

// Scalar broadcasting also works:
const scalar_mul = m.mul(5);
// => array([[0, 5],
//           [10, 15]], dtype=float64)
```

## Reductions

Reduction operations condense the values of an array, usually by applying an aggregation function (like sum, mean, min, max). You can perform reductions over the entire array or along a specific `axis`.

```javascript
const m = ndarray.array([[1, 2, 3], [4, 5, 6]]);

// Sum of all elements in the array
const totalSum = m.sum(); // => 21

// Mean of all elements
const overallMean = m.mean(); // => 3.5

// Sum along columns (axis 0)
const sumAlongAxis0 = m.sum(0);
// => array([5, 7, 9], dtype=float64)
// (1+4=5, 2+5=7, 3+6=9)

// Max along rows (axis 1)
const maxAlongAxis1 = m.max(1);
// => array([3, 6], dtype=float64)
// (max of [1,2,3]=3, max of [4,5,6]=6)

// Calculate standard deviation along a specific axis
const stdAlongAxis0 = m.std(0);
```
Many reduction functions are available, including `sum`, `mean`, `max`, `min`, `var` (variance), and `std` (standard deviation).
