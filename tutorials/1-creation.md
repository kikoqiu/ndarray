# Tutorial 1: Array Creation and Factory Functions

This tutorial explores the fundamental ways to create `NDArray` objects, from simple JavaScript arrays to complex random distributions.

## Creating from Existing Data

The most common way to create an array is from a standard JavaScript `Array`.

### `ndarray.array()`
This function intelligently converts a nested JavaScript array into a multi-dimensional `NDArray`. It will infer the shape and data type.

```javascript
// Import the library
import * as ndarray from 'ndarray';

// Create a 2D array (a 2x3 matrix)
const a = ndarray.array([[1, 2, 3], [4, 5, 6]]);

console.log(a.shape);
// => Int32Array(2) [ 2, 3 ]

console.log(a.toString());
// => array([[1, 2, 3],
//           [4, 5, 6]], dtype=float64)

// You can also specify the data type
const b = ndarray.array([1, 2, 3], 'int32');
console.log(b.dtype);
// => int32
```

## Creating Arrays of a Specific Shape

Several "factory" functions exist to create new arrays with a given shape, often filled with a default value.

### `ndarray.zeros()` and `ndarray.ones()`
These create an array of the specified shape with all elements initialized to 0 or 1.

```javascript
// A 3x2 matrix of all zeros
const z = ndarray.zeros([3, 2]);

// A 2x2x2 cube of all ones
const o = ndarray.ones([2, 2, 2]);
```

### `ndarray.full()`
This creates an array filled with any scalar value you provide.

```javascript
// A 2x4 matrix filled with the number 42
const f = ndarray.full([2, 4], 42);
```

### `ndarray.eye()`
This creates a 2D identity matrix (1s on the diagonal, 0s elsewhere).

```javascript
const i = ndarray.eye(3);
// => array([[1, 0, 0],
//           [0, 1, 0],
//           [0, 0, 1]], dtype=float64)
```

## Creating Arrays from a Numerical Range

### `ndarray.arange()`
Creates a 1D array with values in a defined range with a specific step.

```javascript
// An array of values from 0 up to (but not including) 10
const r1 = ndarray.arange(10);
// => array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float64)

// An array from 2 to 8, with a step of 2
const r2 = ndarray.arange(2, 9, 2);
// => array([2, 4, 6, 8], dtype=float64)
```

### `ndarray.linspace()`
Creates a 1D array with a specific number of points spaced evenly between two values.

```javascript
// 5 points linearly spaced between 0 and 100 (inclusive)
const l = ndarray.linspace(0, 100, 5);
// => array([0, 25, 50, 75, 100], dtype=float64)
```

## Creating Random Arrays

The `ndarray.random` module provides functions to create arrays with random values from various statistical distributions.

### Uniform Distribution
The `random()` function creates an array with values from a uniform distribution.

```javascript
// A 2x2 array of random numbers from a uniform distribution between 0 and 1
const u = ndarray.random.random([2, 2]);

// A 3x3 array of random numbers from a uniform distribution between -5 and 5
const u2 = ndarray.random.random([3, 3], -5, 5);
```

### Normal (Gaussian) Distribution
The `normal()` function samples from a normal distribution.

```javascript
// A 2x2 array of random numbers from a standard normal distribution (mean 0, std 1)
const n = ndarray.random.normal([2, 2]);

// A 3x3 array from a normal distribution with mean 100 and standard deviation 15
const n2 = ndarray.random.normal([3, 3], 100, 15);
```
