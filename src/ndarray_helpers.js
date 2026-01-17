import { NDArray } from './ndarray_core.js';
import { array } from './ndarray_factory.js'


// --- Arithmetic Operations ---

/**
 * Element-wise addition. Supports broadcasting.
 * @param {Array|TypedArray} l
 * @param {Array|TypedArray|number} r
 * @return {NDArray}
 * @see {NDArray#add}
 */
export function add(l, r){
    if(l instanceof Array){
        l = array(l);
    }
    if(r instanceof Array){
        r = array(r);
    }
    return l.add(r);
}

/**
 * Element-wise subtraction. Supports broadcasting.
 * @param {Array|TypedArray} l
 * @param {Array|TypedArray|number} r
 * @return {NDArray}
 * @see {NDArray#sub}
 */
export function sub(l, r){
    if(l instanceof Array){
        l = array(l);
    }
    if(r instanceof Array){
        r = array(r);
    }
    return l.sub(r);
}

/**
 * Element-wise multiplication. Supports broadcasting.
 * @param {Array|TypedArray} l
 * @param {Array|TypedArray|number} r
 * @return {NDArray}
 * @see {NDArray#mul}
 */
export function mul(l, r){
    if(l instanceof Array){
        l = array(l);
    }
    if(r instanceof Array){
        r = array(r);
    }
    return l.mul(r);
}

/**
 * Element-wise division. Supports broadcasting.
 * @param {Array|TypedArray} l
 * @param {Array|TypedArray|number} r
 * @return {NDArray}
 * @see {NDArray#div}
 */
export function div(l, r){
    if(l instanceof Array){
        l = array(l);
    }
    if(r instanceof Array){
        r = array(r);
    }
    return l.div(r);
}

/**
 * Element-wise exponentiation. Supports broadcasting.
 * @param {Array|TypedArray} l
 * @param {Array|TypedArray|number} r
 * @return {NDArray}
 * @see {NDArray#pow}
 */
export function pow(l, r){
    if(l instanceof Array){
        l = array(l);
    }
    if(r instanceof Array){
        r = array(r);
    }
    return l.pow(r);
}

/**
 * Element-wise modulo. Supports broadcasting.
 * @param {Array|TypedArray} l
 * @param {Array|TypedArray|number} r
 * @return {NDArray}
 * @see {NDArray#mod}
 */
export function mod(l, r){
    if(l instanceof Array){
        l = array(l);
    }
    if(r instanceof Array){
        r = array(r);
    }
    return l.mod(r);
}

// --- Bitwise Operations ---

/**
 * Bitwise AND.
 * @param {Array|TypedArray} l
 * @param {Array|TypedArray|number} r
 * @return {NDArray}
 * @see {NDArray#bitwise_and}
 */
export function bitwise_and(l, r){
    if(l instanceof Array){
        l = array(l, "uint32");
    }
    if(r instanceof Array){
        r = array(r, "uint32");
    }
    return l.bitwise_and(r);
}

/**
 * Bitwise OR.
 * @param {Array|TypedArray} l
 * @param {Array|TypedArray|number} r
 * @return {NDArray}
 * @see {NDArray#bitwise_or}
 */
export function bitwise_or(l, r){
    if(l instanceof Array){
        l = array(l, "uint32");
    }
    if(r instanceof Array){
        r = array(r, "uint32");
    }
    return l.bitwise_or(r);
}

/**
 * Bitwise XOR.
 * @param {Array|TypedArray} l
 * @param {Array|TypedArray|number} r
 * @return {NDArray}
 * @see {NDArray#bitwise_xor}
 */
export function bitwise_xor(l, r){
    if(l instanceof Array){
        l = array(l), "uint32";
    }
    if(r instanceof Array){
        r = array(r, "uint32");
    }
    return l.bitwise_xor(r);
}

/**
 * Bitwise left shift.
 * @param {Array|TypedArray} l
 * @param {Array|TypedArray|number} r
 * @return {NDArray}
 * @see {NDArray#bitwise_lshift}
 */
export function bitwise_lshift(l, r){
    if(l instanceof Array){
        l = array(l, "uint32");
    }
    if(r instanceof Array){
        r = array(r, "uint32");
    }
    return l.bitwise_lshift(r);
}

/**
 * Bitwise (logical) right shift.
 * @param {Array|TypedArray} l
 * @param {Array|TypedArray|number} r
 * @return {NDArray}
 * @see {NDArray#bitwise_rshift}
 */
export function bitwise_rshift(l, r){
    if(l instanceof Array){
        l = array(l, "uint32");
    }
    if(r instanceof Array){
        r = array(r, "uint32");
    }
    return l.bitwise_rshift(r);
}

/**
 * Bitwise NOT.
 * @param {Array|TypedArray} x
 * @return {NDArray}
 * @see {NDArray#bitwise_not}
 */
export function bitwise_not(x){
    if(x instanceof Array){
        x = array(x, "uint32");
    }
    return x.bitwise_not();
}

// --- Unary Operations ---

/**
 * Numeric negation.
 * @param {Array|TypedArray} x
 * @return {NDArray}
 * @see {NDArray#neg}
 */
export function neg(x){
    if(x instanceof Array){
        x = array(x);
    }
    return x.neg();
}

/**
 * Absolute value.
 * @param {Array|TypedArray} x
 * @return {NDArray}
 * @see {NDArray#abs}
 */
export function abs(x){
    if(x instanceof Array){
        x = array(x);
    }
    return x.abs();
}

/**
 * Exponential function (e^x).
 * @param {Array|TypedArray} x
 * @return {NDArray}
 * @see {NDArray#exp}
 */
export function exp(x){
    if(x instanceof Array){
        x = array(x);
    }
    return x.exp();
}

/**
 * Square root.
 * @param {Array|TypedArray} x
 * @return {NDArray}
 * @see {NDArray#sqrt}
 */
export function sqrt(x){
    if(x instanceof Array){
        x = array(x);
    }
    return x.sqrt();
}

/**
 * Sine.
 * @param {Array|TypedArray} x
 * @return {NDArray}
 * @see {NDArray#sin}
 */
export function sin(x){
    if(x instanceof Array){
        x = array(x);
    }
    return x.sin();
}

/**
 * Cosine.
 * @param {Array|TypedArray} x
 * @return {NDArray}
 * @see {NDArray#cos}
 */
export function cos(x){
    if(x instanceof Array){
        x = array(x);
    }
    return x.cos();
}

/**
 * Tangent.
 * @param {Array|TypedArray} x
 * @return {NDArray}
 * @see {NDArray#tan}
 */
export function tan(x){
    if(x instanceof Array){
        x = array(x);
    }
    return x.tan();
}

/**
 * Natural logarithm (base e).
 * @param {Array|TypedArray} x
 * @return {NDArray}
 * @see {NDArray#log}
 */
export function log(x){
    if(x instanceof Array){
        x = array(x);
    }
    return x.log();
}

/**
 * Ceiling (round up).
 * @param {Array|TypedArray} x
 * @return {NDArray}
 * @see {NDArray#ceil}
 */
export function ceil(x){
    if(x instanceof Array){
        x = array(x);
    }
    return x.ceil();
}

/**
 * Floor (round down).
 * @param {Array|TypedArray} x
 * @return {NDArray}
 * @see {NDArray#floor}
 */
export function floor(x){
    if(x instanceof Array){
        x = array(x);
    }
    return x.floor();
}

/**
 * Round to nearest integer.
 * @param {Array|TypedArray} x
 * @return {NDArray}
 * @see {NDArray#round}
 */
export function round(x){
    if(x instanceof Array){
        x = array(x);
    }
    return x.round();
}

/**
 * Dot Product
 * @param {Array|TypedArray} l
 * @param {Array|TypedArray} r
 * @return {NDArray}
 * @see {NDArray#dot}
 */
export function dot(l, r){
    if(l instanceof Array){
        l = array(l);
    }
    if(r instanceof Array){
        r = array(r);
    }
    return l.dot(r);
}

/**
 * Cross Product
 * @param {Array|TypedArray} l
 * @param {Array|TypedArray} r
 * @return {NDArray#cross}
 */
export function cross(l, r){
    if(l instanceof Array){
        l = array(l);
    }
    if(r instanceof Array){
        r = array(r);
    }
    return l.cross(r);
}