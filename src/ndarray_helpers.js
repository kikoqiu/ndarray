// --- factory functions ---
import { NDArray, DTYPE_MAP } from './ndarray_core.js';
import { array } from './ndarray_factory.js'


/**
 * Dot Product
 * @param {Array|TypedArray} l
 * @param {Array|TypedArray} r
 * @return {NDArray}
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
 * @return {NDArray}
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