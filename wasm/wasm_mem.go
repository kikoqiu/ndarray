//go:build wasm

package main

import (
	"sync"
	"unsafe"
)

/**
 * Memory Pinning Registry
 * We must keep a reference to every allocated block in Go memory,
 * otherwise the Go Garbage Collector will reclaim the memory even
 * if JavaScript still holds the pointer.
 */
var (
	// registry stores the slice references to prevent GC.
	// Key is the uintptr (the raw address).
	registry = make(map[uintptr][]byte)
	// lock ensures thread-safety (though WASM is currently mostly single-threaded).
	lock sync.Mutex
)

func mem_register(buf []byte) unsafe.Pointer {

	// Get the raw pointer
	ptr := unsafe.Pointer(&buf[0])
	uptr := uintptr(ptr)

	// Register the slice to prevent GC collection
	lock.Lock()
	registry[uptr] = buf
	lock.Unlock()

	return ptr
}

// malloc allocates memory in the Go heap and pins it in a global map.
//
//go:wasmexport malloc
func malloc(size int32) unsafe.Pointer {
	if size <= 0 {
		return nil
	}

	// 1. Allocate the byte slice
	buf := make([]byte, int(size))

	return mem_register(buf)
}

// free unregisters the memory block, allowing the Go GC to collect it.
//
//go:wasmexport free
func free(ptr unsafe.Pointer) {
	if ptr == nil {
		return
	}

	uptr := uintptr(ptr)

	// 4. Remove from registry to unpin the memory
	lock.Lock()
	delete(registry, uptr)
	lock.Unlock()
}

// --- Internal Zero-Copy Helpers (Used by other Go files) ---

// ptrToF64Slice converts a raw pointer back to a Go slice for computation.
func ptrToF64Slice(ptr unsafe.Pointer, size int32) []float64 {
	if ptr == nil || size <= 0 {
		return nil
	}
	// unsafe.Slice is the modern (Go 1.17+) way to create a slice from a pointer.
	return unsafe.Slice((*float64)(ptr), int(size))
}

func ptrToF32Slice(ptr unsafe.Pointer, size int32) []float32 {
	if ptr == nil || size <= 0 {
		return nil
	}
	return unsafe.Slice((*float32)(ptr), int(size))
}

func ptrToI32Slice(ptr unsafe.Pointer, size int32) []int32 {
	if ptr == nil || size <= 0 {
		return nil
	}
	return unsafe.Slice((*int32)(ptr), int(size))
}

// --- Essential Byte Ops ---

//go:wasmexport memcpy
func memcpy(dst, src unsafe.Pointer, size int32) {
	d := unsafe.Slice((*byte)(dst), int(size))
	s := unsafe.Slice((*byte)(src), int(size))
	copy(d, s)
}

//go:wasmexport memset
func memset(ptr unsafe.Pointer, value int32, size int32) {
	b := unsafe.Slice((*byte)(ptr), int(size))
	for i := range b {
		b[i] = byte(value)
	}
}
