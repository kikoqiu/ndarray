//go:build wasm

package main

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/draw"
	"image/jpeg"
	"image/png"
	"unsafe"

	_ "image/gif" // Register GIF decoder

	_ "golang.org/x/image/webp" // Register WebP decoder
)

// decode_image decodes a binary image file into a raw RGBA pixel buffer.
// It accepts a pointer to the image data in WASM memory.
// On success, it returns a pointer to a 16-byte structure in WASM memory:
//   - [0:4]  => pointer to the raw RGBA pixel data
//   - [4:8]  => size of the pixel data buffer (width * height * 4)
//   - [8:12] => width of the image
//   - [12:16]=> height of the image
//
// On failure (e.g., invalid format), it returns nil (0).
// The caller is responsible for freeing both the result structure and the pixel data pointer.
//
//go:wasmexport decode_image
func decode_image(ptr unsafe.Pointer, size int32) unsafe.Pointer {
	if ptr == nil || size <= 0 {
		return nil
	}

	// 1. Create a Go slice from the WASM memory without copying.
	inputBuf := unsafe.Slice((*byte)(ptr), size)

	// 2. Decode the image. The registered decoders will handle format detection.
	img, _, err := image.Decode(bytes.NewReader(inputBuf))
	if err != nil {
		return nil // Indicates decoding failure
	}

	// 3. Convert the image to RGBA to ensure a consistent 4-channel memory layout.
	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y
	rgba := image.NewRGBA(bounds)
	draw.Draw(rgba, rgba.Bounds(), img, bounds.Min, draw.Src)
	pixelData := rgba.Pix

	// Allocate memory for the result structure and write the values.
	resultPtr := malloc(16)
	if resultPtr == nil {
		return nil
	}

	// 4. Allocate memory in WASM for the pixel data and copy it.
	pixelDataSize := len(pixelData)
	pixelDataPtr := mem_register(pixelData)

	resultSlice := unsafe.Slice((*byte)(resultPtr), 16)
	binary.LittleEndian.PutUint32(resultSlice[0:4], uint32(uintptr(pixelDataPtr)))
	binary.LittleEndian.PutUint32(resultSlice[4:8], uint32(pixelDataSize))
	binary.LittleEndian.PutUint32(resultSlice[8:12], uint32(width))
	binary.LittleEndian.PutUint32(resultSlice[12:16], uint32(height))

	return resultPtr
}

// encode_image encodes a raw RGBA pixel buffer into a specified image format (PNG, JPEG, or WebP).
// It accepts a pointer to the pixel data, dimensions, quality (for JPEG), and format string.
// On success, it returns a pointer to an 8-byte structure in WASM memory:
//   - [0:4] => pointer to the encoded image file data (e.g., a full PNG file)
//   - [4:8] => size of the encoded data buffer
//
// On failure, it returns nil (0).
// The caller is responsible for freeing both the result structure and the encoded data pointer.
//
//go:wasmexport encode_image
func encode_image(ptr unsafe.Pointer, width, height, channels, quality int32, formatPtr unsafe.Pointer, formatSize int32) unsafe.Pointer {
	if ptr == nil || width <= 0 || height <= 0 || formatPtr == nil || formatSize <= 0 {
		return nil
	}
	if channels != 1 && channels != 3 && channels != 4 {
		return nil // Only support 1, 3, or 4 channels
	}

	// 1. Determine buffer size and get pixel data from WASM memory.
	pixelDataSize := width * height * channels
	pixelData := unsafe.Slice((*byte)(ptr), pixelDataSize)

	// 2. Create an image.Image based on the number of channels.
	var img image.Image
	switch channels {
	case 1:
		img = &image.Gray{
			Pix:    pixelData,
			Stride: int(width),
			Rect:   image.Rect(0, 0, int(width), int(height)),
		}
	case 3:
		// Convert 3-channel RGB to 4-channel RGBA for encoding.
		rgba := image.NewRGBA(image.Rect(0, 0, int(width), int(height)))
		for y := 0; y < int(height); y++ {
			for x := 0; x < int(width); x++ {
				srcIdx := (y*int(width) + x) * 3
				dstIdx := (y*int(width) + x) * 4
				rgba.Pix[dstIdx+0] = pixelData[srcIdx+0] // R
				rgba.Pix[dstIdx+1] = pixelData[srcIdx+1] // G
				rgba.Pix[dstIdx+2] = pixelData[srcIdx+2] // B
				rgba.Pix[dstIdx+3] = 255                 // A
			}
		}
		img = rgba
	case 4:
		img = &image.RGBA{
			Pix:    pixelData,
			Stride: int(width) * 4,
			Rect:   image.Rect(0, 0, int(width), int(height)),
		}
	}

	// 3. Encode the image into a buffer based on the specified format.
	var buf bytes.Buffer
	format := string(unsafe.Slice((*byte)(formatPtr), formatSize))

	var err error
	switch format {
	case "png":
		err = png.Encode(&buf, img)
	case "jpeg":
		err = jpeg.Encode(&buf, img, &jpeg.Options{Quality: int(quality)})
	// NOTE: WebP encoding is not supported as the standard Go WebP library (golang.org/x/image/webp)
	// only provides a decoder. Including a C-based encoder library would complicate the build process.
	default:
		return nil // Unsupported format
	}

	if err != nil {
		return nil // Encoding failed
	}
	encodedData := buf.Bytes()

	// Allocate and write the result structure [ptr, size].
	resultPtr := malloc(8)
	if resultPtr == nil {
		return nil
	}

	// 4. Allocate memory for the encoded data and copy it over.
	encodedDataSize := len(encodedData)
	encodedDataPtr := mem_register(encodedData)

	resultSlice := unsafe.Slice((*byte)(resultPtr), 8)
	binary.LittleEndian.PutUint32(resultSlice[0:4], uint32(uintptr(encodedDataPtr)))
	binary.LittleEndian.PutUint32(resultSlice[4:8], uint32(encodedDataSize))

	return resultPtr
}
