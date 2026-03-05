//go:build !wasm

package main

import (
	"fmt"
)

func ConsoleLog(a ...any) (n int, err error) {
	return fmt.Println(a...)
}
