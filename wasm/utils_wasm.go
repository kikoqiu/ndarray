//go:build wasm

package main

import (
	"fmt"
	"syscall/js"
)

func ConsoleLog(a ...any) (n int, err error) {
	msg := fmt.Sprintln(a...)
	js.Global().Get("console").Call("log", msg)
	return len(msg), nil
}
