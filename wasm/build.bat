@echo off
setlocal
echo ============================================
echo NDArray Go WASM Build Script
echo ============================================

cd  /d %~dp0
:: 1. Set environment variables
set GOOS=js
set GOARCH=wasm

:: 2. Compile
echo Compiling WASM...
go build -o ndarray_plugin.wasm -ldflags="-s -w" .

if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    exit /b %ERRORLEVEL%
)

:: 3. Extract wasm_exec.js
echo Extracting wasm_exec.js...
for /f "tokens=*" %%i in ('go env GOROOT') do set GOROOT=%%i
copy "%GOROOT%\lib\wasm\wasm_exec.js" . /y

:: 4. Copy artifacts to dist directory
echo Copying artifacts to dist...
if not exist "..\dist" mkdir "..\dist"
copy "ndarray_plugin.wasm" "..\dist\ndarray_plugin.wasm" /y
copy "wasm_exec.js" "..\dist\wasm_exec.js" /y

echo.
echo Build Successful!
echo Generated: ndarray_plugin.wasm and wasm_exec.js
echo Copied to: ..\dist
