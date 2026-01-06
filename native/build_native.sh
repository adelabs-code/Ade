#!/bin/bash

# Build script for native C library

set -e

echo "Building Ade native library..."

# Build the Rust cdylib
cargo build --release

# Copy the shared library to a standard location
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    cp target/release/libade_native.so libade_native.so
    echo "Built libade_native.so"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    cp target/release/libade_native.dylib libade_native.dylib
    echo "Built libade_native.dylib"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    cp target/release/ade_native.dll ade_native.dll
    echo "Built ade_native.dll"
fi

echo "Native library build complete!"








