#!/bin/bash
# Optimized build for Apple M4 (arm64) benchmarking (Rust-native).
set -e

OS=$(uname -s)
ARCH=$(uname -m)

if [ "$OS" != "Darwin" ] || [ "$ARCH" != "arm64" ]; then
    echo "This script targets macOS arm64 (Apple Silicon). Detected: $OS $ARCH"
    exit 1
fi

echo "Building OcMesher (Rust-native, optimized)"
cargo build --release --manifest-path ocmesher-rust/Cargo.toml

echo "✓ Optimized Rust-native build complete."
