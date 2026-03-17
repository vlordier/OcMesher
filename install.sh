#!/bin/bash
# Rust-native build script.
set -e

echo "Building OcMesher (Rust-native)"
cargo build --release --manifest-path ocmesher-rust/Cargo.toml
echo "Build complete."
