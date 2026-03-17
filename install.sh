#!/bin/bash
# Rust-native build script.
set -e

echo "Building OcMesher (Rust-native)"
cargo build --release -p ocmesher-core --manifest-path ocmesher-rust/Cargo.toml
echo "Build complete."
