#!/bin/bash

# Build script for TensorTorrent N300s optimization

echo "Building matrix multiplication with TensorTorrent N300s GPU support..."

# Set TensorTorrent environment
export TT_METAL_HOME=/opt/tenstorrent
export LD_LIBRARY_PATH=$TT_METAL_HOME/lib:$LD_LIBRARY_PATH

# Build with GPU feature enabled
cargo build --release --features gpu

echo "Build complete. Run with:"
echo "cargo run --release --features gpu -- --workload tests/test_workload.json"