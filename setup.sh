#!/bin/bash

set -e

echo "Setting up matmul project requirements..."

# Install system dependencies
if command -v apt &> /dev/null; then
    echo "Installing BLIS dependencies..."
    sudo apt update
    sudo apt install -y libblis-dev libblis-openmp-dev
elif command -v brew &> /dev/null; then
    echo "Installing BLIS via Homebrew..."
    brew install blis libomp
else
    echo "Package manager not supported. Please install BLIS manually."
    exit 1
fi

# Install Rust if not present
if ! command -v rustc &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source ~/.cargo/env
fi

# Build and test the project (avx2 needs x86-64; drop it on other CPUs)
echo "Building and testing..."
features="blis"
if [ "$(uname -m)" = "x86_64" ]; then features="avx2,blis"; fi
cargo test --release --features "$features"

echo "Setup complete!"