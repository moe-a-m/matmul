# Matrix Multiplication Benchmark Suite

High-performance matrix multiplication implementations in Rust with comprehensive benchmarking and optimization analysis.

[![Rust](https://img.shields.io/badge/rust-2024-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Features

- **Multiple Implementations**: Naive, optimized, tiled, vectorized, parallel, BLAS, and GPU
- **Comprehensive Benchmarking**: Automated performance testing with multiple configurations
- **Correctness Validation**: Hash-based verification and error analysis
- **Performance Metrics**: GFLOPS, latency, bandwidth, speedup analysis
- **Reproducibility**: JSON-based workload specs and result tracking

## Implementations

| Implementation | Description | Feature Flag |
|---------------|-------------|--------------|
| **Naive** | Basic triple-loop implementation | (default) |
| **Optimized** | Custom C kernel with compiler optimizations | (default) |
| **Tiled** | Cache-friendly blocked algorithm | (default) |
| **Vectorized** | SIMD-optimized computation | `vectorized` |
| **Parallel** | Multi-threaded with Rayon | `parallel` |
| **BLAS** | BLIS library integration | `blis` |
| **GPU** | Custom XLA backend with OpenMP | `gpu` |

## Quick Start

### Prerequisites

- Rust (2021 edition or later)
- C compiler (gcc/clang)
- BLIS library (optional, for `blis` feature)
- OpenMP (optional, for `gpu` feature)

### Setup

```bash
# Update Cargo.toml with your details
# Edit: authors, repository fields

./setup.sh
```

Or manually:

```bash
# Install BLIS (macOS)
brew install blis

# Install BLIS (Ubuntu/Debian)
sudo apt install libblis-dev libblis-openmp-dev

# Build project
cargo build --release
```

### Basic Usage

```bash
# Run with default configuration (1024x1024x1024)
cargo run --release

# Run with custom workload
cargo run --release -- --workload tests/test_workload.json

# Save results to file
cargo run --release -- --output results/benchmark.json

# Configure benchmark runs
cargo run --release -- --warmup-runs 5 --bench-runs 10
```

### Feature Flags

```bash
# Enable specific features
cargo run --release --features blis
cargo run --release --features parallel,vectorized
cargo run --release --features blis,parallel,vectorized,gpu

# Build with all features
cargo build --release --features blis,parallel,vectorized,gpu
```

## Benchmarking

### Comprehensive Benchmark

Tests all feature combinations with various RUSTFLAGS:

```bash
./benchmark_all.sh
```

Results stored in `results/` (created automatically) with metadata:
- `bench_*.json` - Performance metrics
- `build_info_*.json` - Build configuration

### Fast Benchmark

Finds and runs the fastest configuration:

```bash
./benchmark_fastest.sh
```

Outputs:
- `results/fastest_run_*.json` - Best performance results
- `results/fastest_config_*.json` - Optimal configuration

### Analysis

```bash
# Analyze all benchmark results
python3 analyze_results.py

# Analyze fastest configuration
python3 analyze_fastest.py

# Reproduce fastest benchmark
python3 analyze_fastest.py reproduce
```

## Workload Specification

JSON format for custom matrix sizes:

```json
{
  "workload_id": "matmul_2048_fp32",
  "type": "matrix_multiplication",
  "parameters": {
    "shapes": {
      "M": 2048,
      "N": 2048,
      "K": 2048
    },
    "precision": "fp32",
    "layout": "row_major"
  },
  "constraints": {
    "max_time_ms": 10000,
    "max_memory_mb": 1024
  },
  "validation": {
    "seed": 42,
    "tolerance": 1e-5
  }
}
```

## Output Format

```json
{
  "latency_ms": 45.23,
  "throughput_gflops": 47.56,
  "ops_per_second": 4.756e10,
  "output_hash": "a1b2c3d4...",
  "max_error": 1.23e-6,
  "correctness": true,
  "workload_info": {
    "matrix_size": [1024, 1024, 1024],
    "total_ops": 2147483648,
    "memory_usage_mb": 0.012
  },
  "performance_analysis": {
    "speedup_vs_naive": 125.4,
    "blas_speedup": 98.2,
    "tiled_speedup": 45.6,
    "vectorized_speedup": 67.8,
    "parallel_speedup": 89.3,
    "gpu_speedup": 156.7,
    "memory_bandwidth_gbps": 23.45,
    "compute_efficiency": 0.476
  }
}
```

## Docker Support

```bash
# Build with default features
docker build -t matmul .

# Build with specific features
docker build --build-arg FEATURES="blis,parallel" -t matmul .

# Run benchmark
docker run matmul "cargo run --release"

# Run with custom workload
docker run -v $(pwd)/tests:/app/tests matmul \
  "cargo run --release -- --workload /app/tests/test_workload.json"
```

## Optimization Flags

Recommended RUSTFLAGS for maximum performance:

```bash
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C codegen-units=1" \
  cargo build --release --features blis,parallel,vectorized,gpu
```

## Architecture

```
matmul/
├── src/
│   ├── main.rs              # Entry point and orchestration
│   ├── bench.rs             # Benchmarking infrastructure
│   ├── validate.rs          # Correctness validation
│   └── matmul/
│       ├── mod.rs           # Module exports
│       ├── naive.rs         # Basic implementation
│       ├── optimized.rs     # C kernel wrapper
│       ├── tiled.rs         # Cache-optimized
│       ├── vectorized.rs    # SIMD implementation
│       ├── parallel.rs      # Multi-threaded
│       ├── blis.rs          # BLAS integration
│       └── gpu.rs           # GPU backend
├── kernels/
│   ├── matmul_kernel.c      # Optimized C kernel
│   ├── matmul_kernel.h      # Kernel header
│   └── tt_xla_backend.c     # GPU/XLA backend
├── tests/
│   ├── test_workload.json   # Standard test case
│   └── small_test.json      # Small matrix test
├── build.rs                 # Build script
├── benchmark_all.sh         # Comprehensive benchmark
├── benchmark_fastest.sh     # Fast configuration finder
├── analyze_results.py       # Result analysis
├── analyze_fastest.py       # Fastest config analysis
├── setup.sh                 # Environment setup
├── rustfmt.toml             # Rust formatting config
└── results/                 # Benchmark outputs (auto-created)
```

## Performance Tips

1. **Enable native CPU features**: Use `-C target-cpu=native`
2. **Combine features**: `blis,parallel,vectorized,gpu` often performs best
3. **Tune matrix size**: Performance varies with cache sizes
4. **Use BLIS**: Typically fastest for CPU-only workloads
5. **Profile first**: Run `benchmark_fastest.sh` to find optimal config

## Troubleshooting

### BLIS not found

```bash
# macOS
brew install blis
export LIBRARY_PATH=/opt/homebrew/lib:$LIBRARY_PATH

# Linux
sudo apt install libblis-dev
```

### GPU feature build fails

Ensure OpenMP is installed:

```bash
# macOS
brew install libomp

# Linux
sudo apt install libomp-dev
```

### Cross-compilation to RISC-V

The build script automatically detects RISC-V targets and applies appropriate flags (`-march=rv64gcv`).

## License

Dual-licensed under MIT or Apache 2.0 at your option.

## Contributing

1. Add new implementations in `src/matmul/`
2. Update `mod.rs` to export new modules
3. Add benchmarking in `main.rs`
4. Run `benchmark_all.sh` to validate
5. Update this README with new features
