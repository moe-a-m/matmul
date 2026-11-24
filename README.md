# Matrix Multiplication Benchmark Suite

Single-precision matrix multiplication (SGEMM, `C = A × B`, row-major) on the CPU: a naive reference, Rust loop variants, a packed C kernel, a **BLIS-style AVX2/FMA microkernel in C with OpenMP**, and the BLIS library itself as an external yardstick. Every backend is timed and **checked against the naive reference** in the same run, and every result file records the CPU, build flags, thread settings and git commit.

[![Rust](https://img.shields.io/badge/rust-2024-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Results

Measured 24 Nov 2026 on an **Intel Core i7-5500U** (Broadwell laptop, 2 cores / 4 threads, AVX2 + FMA), 1024 × 1024 × 1024 fp32, `RUSTFLAGS="-C target-cpu=native"`, GCC with `-O3 -march=native`, 3 warm-up + 10 timed runs, best run shown. Raw reports: [`results/2026-11-24_i7-5500U/`](results/2026-11-24_i7-5500U/).

| Backend | What it is | 1 thread (GFLOPS) | 2 threads (GFLOPS) | Correct |
|---|---|---:|---:|:---:|
| `naive` | Rust triple loop (reference) | 0.27–0.35 | 0.30–0.39 | reference |
| `tiled` | Rust, 64³ cache blocking | 1.18 | 1.17 | ✓ |
| `vectorized` | Rust, blocking + 4-wide accumulators | 1.80 | 1.80 | ✓ |
| `parallel` | Rust, Rayon row-parallel triple loop | 0.42 | 0.71 | ✓ |
| `c_packed` | C, tiled with A/B packing | 4.56 | 4.55 | ✓ |
| **`avx2_blis`** | **C, BLIS-style packing + 6×16 AVX2/FMA microkernel, OpenMP** | **51.04** | **88.80** | ✓ |
| `blis` | BLIS library `cblas_sgemm` (Debian `libblis-openmp`) | 77.69 | 127.30 | ✓ |

**Summary:** `avx2_blis` reaches **88.8 GFLOPS on 2 cores, about 70% of BLIS** (127.3) on the same machine and over 200× the naive loop. Single-threaded it reaches about 66% of BLIS (51.0 vs 77.7). All outputs match the reference within a relative error of 1e-4 (observed ≈2e-6).

**Measurement notes:**
- Best results per backend are taken across the sweep in `results/2026-11-24_i7-5500U/` (`bench_<features>_t<threads>.json`).
- This laptop throttles under sustained load (sensors read 85–97 °C). The last configuration of the sweep (`bench_avx2_blis_t2.json`) ran hot and shows `avx2_blis` at single-thread speed. A re-run after cooling (`rerun_cooled_avx2_blis_t2.json`) gives 87.2 / 121.3 GFLOPS. Both files are kept.
- The reports were produced from the working tree before these changes were committed, so they record `git_commit: 0ee78f2-dirty`. Re-run `./benchmark_all.sh` after committing to get a clean commit id.
- *Inference:* BLIS's 77.7 GFLOPS single-threaded means turbo was active. At 3.0 GHz the theoretical fp32 peak is 96 GFLOPS per core (2 FMA units × 8 lanes × 2 FLOP), so `avx2_blis` is at roughly 53% and BLIS at roughly 81% of single-core peak.

## Corrections and history

An earlier version of this README and of the project description contained figures that were **never measured**, and some backends were mislabelled. Fixed in this version:

| Before | Now |
|---|---|
| README "Output Format" example showed `47.56` GFLOPS and `125.4×` with a placeholder hash. These were illustrative values, not results; the only result committed at the time (`results/metrics.json`, Jan 2026) was 3.18 GFLOPS | Real reports with hardware and build metadata (table above) |
| `gpu` feature / `gpu.rs` / `tt_xla_backend.c` ("Custom XLA backend") | It has always been an **x86 AVX2 CPU kernel**; renamed to feature `avx2`, `avx2_blis.rs`, `kernels/avx2_blis_kernel.c`. An experimental Tenstorrent N300s backend was added and removed on 13 Jan 2026; nothing here runs on a GPU or accelerator |
| `blis` backend was a blocked Rust loop and did not call BLIS | Now calls `cblas_sgemm` from the BLIS library |
| Headline throughput always measured the `optimized` C kernel; the AVX2 kernel's output was never checked | Every backend is timed and validated; the report names the fastest correct one |
| `optimized.rs` | Renamed `c_packed.rs` (describes what it does) |
| RISC-V path used invalid inline asm (`read_cycle()`) and pre-1.0 RVV intrinsic names | Fixed (`rdcycle`/`rdinstret`; an `RVV()` macro picks the intrinsic spelling per compiler). It compiles for `rv64gcv` (checked with Clang 15) but **has not been run or measured on RISC-V hardware** |
| `parallel` and `vectorized` Cargo features did nothing | Removed; those backends are always built |

## Quick start

```bash
# Dependencies (BLIS is only needed for the `blis` feature)
sudo apt install libblis-dev libblis-openmp-dev     # Debian/Ubuntu
brew install blis libomp                             # macOS (see notes below)

# Correctness tests: every backend vs. the naive reference on odd shapes
cargo test --release --features avx2,blis

# Benchmark 1024^3 with every backend, on 2 threads
RUSTFLAGS="-C target-cpu=native" OMP_NUM_THREADS=2 BLIS_NUM_THREADS=2 RAYON_NUM_THREADS=2 \
  cargo run --release --features avx2,blis -- --output results/my_run.json
```

The harness prints a table to stderr and the JSON report to stdout (or to `--output`). Options: `--workload <file.json>`, `--warmup-runs`, `--bench-runs`, `--tolerance`.

### Cargo features

| Feature | Adds | Requirements |
|---|---|---|
| (none) | `naive`, `tiled`, `vectorized`, `parallel`, `c_packed` | Rust, a C compiler |
| `avx2` | `avx2_blis` | x86-64 with AVX2/FMA, OpenMP (GCC: built in; macOS: Homebrew `libomp`, untested) |
| `blis` | `blis` (external reference) | `libblis` with the CBLAS interface |

Set `OMP_NUM_THREADS` (for `avx2_blis` and BLIS's OpenMP build), `BLIS_NUM_THREADS` and `RAYON_NUM_THREADS` explicitly; they are recorded in every report. Use the number of **physical** cores for the fairest comparison.

## Benchmarking scripts

```bash
./benchmark_all.sh                     # feature sets x {1 thread, all physical cores} -> results/<date>/
WORKLOAD=tests/test_workload.json THREADS="1 2 4" OUT_DIR=results/custom ./benchmark_all.sh
./benchmark_fastest.sh                 # all available features, 20 timed runs -> results/fastest_<time>.json

python3 analyze_results.py results/2026-11-24_i7-5500U   # tables + best correct result per backend
python3 analyze_fastest.py                                # latest fastest_*.json
python3 analyze_fastest.py reproduce                      # rebuild with the same settings and compare
```

## Report format (schema v2)

Each report contains:
- `workload`: `m`, `n`, `k`, `flop` (= 2·m·n·k), precision, input bytes
- `system`: CPU model, logical CPUs, OS, arch, `OMP_NUM_THREADS`, `BLIS_NUM_THREADS`, Rayon threads
- `build`: features, rustc version, `RUSTFLAGS`, target, git commit (`-dirty` if uncommitted changes), profile
- `settings`: warm-up runs, timed runs, tolerance
- `reference` and `backends[]`: `best_ms`, `median_ms`, `gflops_best`, `gflops_median`, `speedup_vs_naive`, `max_abs_error`, `rel_error` (= max|c − ref| / max|ref|), `correct`
- `fastest_correct_backend`, `latency_ms`, `throughput_gflops`, `output_hash` (SHA-256 of that backend's output)

See any file in `results/2026-11-24_i7-5500U/` for a full example.

## How `avx2_blis` works

Following Goto & van de Geijn ("Anatomy of High-Performance Matrix Multiplication", 2008) and the BLIS framework:

1. Split N into NC = 256 column panels, distributed across OpenMP threads.
2. For each KC = 256 slice of K, pack the B panel into `[jr][k][NR]` order in a 64-byte-aligned, thread-local buffer.
3. Walk MC = 192 rows of A and NR = 16 columns of the packed panel.
4. A **6 × 16 microkernel** keeps 12 AVX2 accumulators (6 rows × 2 vectors of 8 floats) in registers and does a broadcast-FMA update per k.
5. Tiles smaller than 6 × 16 at the matrix edges use a scalar loop. That is why sizes like 1023³ run about 30% slower than 1024³; vectorising the edge path is the next optimisation.

## RISC-V

`kernels/matmul_kernel.c` (`c_packed`) has an RVV path that `build.rs` enables with `-march=rv64gcv` for RISC-V targets:

```bash
cargo build --release --target riscv64gc-unknown-linux-gnu   # needs a RISC-V cross toolchain and sysroot
```

It compiles to vector code (`vsetvli`, `vle32.v`, `vfmacc.vf`), but it has **not been executed or benchmarked on RISC-V hardware**, so no RISC-V performance is claimed.

## Docker

```bash
docker build -t matmul .                                  # features avx2,blis by default
docker build --build-arg FEATURES="blis" -t matmul .
docker run --rm -e OMP_NUM_THREADS=2 -e BLIS_NUM_THREADS=2 matmul
```

`-C target-cpu=native` tunes the binary for the machine that builds the image.

## Layout

```
src/
  main.rs            harness: timing, validation, report
  bench.rs           report structures (schema v2)
  validate.rs        error metrics, output hash
  matmul/
    mod.rs           backend registry + correctness tests
    naive.rs tiled.rs vectorized.rs parallel.rs
    c_packed.rs      FFI to kernels/matmul_kernel.c
    avx2_blis.rs     FFI to kernels/avx2_blis_kernel.c   (feature avx2)
    blis.rs          FFI to cblas_sgemm                  (feature blis)
kernels/
  matmul_kernel.c/.h tiled C kernel with packing; RVV path on RISC-V
  avx2_blis_kernel.c BLIS-style AVX2/FMA kernel with OpenMP
tests/*.json         workload definitions
results/             benchmark reports
```

## Next steps

- Vectorise the edge tiles so odd sizes run as fast as multiples of 6 × 16.
- Pack A as well as B, and tune MC/KC to this cache hierarchy, to close the single-thread gap to BLIS.
- Add a peak-FMA microbenchmark and a roofline plot so results are reported as % of measured peak.
- Measure the RVV path on real RISC-V hardware.

## License

Dual-licensed under MIT or Apache 2.0 at your option.
