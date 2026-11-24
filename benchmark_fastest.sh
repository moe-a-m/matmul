#!/usr/bin/env bash
# Build with every optional backend that compiles here, run a longer benchmark on all
# physical cores, and save the report. The harness itself picks the fastest backend
# whose output matches the naive reference.
set -euo pipefail

physical_cores() {
    if command -v lscpu >/dev/null 2>&1; then
        lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l
    else
        sysctl -n hw.physicalcpu 2>/dev/null || echo 1
    fi
}

export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"
THREADS="${THREADS:-$(physical_cores)}"
mkdir -p results

chosen=""
for features in "avx2,blis" "avx2" "blis" ""; do
    if cargo build --release --quiet ${features:+--features "$features"} 2>/dev/null; then
        chosen="$features"
        break
    fi
done
echo "Features: '${chosen:-none}'  Threads: $THREADS"

out="results/fastest_$(date +%Y%m%d_%H%M%S).json"
OMP_NUM_THREADS=$THREADS BLIS_NUM_THREADS=$THREADS RAYON_NUM_THREADS=$THREADS \
    cargo run --release --quiet ${chosen:+--features "$chosen"} -- \
    --warmup-runs 5 --bench-runs 20 --output "$out"
echo "Saved $out. Inspect or reproduce with: python3 analyze_fastest.py [reproduce]"
