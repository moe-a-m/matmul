#!/usr/bin/env bash
# Benchmark every backend across feature sets and thread counts.
# Each run writes one JSON report (schema v2) that records the CPU, build flags,
# git commit, thread settings and a correctness check of every backend.
#
#   ./benchmark_all.sh                          # 1024^3, 1 thread and all physical cores
#   WORKLOAD=tests/test_workload.json THREADS="1 2 4" ./benchmark_all.sh
set -euo pipefail

physical_cores() {
    if command -v lscpu >/dev/null 2>&1; then
        lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l
    else
        sysctl -n hw.physicalcpu 2>/dev/null || echo 1
    fi
}

OUT_DIR="${OUT_DIR:-results/$(date +%Y-%m-%d)}"
WORKLOAD="${WORKLOAD:-}"
THREADS="${THREADS:-1 $(physical_cores)}"
FEATURE_SETS=("" "avx2" "blis" "avx2,blis")
export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"

mkdir -p "$OUT_DIR"
for features in "${FEATURE_SETS[@]}"; do
    name="${features:-default}"
    name="${name//,/_}"
    if ! cargo build --release --quiet ${features:+--features "$features"}; then
        echo "skip: build failed for features '${features}'" >&2
        continue
    fi
    for t in $THREADS; do
        out="$OUT_DIR/bench_${name}_t${t}.json"
        OMP_NUM_THREADS=$t BLIS_NUM_THREADS=$t RAYON_NUM_THREADS=$t \
            cargo run --release --quiet ${features:+--features "$features"} -- \
            ${WORKLOAD:+--workload "$WORKLOAD"} --output "$out"
        echo "wrote $out"
    done
done
echo "Done. Summarise with: python3 analyze_results.py $OUT_DIR"
