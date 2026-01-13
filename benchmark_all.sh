#!/bin/bash

# Performance benchmark script for matmul with various Rust optimization flags
# Results stored in results/ folder for performance analysis

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

# Test configurations
FEATURES=("" "blis" "parallel" "vectorized" "gpu" "blis,parallel" "blis,vectorized" "blis,gpu" "parallel,vectorized" "parallel,gpu" "vectorized,gpu" "blis,parallel,vectorized" "blis,parallel,gpu" "blis,vectorized,gpu" "parallel,vectorized,gpu" "blis,parallel,vectorized,gpu")
RUSTFLAGS_CONFIGS=(
    ""
    "-C target-cpu=native"
    "-C target-cpu=native -C opt-level=3"
    "-C target-cpu=native -C opt-level=3 -C lto=fat"
    "-C target-cpu=native -C opt-level=3 -C lto=fat -C codegen-units=1"
    "-C target-cpu=native -C opt-level=3 -C lto=fat -C codegen-units=1 -C panic=abort"
)

echo "Starting comprehensive matmul benchmark..."
echo "Results will be stored in $RESULTS_DIR/"

for i in "${!RUSTFLAGS_CONFIGS[@]}"; do
    rustflags="${RUSTFLAGS_CONFIGS[$i]}"
    rustflags_name="config_$i"
    
    for features in "${FEATURES[@]}"; do
        feature_name="${features:-default}"
        feature_name="${feature_name//,/_}"
        
        result_file="$RESULTS_DIR/bench_${rustflags_name}_${feature_name}.json"
        
        echo "Testing: RUSTFLAGS='$rustflags' Features='$features'"
        
        # Build with specific flags
        if [ -n "$features" ]; then
            RUSTFLAGS="$rustflags" cargo build --release --features "$features"
        else
            RUSTFLAGS="$rustflags" cargo build --release
        fi
        
        if [ $? -eq 0 ]; then
            # Run benchmark and capture output
            if [ -n "$features" ]; then
                timeout 300 cargo run --release --features "$features" > "$result_file" 2>&1
            else
                timeout 300 cargo run --release > "$result_file" 2>&1
            fi
            
            # Store build info and system details
            build_info_file="$RESULTS_DIR/build_info_${rustflags_name}_${feature_name}.json"
            {
                echo "{"
                echo "  \"rustflags\": \"$rustflags\","
                echo "  \"features\": \"$features\","
                echo "  \"timestamp\": \"$(date -Iseconds)\","
                echo "  \"rust_version\": \"$(rustc --version)\","
                echo "  \"cargo_version\": \"$(cargo --version)\","
                echo "  \"system_info\": \"$(uname -a)\","
                echo "  \"cpu_info\": \"$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'N/A')\","
                echo "  \"exit_code\": $?"
                echo "}"
            } > "$build_info_file"
            
            # Add metadata to result file
            echo "" >> "$result_file"
            echo "=== BENCHMARK METADATA ===" >> "$result_file"
            echo "RUSTFLAGS: $rustflags" >> "$result_file"
            echo "Features: $features" >> "$result_file"
            echo "Timestamp: $(date -Iseconds)" >> "$result_file"
            echo "Build Info: $build_info_file" >> "$result_file"
            
            echo "✓ Completed: $result_file"
        else
            echo "✗ Build failed for RUSTFLAGS='$rustflags' Features='$features'"
        fi
    done
done

echo "Benchmark complete! Results stored in $RESULTS_DIR/"
echo "Summary of result files:"
ls -la "$RESULTS_DIR"/bench_*.json