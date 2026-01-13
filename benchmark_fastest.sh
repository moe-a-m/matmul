#!/bin/bash

# Fast benchmark script - finds and runs the fastest matmul configuration
# Saves results for reproducibility

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

# Optimal configurations based on typical performance patterns
FAST_CONFIGS=(
    "blis,parallel,vectorized,gpu:-C target-cpu=native -C opt-level=3 -C lto=fat -C codegen-units=1"
    "blis,gpu:-C target-cpu=native -C opt-level=3 -C lto=fat"
    "gpu,parallel:-C target-cpu=native -C opt-level=3 -C lto=fat"
    "gpu,vectorized:-C target-cpu=native -C opt-level=3"
    "gpu:-C target-cpu=native -C opt-level=3"
    "blis,parallel,vectorized:-C target-cpu=native -C opt-level=3 -C lto=fat -C codegen-units=1"
    "blis,parallel:-C target-cpu=native -C opt-level=3 -C lto=fat"
    "blis,vectorized:-C target-cpu=native -C opt-level=3"
)

echo "Finding fastest matmul configuration..."

best_time=999999
best_config=""
best_features=""
best_rustflags=""

for config in "${FAST_CONFIGS[@]}"; do
    features="${config%%:*}"
    rustflags="${config##*:}"
    
    echo "Testing: Features='$features' RUSTFLAGS='$rustflags'"
    
    # Build
    RUSTFLAGS="$rustflags" cargo build --release --features "$features" --quiet
    
    if [ $? -eq 0 ]; then
        # Quick benchmark run
        result=$(timeout 60 cargo run --release --features "$features" --quiet 2>/dev/null | jq -r '.latency_ms // empty' 2>/dev/null)
        
        if [ -n "$result" ] && [ "$result" != "null" ]; then
            echo "  Time: ${result}ms"
            
            if (( $(echo "$result < $best_time" | bc -l) )); then
                best_time=$result
                best_config=$config
                best_features=$features
                best_rustflags=$rustflags
            fi
        fi
    fi
done

if [ -n "$best_config" ]; then
    echo ""
    echo "Fastest configuration found:"
    echo "  Features: $best_features"
    echo "  RUSTFLAGS: $best_rustflags"
    echo "  Time: ${best_time}ms"
    echo ""
    
    # Run comprehensive benchmark with best config
    echo "Running comprehensive benchmark with fastest configuration..."
    
    timestamp=$(date +%Y%m%d_%H%M%S)
    result_file="$RESULTS_DIR/fastest_run_$timestamp.json"
    
    RUSTFLAGS="$best_rustflags" cargo build --release --features "$best_features" --quiet
    
    # Extended benchmark run
    cargo run --release --features "$best_features" \
        --quiet \
        -- --warmup-runs 5 --bench-runs 10 \
        --output "$result_file"
    
    # Save configuration metadata
    config_file="$RESULTS_DIR/fastest_config_$timestamp.json"
    {
        echo "{"
        echo "  \"features\": \"$best_features\","
        echo "  \"rustflags\": \"$best_rustflags\","
        echo "  \"timestamp\": \"$(date -Iseconds)\","
        echo "  \"rust_version\": \"$(rustc --version)\","
        echo "  \"system_info\": \"$(uname -a)\","
        echo "  \"cpu_info\": \"$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'N/A')\","
        echo "  \"result_file\": \"$result_file\""
        echo "}"
    } > "$config_file"
    
    echo "✓ Results saved:"
    echo "  Benchmark: $result_file"
    echo "  Config: $config_file"
else
    echo "✗ No working configuration found"
    exit 1
fi