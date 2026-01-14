#!/usr/bin/env python3

import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def load_fastest_results():
    """Load the most recent fastest benchmark results."""
    results_dir = Path("results")
    if not results_dir.exists():
        return None, None
    
    # Find most recent fastest run
    fastest_files = list(results_dir.glob("fastest_run_*.json"))
    config_files = list(results_dir.glob("fastest_config_*.json"))
    
    if not fastest_files or not config_files:
        return None, None
    
    # Get most recent
    latest_result = max(fastest_files, key=os.path.getctime)
    latest_config = max(config_files, key=os.path.getctime)
    
    with open(latest_result) as f:
        result_data = json.load(f)
    
    with open(latest_config) as f:
        config_data = json.load(f)
    
    return result_data, config_data

def reproduce_benchmark(config_data):
    """Reproduce benchmark with saved configuration."""
    features = config_data["features"]
    rustflags = config_data["rustflags"]
    
    print(f"Reproducing benchmark with:")
    print(f"  Features: {features}")
    print(f"  RUSTFLAGS: {rustflags}")
    
    # Build
    env = os.environ.copy()
    env["RUSTFLAGS"] = rustflags
    
    build_cmd = ["cargo", "build", "--release", "--features", features, "--quiet"]
    result = subprocess.run(build_cmd, env=env, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Build failed: {result.stderr}")
        return None
    
    # Run benchmark
    run_cmd = ["cargo", "run", "--release", "--features", features, "--quiet", 
               "--", "--warmup-runs", "5", "--bench-runs", "10"]
    result = subprocess.run(run_cmd, env=env, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Benchmark failed: {result.stderr}")
        return None
    
    return json.loads(result.stdout)

def print_summary(result_data, config_data):
    """Print benchmark summary."""
    print("\n=== FASTEST BENCHMARK RESULTS ===")
    print(f"Timestamp: {config_data['timestamp']}")
    print(f"Configuration: {config_data['features']}")
    print(f"Latency: {result_data['latency_ms']:.2f}ms")
    print(f"Throughput: {result_data['throughput_gflops']:.2f} GFLOPS")
    print(f"Matrix Size: {result_data['workload_info']['matrix_size']}")
    print(f"Correctness: {'✓' if result_data['correctness'] else '✗'}")
    print(f"Max Error: {result_data['max_error']:.2e}")
    
    perf = result_data['performance_analysis']
    print(f"\nSpeedups:")
    print(f"  vs Naive: {perf['speedup_vs_naive']:.1f}x")
    print(f"  BLAS: {perf['blas_speedup']:.1f}x")
    print(f"  Parallel: {perf['parallel_speedup']:.1f}x")
    print(f"  GPU: {perf['gpu_speedup']:.1f}x")
    print(f"  Vectorized: {perf['vectorized_speedup']:.1f}x")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "reproduce":
        # Reproduce mode
        result_data, config_data = load_fastest_results()
        if not result_data or not config_data:
            print("No fastest results found. Run benchmark_fastest.sh first.")
            sys.exit(1)
        
        print("Reproducing fastest benchmark...")
        new_result = reproduce_benchmark(config_data)
        
        if new_result:
            print("\n=== REPRODUCTION RESULTS ===")
            print(f"Original: {result_data['latency_ms']:.2f}ms")
            print(f"Reproduced: {new_result['latency_ms']:.2f}ms")
            diff = abs(new_result['latency_ms'] - result_data['latency_ms'])
            print(f"Difference: {diff:.2f}ms ({diff/result_data['latency_ms']*100:.1f}%)")
            
            # Save reproduction result
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            repro_file = f"results/reproduction_{timestamp}.json"
            with open(repro_file, 'w') as f:
                json.dump(new_result, f, indent=2)
            print(f"Reproduction saved: {repro_file}")
    else:
        # Summary mode
        result_data, config_data = load_fastest_results()
        if not result_data or not config_data:
            print("No fastest results found. Run benchmark_fastest.sh first.")
            sys.exit(1)
        
        print_summary(result_data, config_data)

if __name__ == "__main__":
    main()