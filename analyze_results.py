#!/usr/bin/env python3
"""
Performance analysis script for matmul benchmark results
Analyzes all result files and generates performance summary
"""

import json
import os
import re
from pathlib import Path

def parse_result_file(filepath):
    """Parse benchmark result file and extract performance metrics"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract metadata
    metadata = {}
    if "=== BENCHMARK METADATA ===" in content:
        metadata_section = content.split("=== BENCHMARK METADATA ===")[1]
        for line in metadata_section.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip()
    
    # Extract performance metrics (adjust regex based on your output format)
    metrics = {}
    
    # Look for timing information
    time_patterns = [
        r'Time:\s*(\d+\.?\d*)\s*ms',
        r'Elapsed:\s*(\d+\.?\d*)\s*ms',
        r'Duration:\s*(\d+\.?\d*)\s*ms',
        r'(\d+\.?\d*)\s*ms'
    ]
    
    for pattern in time_patterns:
        matches = re.findall(pattern, content)
        if matches:
            metrics['time_ms'] = float(matches[0])
            break
    
    # Look for GFLOPS
    gflops_match = re.search(r'(\d+\.?\d*)\s*GFLOPS', content)
    if gflops_match:
        metrics['gflops'] = float(gflops_match.group(1))
    
    return {
        'file': filepath.name,
        'metadata': metadata,
        'metrics': metrics,
        'raw_output': content
    }

def analyze_results():
    """Analyze all benchmark results and generate summary"""
    results_dir = Path('results')
    
    if not results_dir.exists():
        print("Results directory not found!")
        return
    
    results = []
    
    # Parse all benchmark result files
    for result_file in results_dir.glob('bench_*.json'):
        try:
            result = parse_result_file(result_file)
            results.append(result)
        except Exception as e:
            print(f"Error parsing {result_file}: {e}")
    
    if not results:
        print("No benchmark results found!")
        return
    
    # Sort results by performance
    results_with_time = [r for r in results if 'time_ms' in r['metrics']]
    results_with_time.sort(key=lambda x: x['metrics']['time_ms'])
    
    # Generate summary
    summary = {
        'total_configurations': len(results),
        'successful_runs': len(results_with_time),
        'best_performance': results_with_time[0] if results_with_time else None,
        'worst_performance': results_with_time[-1] if results_with_time else None,
        'all_results': results
    }
    
    # Save summary
    with open(results_dir / 'performance_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print summary
    print(f"\n=== PERFORMANCE ANALYSIS SUMMARY ===")
    print(f"Total configurations tested: {summary['total_configurations']}")
    print(f"Successful runs: {summary['successful_runs']}")
    
    if summary['best_performance']:
        best = summary['best_performance']
        print(f"\nBest Performance:")
        print(f"  File: {best['file']}")
        print(f"  Time: {best['metrics'].get('time_ms', 'N/A')} ms")
        print(f"  GFLOPS: {best['metrics'].get('gflops', 'N/A')}")
        print(f"  Features: {best['metadata'].get('Features', 'N/A')}")
        print(f"  RUSTFLAGS: {best['metadata'].get('RUSTFLAGS', 'N/A')}")
    
    if summary['worst_performance'] and len(results_with_time) > 1:
        worst = summary['worst_performance']
        print(f"\nWorst Performance:")
        print(f"  File: {worst['file']}")
        print(f"  Time: {worst['metrics'].get('time_ms', 'N/A')} ms")
        print(f"  GFLOPS: {worst['metrics'].get('gflops', 'N/A')}")
        print(f"  Features: {worst['metadata'].get('Features', 'N/A')}")
        print(f"  RUSTFLAGS: {worst['metadata'].get('RUSTFLAGS', 'N/A')}")
    
    print(f"\nDetailed results saved to: {results_dir / 'performance_summary.json'}")

if __name__ == "__main__":
    analyze_results()