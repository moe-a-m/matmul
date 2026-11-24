#!/usr/bin/env python3
"""Summarise benchmark reports (schema v2) written by the matmul harness.

    python3 analyze_results.py [DIR_OR_FILE ...]     # default: results/
"""
import json
import sys
from pathlib import Path


def load_reports(paths):
    reports = []
    for root in paths:
        files = [root] if root.is_file() else sorted(root.rglob("*.json"))
        for f in files:
            try:
                data = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                continue  # not a report (e.g. old-format logs)
            if isinstance(data, dict) and data.get("schema_version") == 2:
                reports.append((f, data))
    return reports


def main():
    paths = [Path(p) for p in sys.argv[1:]] or [Path("results")]
    reports = load_reports(paths)
    if not reports:
        print("No schema-v2 reports found. Run ./benchmark_all.sh first.")
        return 1

    best = {}
    for path, r in reports:
        w, s, b = r["workload"], r["system"], r["build"]
        threads = s.get("omp_num_threads") or s.get("rayon_threads")
        print(f"\n{path}  |  {w['m']}x{w['n']}x{w['k']}  |  {s['cpu_model']}  |  "
              f"features={','.join(b['features']) or 'none'}  threads={threads}  commit={b['git_commit']}")
        print(f"  {'backend':<12} {'best ms':>10} {'GFLOPS':>9} {'median':>9} {'vs naive':>9}  correct")
        for x in [r["reference"]] + r["backends"]:
            print(f"  {x['name']:<12} {x['best_ms']:>10.2f} {x['gflops_best']:>9.2f} "
                  f"{x['gflops_median']:>9.2f} {x['speedup_vs_naive']:>8.1f}x  {x['correct']}")
            key = (x["name"], f"{w['m']}x{w['n']}x{w['k']}")
            if x["correct"] and x["gflops_best"] > best.get(key, (0, None))[0]:
                best[key] = (x["gflops_best"], f"{path.name} (threads={threads})")

    print("\nBest correct result per backend and size:")
    for (name, size), (gflops, where) in sorted(best.items(), key=lambda kv: -kv[1][0]):
        print(f"  {name:<12} {size:<16} {gflops:>8.2f} GFLOPS   {where}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
