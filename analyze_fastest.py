#!/usr/bin/env python3
"""Show the most recent results/fastest_*.json, or re-run it and compare.

    python3 analyze_fastest.py             # show the latest report
    python3 analyze_fastest.py reproduce   # rebuild with the same features/threads and compare
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def latest_report():
    files = sorted(Path("results").glob("fastest_*.json"), key=os.path.getmtime)
    return (files[-1], json.loads(files[-1].read_text())) if files else (None, None)


def show(path, r):
    s, b = r["system"], r["build"]
    print(f"Report:   {path}")
    print(f"CPU:      {s['cpu_model']} ({s['logical_cpus']} logical CPUs), threads={s.get('omp_num_threads')}")
    print(f"Build:    features={','.join(b['features']) or 'none'}  rustflags='{b['rustflags']}'  commit={b['git_commit']}")
    print(f"Fastest correct backend: {r['fastest_correct_backend']}  "
          f"{r['throughput_gflops']:.2f} GFLOPS ({r['latency_ms']:.2f} ms)")
    for x in r["backends"]:
        print(f"  {x['name']:<12} {x['gflops_best']:>8.2f} GFLOPS  rel_err={x['rel_error']:.1e}  correct={x['correct']}")


def reproduce(r):
    b, s = r["build"], r["system"]
    features = ",".join(b["features"])
    threads = s.get("omp_num_threads") or str(s.get("rayon_threads", 1))
    env = dict(os.environ, RUSTFLAGS=b["rustflags"], OMP_NUM_THREADS=threads,
               BLIS_NUM_THREADS=threads, RAYON_NUM_THREADS=threads)
    cmd = ["cargo", "run", "--release", "--quiet"] + (["--features", features] if features else [])
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out = tmp.name
    subprocess.run(cmd + ["--", "--warmup-runs", str(r["settings"]["warmup_runs"]),
                          "--bench-runs", str(r["settings"]["bench_runs"]), "--output", out],
                   env=env, check=True)
    new = json.loads(Path(out).read_text())
    old_g, new_g = r["throughput_gflops"], new["throughput_gflops"]
    print(f"\nOriginal:   {r['fastest_correct_backend']} {old_g:.2f} GFLOPS")
    print(f"Reproduced: {new['fastest_correct_backend']} {new_g:.2f} GFLOPS ({(new_g - old_g) / old_g * 100:+.1f}%)")


def main():
    path, report = latest_report()
    if report is None:
        print("No results/fastest_*.json found. Run ./benchmark_fastest.sh first.")
        return 1
    show(path, report)
    if len(sys.argv) > 1 and sys.argv[1] == "reproduce":
        reproduce(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
