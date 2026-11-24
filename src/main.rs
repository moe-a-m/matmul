use clap::Parser;
use serde::Deserialize;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

mod bench;
mod matmul;
mod validate;

use bench::{BackendResult, BuildInfo, Report, Settings, SystemInfo, WorkloadInfo};
use matmul::{Backend, REFERENCE, backends};
use validate::{compare, compute_hash};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Benchmark every compiled matmul backend and check each one against a naive reference.
#[derive(Parser)]
struct Args {
    /// JSON workload file (see tests/*.json). Default: 1024 x 1024 x 1024.
    #[arg(long)]
    workload: Option<String>,
    /// Write the JSON report here instead of stdout.
    #[arg(long)]
    output: Option<String>,
    #[arg(long, default_value = "3")]
    warmup_runs: usize,
    #[arg(long, default_value = "10")]
    bench_runs: usize,
    /// Maximum allowed max|c - ref| / max|ref| for a backend to count as correct.
    #[arg(long, default_value = "1e-4")]
    tolerance: f32,
}

#[derive(Deserialize)]
struct WorkloadSpec {
    parameters: Parameters,
}

#[derive(Deserialize)]
struct Parameters {
    shapes: Shapes,
}

#[derive(Deserialize)]
struct Shapes {
    #[serde(rename = "M")]
    m: usize,
    #[serde(rename = "N")]
    n: usize,
    #[serde(rename = "K")]
    k: usize,
}

fn load_shape(path: Option<String>) -> Result<(usize, usize, usize)> {
    match path {
        Some(path) => {
            let spec: WorkloadSpec = serde_json::from_str(&std::fs::read_to_string(path)?)?;
            let s = spec.parameters.shapes;
            Ok((s.m, s.n, s.k))
        },
        None => Ok((1024, 1024, 1024)),
    }
}

fn init_matrices(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let a = (0..m * k).map(|i| ((i * 7) % 100) as f32 / 100.0).collect();
    let b = (0..k * n)
        .map(|i| ((i * 11) % 100) as f32 / 100.0)
        .collect();
    (a, b)
}

/// Runs a backend `warmup` + `runs` times; returns (sorted timings in seconds, last output).
fn time_backend(
    backend: &Backend,
    a: &[f32],
    b: &[f32],
    dims: (usize, usize, usize),
    warmup: usize,
    runs: usize,
) -> (Vec<f64>, Vec<f32>) {
    let (m, n, k) = dims;
    let mut c = vec![0.0f32; m * n];
    for _ in 0..warmup {
        (backend.run)(a, b, &mut c, m, n, k);
    }
    let mut times: Vec<f64> = (0..runs.max(1))
        .map(|_| {
            let start = Instant::now();
            (backend.run)(a, b, &mut c, m, n, k);
            start.elapsed().as_secs_f64()
        })
        .collect();
    times.sort_by(|x, y| x.partial_cmp(y).unwrap());
    (times, c)
}

fn summarize(
    backend: &Backend,
    times: &[f64],
    flop: u64,
    naive_best: f64,
    errors: (f32, f32),
    tolerance: f32,
) -> BackendResult {
    let best = times[0];
    let median = times[times.len() / 2];
    BackendResult {
        name: backend.name,
        description: backend.description,
        runs: times.len(),
        best_ms: best * 1e3,
        median_ms: median * 1e3,
        gflops_best: flop as f64 / best / 1e9,
        gflops_median: flop as f64 / median / 1e9,
        speedup_vs_naive: naive_best / best,
        max_abs_error: errors.0,
        rel_error: errors.1,
        correct: errors.1 <= tolerance,
    }
}

fn cpu_model() -> String {
    if let Ok(info) = std::fs::read_to_string("/proc/cpuinfo")
        && let Some(line) = info.lines().find(|l| l.starts_with("model name"))
        && let Some((_, v)) = line.split_once(':')
    {
        return v.trim().to_string();
    }
    std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn compiled_features() -> Vec<&'static str> {
    let mut f = Vec::new();
    if cfg!(feature = "avx2") {
        f.push("avx2");
    }
    if cfg!(feature = "blis") {
        f.push("blis");
    }
    f
}

fn main() -> Result<()> {
    let args = Args::parse();
    let (m, n, k) = load_shape(args.workload)?;
    let dims = (m, n, k);
    let flop = 2 * (m as u64) * (n as u64) * (k as u64);
    let (a, b) = init_matrices(m, n, k);

    // The naive reference is slow; time it once (plus one warm-up) on large problems.
    let (ref_warmup, ref_runs) = if flop > 200_000_000 {
        (0, 1)
    } else {
        (args.warmup_runs, args.bench_runs)
    };
    let (ref_times, reference_out) = time_backend(&REFERENCE, &a, &b, dims, ref_warmup, ref_runs);
    let naive_best = ref_times[0];
    let reference = summarize(
        &REFERENCE,
        &ref_times,
        flop,
        naive_best,
        (0.0, 0.0),
        args.tolerance,
    );

    let mut results: Vec<BackendResult> = Vec::new();
    let mut fastest: Option<(usize, Vec<f32>)> = None;
    for backend in backends() {
        let (times, out) = time_backend(&backend, &a, &b, dims, args.warmup_runs, args.bench_runs);
        let result = summarize(
            &backend,
            &times,
            flop,
            naive_best,
            compare(&out, &reference_out),
            args.tolerance,
        );
        let is_faster = fastest
            .as_ref()
            .is_none_or(|(i, _)| result.best_ms < results[*i].best_ms);
        if result.correct && is_faster {
            fastest = Some((results.len(), out));
        }
        results.push(result);
    }

    eprintln!(
        "{m}x{n}x{k} fp32 | {} | features: {:?}",
        cpu_model(),
        compiled_features()
    );
    eprintln!(
        "{:<12} {:>10} {:>10} {:>9} {:>10} {:>8}",
        "backend", "best ms", "GFLOPS", "vs naive", "rel err", "correct"
    );
    for r in std::iter::once(&reference).chain(results.iter()) {
        eprintln!(
            "{:<12} {:>10.2} {:>10.2} {:>8.1}x {:>10.2e} {:>8}",
            r.name, r.best_ms, r.gflops_best, r.speedup_vs_naive, r.rel_error, r.correct
        );
    }

    let (fastest_name, latency, gflops, hash) = match &fastest {
        Some((i, out)) => (
            Some(results[*i].name.to_string()),
            Some(results[*i].best_ms),
            Some(results[*i].gflops_best),
            Some(compute_hash(out)),
        ),
        None => (None, None, None, None),
    };

    let report = Report {
        schema_version: 2,
        timestamp_unix: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        workload: WorkloadInfo {
            m,
            n,
            k,
            flop,
            precision: "fp32",
            input_bytes: ((m * k + k * n) * 4) as u64,
        },
        system: SystemInfo {
            cpu_model: cpu_model(),
            logical_cpus: std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1),
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
            omp_num_threads: std::env::var("OMP_NUM_THREADS").ok(),
            blis_num_threads: std::env::var("BLIS_NUM_THREADS").ok(),
            rayon_threads: rayon::current_num_threads(),
        },
        build: BuildInfo {
            features: compiled_features(),
            rustc: env!("MATMUL_RUSTC_VERSION"),
            rustflags: env!("MATMUL_RUSTFLAGS"),
            target: env!("MATMUL_TARGET"),
            git_commit: env!("MATMUL_GIT_COMMIT"),
            profile: if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            },
        },
        settings: Settings {
            warmup_runs: args.warmup_runs,
            bench_runs: args.bench_runs,
            tolerance: args.tolerance,
        },
        reference,
        backends: results,
        fastest_correct_backend: fastest_name,
        latency_ms: latency,
        throughput_gflops: gflops,
        output_hash: hash,
    };

    let json = serde_json::to_string_pretty(&report)?;
    match args.output {
        Some(path) => std::fs::write(path, json)?,
        None => println!("{json}"),
    }
    Ok(())
}
