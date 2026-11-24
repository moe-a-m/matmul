use serde::Serialize;

/// One benchmark run of every compiled backend on one workload.
#[derive(Serialize)]
pub struct Report {
    pub schema_version: u32,
    pub timestamp_unix: u64,
    pub workload: WorkloadInfo,
    pub system: SystemInfo,
    pub build: BuildInfo,
    pub settings: Settings,
    pub reference: BackendResult,
    pub backends: Vec<BackendResult>,
    /// Fastest backend whose output matched the reference.
    pub fastest_correct_backend: Option<String>,
    /// Best latency and throughput of `fastest_correct_backend`.
    pub latency_ms: Option<f64>,
    pub throughput_gflops: Option<f64>,
    /// SHA-256 of the fastest correct backend's output matrix.
    pub output_hash: Option<String>,
}

#[derive(Serialize)]
pub struct WorkloadInfo {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    /// 2 * m * n * k floating-point operations.
    pub flop: u64,
    pub precision: &'static str,
    pub input_bytes: u64,
}

#[derive(Serialize)]
pub struct SystemInfo {
    pub cpu_model: String,
    pub logical_cpus: usize,
    pub os: &'static str,
    pub arch: &'static str,
    pub omp_num_threads: Option<String>,
    pub blis_num_threads: Option<String>,
    pub rayon_threads: usize,
}

#[derive(Serialize)]
pub struct BuildInfo {
    pub features: Vec<&'static str>,
    pub rustc: &'static str,
    pub rustflags: &'static str,
    pub target: &'static str,
    pub git_commit: &'static str,
    pub profile: &'static str,
}

#[derive(Serialize)]
pub struct Settings {
    pub warmup_runs: usize,
    pub bench_runs: usize,
    pub tolerance: f32,
}

#[derive(Serialize)]
pub struct BackendResult {
    pub name: &'static str,
    pub description: &'static str,
    pub runs: usize,
    pub best_ms: f64,
    pub median_ms: f64,
    pub gflops_best: f64,
    pub gflops_median: f64,
    pub speedup_vs_naive: f64,
    pub max_abs_error: f32,
    /// max |c - ref| / max |ref|
    pub rel_error: f32,
    pub correct: bool,
}
