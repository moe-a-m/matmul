use serde::Serialize;

#[derive(Serialize)]
pub struct Metrics {
    pub latency_ms: f64,
    pub throughput_gflops: f64,
    pub ops_per_second: f64,
    pub output_hash: String,
    pub max_error: f32,
    pub correctness: bool,
    pub workload_info: WorkloadInfo,
    pub performance_analysis: PerformanceAnalysis,
}

#[derive(Serialize)]
pub struct WorkloadInfo {
    pub matrix_size: [usize; 3],
    pub total_ops: u64,
    pub memory_usage_mb: f64,
}

#[derive(Serialize)]
pub struct PerformanceAnalysis {
    pub speedup_vs_naive: f64,
    pub blas_speedup: f64,
    pub tiled_speedup: f64,
    pub vectorized_speedup: f64,
    pub parallel_speedup: f64,
    pub memory_bandwidth_gbps: f64,
    pub compute_efficiency: f64,
}
