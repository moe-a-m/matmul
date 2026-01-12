use clap::Parser;
use serde::Deserialize;
use std::time::Instant;

mod bench;
mod matmul;
mod validate;

use bench::{Metrics, PerformanceAnalysis, WorkloadInfo};
use matmul::{blis, naive, optimized, parallel, tiled, vectorized};
use validate::compute_hash;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    workload: Option<String>,
    #[arg(long)]
    output: Option<String>,
    #[arg(long, default_value = "3")]
    warmup_runs: usize,
    #[arg(long, default_value = "5")]
    bench_runs: usize,
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

impl Default for WorkloadSpec {
    fn default() -> Self {
        Self {
            parameters: Parameters {
                shapes: Shapes { m: 1024, n: 1024, k: 1024 },
            },
        }
    }
}

fn benchmark_kernel<F>(kernel: F, runs: usize) -> f64
where
    F: Fn(),
{
    (0..runs)
        .map(|_| {
            let start = Instant::now();
            kernel();
            start.elapsed().as_secs_f64()
        })
        .sum::<f64>() / runs as f64
}

fn init_matrices(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let a = (0..m * k)
        .map(|i| ((i * 7) % 100) as f32 / 100.0)
        .collect();
    
    let b = (0..k * n)
        .map(|i| ((i * 11) % 100) as f32 / 100.0)
        .collect();
    
    (a, b)
}

#[derive(Debug)]
struct BenchmarkResults {
    opt_time: f64,
    naive_time: f64,
    blas_time: f64,
    tiled_time: f64,
    vectorized_time: f64,
    parallel_time: f64,
}

impl BenchmarkResults {
    fn new(
        a: &[f32], 
        b: &[f32], 
        m: usize, 
        n: usize, 
        k: usize, 
        warmup_runs: usize, 
        bench_runs: usize
    ) -> Self {
        // Warmup
        let mut c_warmup = vec![0.0f32; m * n];
        for _ in 0..warmup_runs {
            optimized::matmul(a, b, &mut c_warmup, m, n, k);
        }

        let opt_time = Self::benchmark_impl(|| {
            let mut c = vec![0.0f32; m * n];
            optimized::matmul(a, b, &mut c, m, n, k);
        }, bench_runs);

        let naive_time = Self::benchmark_impl(|| {
            let mut c = vec![0.0f32; m * n];
            naive::matmul(a, b, &mut c, m, n, k);
        }, 1);

        let blas_time = if cfg!(feature = "blis") {
            Self::benchmark_impl(|| {
                let mut c = vec![0.0f32; m * n];
                blis::matmul(a, b, &mut c, m, n, k);
            }, bench_runs)
        } else {
            naive_time
        };

        let tiled_time = Self::benchmark_impl(|| {
            let mut c = vec![0.0f32; m * n];
            tiled::matmul(a, b, &mut c, m, n, k);
        }, bench_runs);

        let vectorized_time = Self::benchmark_impl(|| {
            let mut c = vec![0.0f32; m * n];
            vectorized::matmul(a, b, &mut c, m, n, k);
        }, bench_runs);

        let parallel_time = Self::benchmark_impl(|| {
            let mut c = vec![0.0f32; m * n];
            parallel::matmul(a, b, &mut c, m, n, k);
        }, bench_runs);

        Self {
            opt_time,
            naive_time,
            blas_time,
            tiled_time,
            vectorized_time,
            parallel_time,
        }
    }

    fn benchmark_impl<F>(kernel: F, runs: usize) -> f64
    where
        F: Fn(),
    {
        benchmark_kernel(kernel, runs)
    }
}

fn compute_correctness(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>, f32) {
    let mut c_opt = vec![0.0f32; m * n];
    let mut c_naive = vec![0.0f32; m * n];
    
    optimized::matmul(a, b, &mut c_opt, m, n, k);
    naive::matmul(a, b, &mut c_naive, m, n, k);
    
    let max_error = c_opt
        .iter()
        .zip(&c_naive)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);

    (c_opt, c_naive, max_error)
}

impl From<(BenchmarkResults, &[f32], f32, usize, usize, usize)> for Metrics {
    fn from((results, c_opt, max_error, m, n, k): (BenchmarkResults, &[f32], f32, usize, usize, usize)) -> Self {
        let total_ops = 2u64 * m as u64 * n as u64 * k as u64;
        let memory_mb = (m * k + k * n + m * n) as f64 * 4.0 / (1024.0 * 1024.0 * 1024.0);
        let gflops = total_ops as f64 / 1e9 / results.opt_time;
        let bandwidth_gbps = memory_mb * 1024.0 / results.opt_time;

        Self {
            latency_ms: results.opt_time * 1000.0,
            throughput_gflops: gflops,
            ops_per_second: total_ops as f64 / results.opt_time,
            output_hash: compute_hash(c_opt),
            max_error,
            correctness: max_error < 1e-3,
            workload_info: WorkloadInfo {
                matrix_size: [m, n, k],
                total_ops,
                memory_usage_mb: memory_mb,
            },
            performance_analysis: PerformanceAnalysis {
                speedup_vs_naive: results.naive_time / results.opt_time,
                blas_speedup: results.naive_time / results.blas_time,
                tiled_speedup: results.naive_time / results.tiled_time,
                vectorized_speedup: results.naive_time / results.vectorized_time,
                parallel_speedup: results.naive_time / results.parallel_time,
                memory_bandwidth_gbps: bandwidth_gbps,
                compute_efficiency: gflops / 100.0,
            },
        }
    }
}

fn load_workload(path: Option<String>) -> Result<WorkloadSpec> {
    match path {
        Some(path) => {
            let data = std::fs::read_to_string(path)?;
            Ok(serde_json::from_str(&data)?)
        }
        None => Ok(WorkloadSpec::default()),
    }
}

fn output_results(metrics: &Metrics, output_path: Option<String>) -> Result<()> {
    let json = serde_json::to_string_pretty(metrics)?;
    
    match output_path {
        Some(path) => std::fs::write(path, json).map_err(Into::into),
        None => {
            print!("{}", json);
            Ok(())
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let spec = load_workload(args.workload)?;
    let (m, n, k) = (spec.parameters.shapes.m, spec.parameters.shapes.n, spec.parameters.shapes.k);
    
    let (a, b) = init_matrices(m, n, k);
    let results = BenchmarkResults::new(&a, &b, m, n, k, args.warmup_runs, args.bench_runs);
    let (c_opt, _c_naive, max_error) = compute_correctness(&a, &b, m, n, k);
    
    let metrics = Metrics::from((results, c_opt.as_slice(), max_error, m, n, k));
    output_results(&metrics, args.output)?;
    
    Ok(())
}