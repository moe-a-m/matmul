#[cfg(all(feature = "avx2", target_arch = "x86_64"))]
pub mod avx2_blis;
#[cfg(feature = "blis")]
pub mod blis;
pub mod c_packed;
pub mod naive;
pub mod parallel;
pub mod tiled;
pub mod vectorized;

/// Signature shared by every backend: C (m x n) = A (m x k) * B (k x n), row-major fp32.
pub type MatmulFn = fn(&[f32], &[f32], &mut [f32], usize, usize, usize);

pub struct Backend {
    pub name: &'static str,
    pub description: &'static str,
    pub run: MatmulFn,
}

/// The naive triple loop, used as the correctness and speed reference.
pub const REFERENCE: Backend = Backend {
    name: "naive",
    description: "Rust triple loop (reference)",
    run: naive::matmul,
};

/// Every backend compiled into this build, excluding the reference.
pub fn backends() -> Vec<Backend> {
    #[allow(unused_mut)] // only mutated when optional backends are compiled in
    let mut list = vec![
        Backend {
            name: "tiled",
            description: "Rust, 64x64x64 cache blocking",
            run: tiled::matmul,
        },
        Backend {
            name: "vectorized",
            description: "Rust, cache blocking + 4-wide accumulators (relies on auto-vectorisation)",
            run: vectorized::matmul,
        },
        Backend {
            name: "parallel",
            description: "Rust, Rayon row-parallel triple loop",
            run: parallel::matmul,
        },
        Backend {
            name: "c_packed",
            description: "C, tiled with A/B packing (RVV intrinsics on RISC-V)",
            run: c_packed::matmul,
        },
    ];
    #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
    list.push(Backend {
        name: "avx2_blis",
        description: "C, BLIS-style packing + 6x16 AVX2/FMA microkernel, OpenMP",
        run: avx2_blis::matmul,
    });
    #[cfg(feature = "blis")]
    list.push(Backend {
        name: "blis",
        description: "BLIS library cblas_sgemm (external reference)",
        run: blis::matmul,
    });
    list
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill(len: usize, seed: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (((i * 7 + seed * 13) % 97) as f32 / 97.0) - 0.5)
            .collect()
    }

    /// Odd shapes exercise the edge paths (not multiples of 6, 16, 64 or the BLIS block sizes).
    const SHAPES: &[(usize, usize, usize)] = &[
        (1, 1, 1),
        (6, 16, 8),
        (17, 23, 31),
        (64, 64, 64),
        (130, 67, 257),
        (193, 300, 270),
    ];

    #[test]
    fn every_backend_matches_the_reference() {
        for &(m, n, k) in SHAPES {
            let a = fill(m * k, 1);
            let b = fill(k * n, 2);
            let mut expected = vec![0.0f32; m * n];
            (REFERENCE.run)(&a, &b, &mut expected, m, n, k);
            let scale = expected.iter().fold(1.0f32, |acc, v| acc.max(v.abs()));

            for backend in backends() {
                // Pre-fill with garbage: backends must overwrite C, not accumulate into it.
                let mut c = vec![f32::NAN; m * n];
                (backend.run)(&a, &b, &mut c, m, n, k);
                let max_err = c
                    .iter()
                    .zip(&expected)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_err / scale <= 1e-4,
                    "{} wrong for {m}x{n}x{k}: max abs error {max_err}",
                    backend.name
                );
            }
        }
    }
}
