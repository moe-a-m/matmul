//! BLIS-style packed SGEMM with a 6x16 AVX2/FMA microkernel, written in C
//! (`kernels/avx2_blis_kernel.c`) and parallelised with OpenMP.
//!
//! Available only with the `avx2` feature on x86-64. This backend used to be
//! exposed as the `gpu` feature; it has always run on the CPU.

unsafe extern "C" {
    fn avx2_blis_matmul_c(a: *const f32, b: *const f32, c: *mut f32, m: usize, n: usize, k: usize);
}

pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert!(a.len() >= m * k && b.len() >= k * n && c.len() >= m * n);
    // SAFETY: the slices are at least as large as the dimensions passed to C.
    unsafe {
        avx2_blis_matmul_c(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, n, k);
    }
}
