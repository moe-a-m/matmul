//! Tiled C kernel with A/B packing (`kernels/matmul_kernel.c`).
//! On RISC-V targets with the V extension it uses RVV intrinsics; elsewhere a
//! scalar inner loop that the C compiler may auto-vectorise.

unsafe extern "C" {
    fn matmul_kernel_c(a: *const f32, b: *const f32, c: *mut f32, m: usize, n: usize, k: usize);
}

pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert!(a.len() >= m * k && b.len() >= k * n && c.len() >= m * n);
    // SAFETY: the slices are at least as large as the dimensions passed to C.
    unsafe {
        matmul_kernel_c(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, n, k);
    }
}
