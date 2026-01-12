unsafe extern "C" {
    unsafe fn matmul_kernel_c(a: *const f32, b: *const f32, c: *mut f32, m: usize, n: usize, k: usize);
}


pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    unsafe {
        matmul_kernel_c(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, n, k);
    }
}
