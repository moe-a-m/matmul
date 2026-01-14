#[cfg(feature = "gpu")]
mod gpu_impl {
    unsafe extern "C" {
        fn tt_matmul_c(
            a: *const f32,
            b: *const f32,
            c: *mut f32,
            m: usize,
            n: usize,
            k: usize,
        );
    }

    pub fn matmul_gpu(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        unsafe {
            tt_matmul_c(
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
                m,
                n,
                k,
            );
        }
    }
}

pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    #[cfg(feature = "gpu")]
    {
        gpu_impl::matmul_gpu(a, b, c, m, n, k);
    }
    #[cfg(not(feature = "gpu"))]
    {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }
}