#[cfg(feature = "gpu")]
mod gpu_impl {
    use std::ffi::c_void;

    extern "C" {
        fn matmul_tiled(
            a: *const f32,
            b: *const f32,
            c: *mut f32,
            m: i32,
            n: i32,
            k: i32,
        );
    }

    pub fn matmul_gpu(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        unsafe {
            matmul_tiled(
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
                m as i32,
                n as i32,
                k as i32,
            );
        }
    }
}

fn fallback_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
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

pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    #[cfg(feature = "gpu")]
    {
        gpu_impl::matmul_gpu(a, b, c, m, n, k);
    }
    #[cfg(not(feature = "gpu"))]
    {
        fallback_matmul(a, b, c, m, n, k);
    }
}