#[cfg(not(feature = "blis"))]
use rayon::prelude::*;

#[cfg(feature = "blis")]
extern "C" {
    fn blis_gemm(
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f32,
        a: *const f32,
        rsa: i32,
        csa: i32,
        b: *const f32,
        rsb: i32,
        csb: i32,
        beta: *const f32,
        c: *mut f32,
        rsc: i32,
        csc: i32,
    );
}

#[cfg(feature = "blis")]
const BLIS_NO_TRANSPOSE: i32 = 0;

#[allow(dead_code)]
pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], _m: usize, n: usize, k: usize) {
    #[cfg(feature = "blis")]
    {
        let alpha = 1.0f32;
        let beta = 0.0f32;

        unsafe {
            blis_gemm(
                BLIS_NO_TRANSPOSE,
                BLIS_NO_TRANSPOSE,
                m as i32,
                n as i32,
                k as i32,
                &alpha,
                a.as_ptr(),
                1,
                k as i32,
                b.as_ptr(),
                1,
                n as i32,
                &beta,
                c.as_mut_ptr(),
                1,
                n as i32,
            );
        }
    }
    #[cfg(not(feature = "blis"))]
    {
        c.par_chunks_mut(n).enumerate().for_each(|(i, c_row)| {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c_row[j] = sum;
            }
        });
    }
}
