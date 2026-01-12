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
        rsc: f32,
        csc: f32,
    );
}

#[cfg(feature = "blis")]
const BLIS_NO_TRANSPOSE: i32 = 0;

#[allow(dead_code)]
pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
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
        // Vectorized fallback
        const TILE: usize = 64;
        const VECTOR_SIZE: usize = 4;

        c.fill(0.0);

        for ii in (0..m).step_by(TILE) {
            for jj in (0..n).step_by(TILE) {
                for kk in (0..k).step_by(TILE) {
                    let i_end = (ii + TILE).min(m);
                    let j_end = (jj + TILE).min(n);
                    let k_end = (kk + TILE).min(k);

                    for i in ii..i_end {
                        for j in (jj..j_end).step_by(VECTOR_SIZE) {
                            let j_vec_end = (j + VECTOR_SIZE).min(j_end);
                            let mut acc = [0.0f32; 4];

                            for l in kk..k_end {
                                let a_val = a[i * k + l];
                                for (idx, jj) in (j..j_vec_end).enumerate() {
                                    acc[idx] += a_val * b[l * n + jj];
                                }
                            }

                            for (idx, jj) in (j..j_vec_end).enumerate() {
                                c[i * n + jj] += acc[idx];
                            }
                        }
                    }
                }
            }
        }
    }
}
