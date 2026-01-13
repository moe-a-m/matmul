pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    // Simple optimized implementation without external BLIS
    c.fill(0.0);

    const BLOCK_SIZE: usize = 32;

    for ii in (0..m).step_by(BLOCK_SIZE) {
        for jj in (0..n).step_by(BLOCK_SIZE) {
            for kk in (0..k).step_by(BLOCK_SIZE) {
                let i_end = (ii + BLOCK_SIZE).min(m);
                let j_end = (jj + BLOCK_SIZE).min(n);
                let k_end = (kk + BLOCK_SIZE).min(k);

                for i in ii..i_end {
                    for j in jj..j_end {
                        let mut sum = c[i * n + j];
                        for l in kk..k_end {
                            sum += a[i * k + l] * b[l * n + j];
                        }
                        c[i * n + j] = sum;
                    }
                }
            }
        }
    }
}
