#[allow(dead_code)]
pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    const TILE: usize = 64;

    c.fill(0.0);

    for ii in (0..m).step_by(TILE) {
        for jj in (0..n).step_by(TILE) {
            for kk in (0..k).step_by(TILE) {
                let i_end = (ii + TILE).min(m);
                let j_end = (jj + TILE).min(n);
                let k_end = (kk + TILE).min(k);

                for i in ii..i_end {
                    for j in jj..j_end {
                        let mut sum = c[i + n + j];
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
