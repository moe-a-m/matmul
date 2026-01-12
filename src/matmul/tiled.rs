const TILE_SIZE: usize = 64;

pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    c.fill(0.0);

    for ii in (0..m).step_by(TILE_SIZE) {
        for jj in (0..n).step_by(TILE_SIZE) {
            for kk in (0..k).step_by(TILE_SIZE) {
                let i_end = (ii + TILE_SIZE).min(m);
                let j_end = (jj + TILE_SIZE).min(n);
                let k_end = (kk + TILE_SIZE).min(k);

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