const TILE_SIZE: usize = 64;
const VECTOR_SIZE: usize = 4;

pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    c.fill(0.0);

    for ii in (0..m).step_by(TILE_SIZE) {
        for jj in (0..n).step_by(TILE_SIZE) {
            for kk in (0..k).step_by(TILE_SIZE) {
                let i_end = (ii + TILE_SIZE).min(m);
                let j_end = (jj + TILE_SIZE).min(n);
                let k_end = (kk + TILE_SIZE).min(k);

                for i in ii..i_end {
                    for j in (jj..j_end).step_by(VECTOR_SIZE) {
                        let j_vec_end = (j + VECTOR_SIZE).min(j_end);
                        let mut acc = [0.0f32; VECTOR_SIZE];

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