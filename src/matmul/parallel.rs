use rayon::prelude::*;

pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], _m: usize, n: usize, k: usize) {
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