use sha2::{Digest, Sha256};

pub fn compute_hash(matrix: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for v in matrix {
        hasher.update(v.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

/// Returns (max absolute error, max absolute error / max |reference|).
pub fn compare(output: &[f32], reference: &[f32]) -> (f32, f32) {
    let max_abs = output
        .iter()
        .zip(reference)
        .map(|(x, y)| {
            if x.is_nan() || y.is_nan() {
                f32::INFINITY
            } else {
                (x - y).abs()
            }
        })
        .fold(0.0f32, f32::max);
    let scale = reference
        .iter()
        .fold(f32::MIN_POSITIVE, |acc, v| acc.max(v.abs()));
    (max_abs, max_abs / scale)
}
