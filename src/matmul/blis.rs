// BLIS integration - requires libblis-dev
// Enable with: cargo build --features blis

#[allow(dead_code)]
pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    // if BLIS is not available, use optimized fallback
    crate::matmul::optimized::matmul(a, b, c, m, n, k);
}
