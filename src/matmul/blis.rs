//! Reference SGEMM from the BLIS library through its CBLAS interface.
//! Available with the `blis` feature (requires `libblis`; see README).
//!
//! Earlier versions of this file did not call BLIS at all (they contained a
//! blocked Rust loop); this now calls `cblas_sgemm` from the system library.

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;

unsafe extern "C" {
    fn cblas_sgemm(
        layout: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert!(a.len() >= m * k && b.len() >= k * n && c.len() >= m * n);
    let dim = |x: usize| i32::try_from(x).expect("matrix dimension exceeds i32");
    let (mi, ni, ki) = (dim(m), dim(n), dim(k));
    // SAFETY: row-major, no transpose, leading dimensions match the slice layouts.
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            mi,
            ni,
            ki,
            1.0,
            a.as_ptr(),
            ki.max(1),
            b.as_ptr(),
            ni.max(1),
            0.0,
            c.as_mut_ptr(),
            ni.max(1),
        );
    }
}
