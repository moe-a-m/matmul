use sha2::{Digest, Sha256};

pub fn compute_hash(matrix: &[f32]) -> String {
    let mut hasher = Sha256::new();
    let bytes = unsafe {
        std::slice::from_raw_parts(matrix.as_ptr() as *const u8, std::mem::size_of_val(matrix))
    };
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}
