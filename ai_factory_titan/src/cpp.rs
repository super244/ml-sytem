#[cfg(feature = "cpp")]
unsafe extern "C" {
    fn titan_dot_f32(lhs: *const f32, rhs: *const f32, len: usize) -> f32;
    fn titan_matmul_f32(lhs: *const f32, m: usize, k: usize, rhs: *const f32, n: usize, out: *mut f32);
}

#[cfg(feature = "cpp")]
pub fn dot_f32(lhs: &[f32], rhs: &[f32]) -> Option<f32> {
    if lhs.len() != rhs.len() {
        return None;
    }
    Some(unsafe { titan_dot_f32(lhs.as_ptr(), rhs.as_ptr(), lhs.len()) })
}

#[cfg(not(feature = "cpp"))]
pub fn dot_f32(_lhs: &[f32], _rhs: &[f32]) -> Option<f32> {
    None
}

#[cfg(feature = "cpp")]
pub fn matmul_f32(lhs: &[f32], m: usize, k: usize, rhs: &[f32], n: usize) -> Option<Vec<f32>> {
    if lhs.len() != m.saturating_mul(k) || rhs.len() != k.saturating_mul(n) {
        return None;
    }
    let mut out = vec![0.0f32; m.saturating_mul(n)];
    unsafe { titan_matmul_f32(lhs.as_ptr(), m, k, rhs.as_ptr(), n, out.as_mut_ptr()) };
    Some(out)
}

#[cfg(not(feature = "cpp"))]
pub fn matmul_f32(_lhs: &[f32], _m: usize, _k: usize, _rhs: &[f32], _n: usize) -> Option<Vec<f32>> {
    None
}
