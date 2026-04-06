#[cfg(feature = "cpp")]
use std::os::raw::c_float;

#[cfg(feature = "cpp")]
#[link(name = "titan_cpp_kernels", kind = "static")]
extern "C" {
    fn titan_dot_f32(lhs: *const c_float, rhs: *const c_float, len: usize) -> c_float;
    fn titan_matmul_f32(
        lhs: *const c_float,
        m: usize,
        k: usize,
        rhs: *const c_float,
        n: usize,
        out: *mut c_float,
    );
    fn titan_vec_add_f32(a: *const c_float, b: *const c_float, out: *mut c_float, len: usize);
    fn titan_vec_mul_f32(a: *const c_float, b: *const c_float, out: *mut c_float, len: usize);
    fn titan_rms_norm_f32(input: *const c_float, output: *mut c_float, len: usize, eps: c_float);
    fn titan_softmax_f32(input: *const c_float, output: *mut c_float, len: usize);
    fn titan_silu_f32(input: *const c_float, output: *mut c_float, len: usize);
}

#[cfg(feature = "cpp")]
pub fn dot_f32(lhs: &[f32], rhs: &[f32]) -> Option<f32> {
    if lhs.len() != rhs.len() {
        return None;
    }
    unsafe { Some(titan_dot_f32(lhs.as_ptr(), rhs.as_ptr(), lhs.len())) }
}

#[cfg(feature = "cpp")]
pub fn matmul_f32(lhs: &[f32], m: usize, k: usize, rhs: &[f32], n: usize) -> Option<Vec<f32>> {
    if lhs.len() != m * k || rhs.len() != k * n {
        return None;
    }
    let mut out = vec![0.0f32; m * n];
    unsafe {
        titan_matmul_f32(lhs.as_ptr(), m, k, rhs.as_ptr(), n, out.as_mut_ptr());
    }
    Some(out)
}

#[cfg(feature = "cpp")]
pub fn vec_add_f32(a: &[f32], b: &[f32]) -> Option<Vec<f32>> {
    if a.len() != b.len() {
        return None;
    }
    let mut out = vec![0.0f32; a.len()];
    unsafe {
        titan_vec_add_f32(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), a.len());
    }
    Some(out)
}

#[cfg(feature = "cpp")]
pub fn vec_mul_f32(a: &[f32], b: &[f32]) -> Option<Vec<f32>> {
    if a.len() != b.len() {
        return None;
    }
    let mut out = vec![0.0f32; a.len()];
    unsafe {
        titan_vec_mul_f32(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), a.len());
    }
    Some(out)
}

#[cfg(feature = "cpp")]
pub fn rms_norm_f32(input: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    unsafe {
        titan_rms_norm_f32(input.as_ptr(), out.as_mut_ptr(), input.len(), eps);
    }
    out
}

#[cfg(feature = "cpp")]
pub fn softmax_f32(input: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    unsafe {
        titan_softmax_f32(input.as_ptr(), out.as_mut_ptr(), input.len());
    }
    out
}

#[cfg(feature = "cpp")]
pub fn silu_f32(input: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    unsafe {
        titan_silu_f32(input.as_ptr(), out.as_mut_ptr(), input.len());
    }
    out
}

#[cfg(feature = "cpp")]
#[repr(C)]
pub struct BlockQ4_0 {
    pub d: f32,
    pub qs: [u8; 16],
}

#[cfg(feature = "cpp")]
pub fn dequantize_q4_0(blocks: &[BlockQ4_0]) -> Vec<f32> {
    let num_blocks = blocks.len();
    let mut out = vec![0.0f32; num_blocks * 32];
    // SAFETY: BlockQ4_0 has same layout as C struct
    unsafe {
        extern "C" {
            fn titan_dequantize_q4_0(
                blocks: *const BlockQ4_0,
                num_blocks: usize,
                output: *mut f32,
            );
        }
        titan_dequantize_q4_0(blocks.as_ptr(), num_blocks, out.as_mut_ptr());
    }
    out
}

#[cfg(not(feature = "cpp"))]
pub fn dot_f32(_lhs: &[f32], _rhs: &[f32]) -> Option<f32> {
    None
}

#[cfg(not(feature = "cpp"))]
pub fn matmul_f32(_lhs: &[f32], _m: usize, _k: usize, _rhs: &[f32], _n: usize) -> Option<Vec<f32>> {
    None
}

#[cfg(not(feature = "cpp"))]
pub fn vec_add_f32(_a: &[f32], _b: &[f32]) -> Option<Vec<f32>> {
    None
}

#[cfg(not(feature = "cpp"))]
pub fn vec_mul_f32(_a: &[f32], _b: &[f32]) -> Option<Vec<f32>> {
    None
}

#[cfg(not(feature = "cpp"))]
pub fn rms_norm_f32(_input: &[f32], _eps: f32) -> Vec<f32> {
    vec![]
}

#[cfg(not(feature = "cpp"))]
pub fn softmax_f32(_input: &[f32]) -> Vec<f32> {
    vec![]
}

#[cfg(not(feature = "cpp"))]
pub fn silu_f32(_input: &[f32]) -> Vec<f32> {
    vec![]
}

#[cfg(not(feature = "cpp"))]
#[repr(C)]
pub struct BlockQ4_0 {
    pub d: f32,
    pub qs: [u8; 16],
}

#[cfg(not(feature = "cpp"))]
pub fn dequantize_q4_0(_blocks: &[BlockQ4_0]) -> Vec<f32> {
    vec![]
}
