use anyhow::{anyhow, ensure};

#[cfg(feature = "cpp")]
use crate::cpp;

pub fn dot_f32(lhs: &[f32], rhs: &[f32]) -> anyhow::Result<f32> {
    ensure!(lhs.len() == rhs.len(), "dot product dimension mismatch");

    #[cfg(feature = "cpp")]
    if let Some(value) = cpp::dot_f32(lhs, rhs) {
        return Ok(value);
    }

    Ok(lhs.iter().zip(rhs).map(|(left, right)| left * right).sum())
}

pub fn matmul_f32(lhs: &[f32], m: usize, k: usize, rhs: &[f32], n: usize) -> anyhow::Result<Vec<f32>> {
    ensure!(lhs.len() == m.saturating_mul(k), "lhs shape does not match m x k");
    ensure!(rhs.len() == k.saturating_mul(n), "rhs shape does not match k x n");

    #[cfg(feature = "cpp")]
    if let Some(values) = cpp::matmul_f32(lhs, m, k, rhs, n) {
        return Ok(values);
    }

    let mut out = vec![0.0f32; m.saturating_mul(n)];
    for row in 0..m {
        let lhs_row = &lhs[row * k..(row + 1) * k];
        for col in 0..n {
            let mut acc = 0.0f32;
            for inner in 0..k {
                acc += lhs_row[inner] * rhs[inner * n + col];
            }
            out[row * n + col] = acc;
        }
    }

    if out.iter().any(|value| !value.is_finite()) {
        return Err(anyhow!("matmul produced non-finite output"));
    }
    Ok(out)
}

pub fn vec_add_f32(a: &[f32], b: &[f32]) -> anyhow::Result<Vec<f32>> {
    ensure!(a.len() == b.len(), "vector addition dimension mismatch");

    #[cfg(feature = "cpp")]
    if let Some(result) = cpp::vec_add_f32(a, b) {
        return Ok(result);
    }

    Ok(a.iter().zip(b).map(|(x, y)| x + y).collect())
}

pub fn vec_mul_f32(a: &[f32], b: &[f32]) -> anyhow::Result<Vec<f32>> {
    ensure!(a.len() == b.len(), "vector multiplication dimension mismatch");

    #[cfg(feature = "cpp")]
    if let Some(result) = cpp::vec_mul_f32(a, b) {
        return Ok(result);
    }

    Ok(a.iter().zip(b).map(|(x, y)| x * y).collect())
}

pub fn rms_norm_f32(input: &[f32], eps: f32) -> Vec<f32> {
    #[cfg(feature = "cpp")]
    {
        return cpp::rms_norm_f32(input, eps);
    }
    #[cfg(not(feature = "cpp"))]
    {
        if input.is_empty() {
            return vec![];
        }
        let sum_sq: f32 = input.iter().map(|x| x * x).sum();
        let rms = (sum_sq / input.len() as f32 + eps).sqrt();
        let scale = 1.0 / rms;
        input.iter().map(|x| x * scale).collect()
    }
}

pub fn softmax_f32(input: &[f32]) -> Vec<f32> {
    #[cfg(feature = "cpp")]
    {
        return cpp::softmax_f32(input);
    }
    #[cfg(not(feature = "cpp"))]
    {
        if input.is_empty() {
            return vec![];
        }
        let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_vals: Vec<f32> = input.iter().map(|x| (x - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        exp_vals.iter().map(|x| x / sum).collect()
    }
}

pub fn silu_f32(input: &[f32]) -> Vec<f32> {
    #[cfg(feature = "cpp")]
    {
        return cpp::silu_f32(input);
    }
    #[cfg(not(feature = "cpp"))]
    {
        input.iter().map(|&x| x / (1.0 + (-x).exp())).collect()
    }
}
