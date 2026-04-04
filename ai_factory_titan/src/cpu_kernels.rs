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
