#[cfg(feature = "python")]
use pyo3::exceptions::PyRuntimeError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use crate::detect::detect_hardware;

#[cfg(feature = "python")]
#[pyfunction]
fn titan_status_json() -> PyResult<String> {
    serde_json::to_string(&detect_hardware())
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

#[cfg(feature = "python")]
#[pyfunction]
fn compute_dot_product(lhs: Vec<f32>, rhs: Vec<f32>) -> PyResult<f32> {
    crate::cpu_kernels::dot_f32(&lhs, &rhs)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

#[cfg(feature = "python")]
#[pyfunction]
fn compute_matmul(
    lhs: Vec<f32>,
    m: usize,
    k: usize,
    rhs: Vec<f32>,
    n: usize,
) -> PyResult<Vec<f32>> {
    crate::cpu_kernels::matmul_f32(&lhs, m, k, &rhs, n)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

#[cfg(feature = "python")]
#[pyfunction]
fn vec_add(a: Vec<f32>, b: Vec<f32>) -> PyResult<Vec<f32>> {
    crate::cpu_kernels::vec_add_f32(&a, &b)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

#[cfg(feature = "python")]
#[pyfunction]
fn vec_mul(a: Vec<f32>, b: Vec<f32>) -> PyResult<Vec<f32>> {
    crate::cpu_kernels::vec_mul_f32(&a, &b)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

#[cfg(feature = "python")]
#[pyfunction]
fn rms_norm(input: Vec<f32>, eps: f32) -> PyResult<Vec<f32>> {
    Ok(crate::cpu_kernels::rms_norm_f32(&input, eps))
}

#[cfg(feature = "python")]
#[pyfunction]
fn softmax(input: Vec<f32>) -> PyResult<Vec<f32>> {
    Ok(crate::cpu_kernels::softmax_f32(&input))
}

#[cfg(feature = "python")]
#[pyfunction]
fn silu(input: Vec<f32>) -> PyResult<Vec<f32>> {
    Ok(crate::cpu_kernels::silu_f32(&input))
}

#[cfg(feature = "python")]
#[pymodule]
fn ai_factory_titan_py(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(titan_status_json, module)?)?;
    module.add_function(wrap_pyfunction!(compute_dot_product, module)?)?;
    module.add_function(wrap_pyfunction!(compute_matmul, module)?)?;
    module.add_function(wrap_pyfunction!(vec_add, module)?)?;
    module.add_function(wrap_pyfunction!(vec_mul, module)?)?;
    module.add_function(wrap_pyfunction!(rms_norm, module)?)?;
    module.add_function(wrap_pyfunction!(softmax, module)?)?;
    module.add_function(wrap_pyfunction!(silu, module)?)?;
    Ok(())
}
