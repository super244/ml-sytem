#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::exceptions::PyRuntimeError;

#[cfg(feature = "python")]
use crate::detect::detect_hardware;

#[cfg(feature = "python")]
#[pyfunction]
fn titan_status_json() -> PyResult<String> {
    serde_json::to_string(&detect_hardware()).map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

#[cfg(feature = "python")]
#[pymodule]
fn ai_factory_titan_py(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(titan_status_json, module)?)?;
    Ok(())
}
