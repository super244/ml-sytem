use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct TelemetryFrame {
    pub thermals_c: f32,
    pub memory_pressure: f32,
    pub flops_tflops: f32,
    pub queue_depth: usize,
}

impl TelemetryFrame {
    pub fn degraded(&self) -> bool {
        self.memory_pressure > 0.9 || self.thermals_c > 92.0
    }
}
