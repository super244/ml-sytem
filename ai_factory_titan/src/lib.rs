pub mod backend;
pub mod detect;
pub mod python;
pub mod quantization;
pub mod scheduler;
pub mod telemetry;

pub use backend::{BackendKind, TitanBackend};
pub use detect::{detect_hardware, HardwareProfile};
pub use scheduler::{GpuTask, TitanScheduler};
