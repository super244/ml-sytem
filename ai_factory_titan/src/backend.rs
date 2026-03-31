use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Metal,
    Cuda,
    CpuFallback,
}

#[derive(Clone, Debug, Serialize)]
pub struct TitanBackend {
    pub kind: BackendKind,
    pub mode: String,
    pub zero_copy: bool,
    pub silent_mode_cap_pct: u8,
}

impl TitanBackend {
    pub fn new(kind: BackendKind, silent_mode_cap_pct: u8) -> Self {
        let mode = match kind {
            BackendKind::Metal => "Metal-Direct",
            BackendKind::Cuda => "CUDA-Direct",
            BackendKind::CpuFallback => "CPU-Fallback",
        }
        .to_string();
        let zero_copy = matches!(kind, BackendKind::Metal);
        Self {
            kind,
            mode,
            zero_copy,
            silent_mode_cap_pct,
        }
    }
}
