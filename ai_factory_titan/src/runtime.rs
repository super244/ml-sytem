use serde::Serialize;

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TitanRuntimeMode {
    PythonFallback,
    RustCanary,
    RustPrimary,
}

impl TitanRuntimeMode {
    pub fn from_env() -> Self {
        match std::env::var("AI_FACTORY_TITAN_RUNTIME")
            .ok()
            .unwrap_or_else(|| "python".to_string())
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "rust" | "rust-primary" => Self::RustPrimary,
            "rust-canary" => Self::RustCanary,
            _ => Self::PythonFallback,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::PythonFallback => "python-fallback",
            Self::RustCanary => "rust-canary",
            Self::RustPrimary => "rust-primary",
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct TitanRuntimePlan {
    pub mode: TitanRuntimeMode,
    pub can_generate: bool,
    pub gguf_enabled: bool,
    pub kv_cache_enabled: bool,
    pub sampler_enabled: bool,
    pub reason: &'static str,
}

impl TitanRuntimePlan {
    pub fn current() -> Self {
        let mode = TitanRuntimeMode::from_env();
        let metadata_enabled = !matches!(mode, TitanRuntimeMode::PythonFallback);
        Self {
            mode,
            // Titan can surface runtime metadata today, but the decode loop still lives on the
            // Python side. Keep the status contract honest until Rust generation is implemented.
            can_generate: false,
            gguf_enabled: metadata_enabled,
            kv_cache_enabled: metadata_enabled,
            sampler_enabled: metadata_enabled,
            reason: if metadata_enabled {
                "Rust Titan metadata path enabled, but generation remains Python-backed."
            } else {
                "Python Transformers path remains primary until Titan runtime is promoted."
            },
        }
    }
}
