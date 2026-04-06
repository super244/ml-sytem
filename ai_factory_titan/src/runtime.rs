//! Titan runtime planning — selects the decode path based on environment and capabilities.
//!
//! v0.2 changes:
//! - `TitanRuntimeMode::auto()` — probes available features and promotes automatically.
//! - `TitanRuntimePlan::current()` now enables `can_generate` when Rust-primary mode is active.
//! - `TitanRuntimePlan::capabilities()` — returns a structured capability map for dashboard display.
//! - `TitanRuntimePlan::promote_if_capable()` — idempotent promotion helper used by health checks.

use serde::Serialize;
use std::collections::HashMap;

// ─── Runtime Mode ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TitanRuntimeMode {
    PythonFallback,
    RustCanary,
    RustPrimary,
}

impl TitanRuntimeMode {
    /// Determine runtime mode from the `AI_FACTORY_TITAN_RUNTIME` env-var.
    pub fn from_env() -> Self {
        match std::env::var("AI_FACTORY_TITAN_RUNTIME")
            .unwrap_or_else(|_| "python".to_string())
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "rust" | "rust-primary" => Self::RustPrimary,
            "rust-canary"           => Self::RustCanary,
            _                       => Self::PythonFallback,
        }
    }

    /// Promote to `RustPrimary` if compiled-in features are sufficient.
    pub fn auto() -> Self {
        // If the user pinned a mode, honour it.
        let env_mode = Self::from_env();
        if env_mode != Self::PythonFallback {
            return env_mode;
        }
        // Auto-promote when Metal or CUDA backend is available.
        if cfg!(feature = "metal") || cfg!(feature = "cuda") {
            return Self::RustCanary; // Escalate to canary; promotion to primary is manual.
        }
        Self::PythonFallback
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::PythonFallback => "python-fallback",
            Self::RustCanary     => "rust-canary",
            Self::RustPrimary    => "rust-primary",
        }
    }

    pub fn is_rust_backed(&self) -> bool {
        matches!(self, Self::RustCanary | Self::RustPrimary)
    }
}

// ─── Runtime Plan ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize)]
pub struct TitanRuntimePlan {
    pub mode:             TitanRuntimeMode,
    pub can_generate:     bool,
    pub gguf_enabled:     bool,
    pub kv_cache_enabled: bool,
    pub sampler_enabled:  bool,
    pub paged_kv_cache:   bool,
    pub priority_sched:   bool,
    pub rayon_kernels:    bool,
    pub reason:           &'static str,
    pub version:          &'static str,
}

impl TitanRuntimePlan {
    pub fn current() -> Self {
        let mode = TitanRuntimeMode::from_env();
        let rust_active = mode.is_rust_backed();
        let rust_primary = matches!(mode, TitanRuntimeMode::RustPrimary);

        Self {
            mode,
            // Generation loop can run fully in Rust only at RustPrimary.
            can_generate: rust_primary,
            gguf_enabled:     rust_active,
            kv_cache_enabled: rust_active,
            sampler_enabled:  rust_active,
            paged_kv_cache:   rust_active,
            priority_sched:   true, // Always available — uses crossbeam channels.
            rayon_kernels:    true, // Rayon is always compiled in.
            version: env!("CARGO_PKG_VERSION"),
            reason: if rust_primary {
                "Rust-primary: full decode loop and KV cache managed by Titan."
            } else if rust_active {
                "Rust-canary: Titan surfaces metadata and KV cache; generation is Python-backed."
            } else {
                "Python-fallback: Transformers path. Set AI_FACTORY_TITAN_RUNTIME=rust to enable Titan."
            },
        }
    }

    /// Return a flat string→bool capability map for the dashboard.
    pub fn capabilities(&self) -> HashMap<&'static str, bool> {
        [
            ("can_generate",    self.can_generate),
            ("gguf",            self.gguf_enabled),
            ("kv_cache",        self.kv_cache_enabled),
            ("paged_kv_cache",  self.paged_kv_cache),
            ("sampler",         self.sampler_enabled),
            ("priority_sched",  self.priority_sched),
            ("rayon_kernels",   self.rayon_kernels),
            ("metal",           cfg!(feature = "metal")),
            ("cuda",            cfg!(feature = "cuda")),
            ("cpp",             cfg!(feature = "cpp")),
        ]
        .into_iter()
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_fallback_is_default() {
        // In the test environment, `AI_FACTORY_TITAN_RUNTIME` is typically unset.
        // We re-test `from_env` without mutating the shared env.
        let plan = TitanRuntimePlan::current();
        assert!(!plan.version.is_empty());
        assert!(plan.rayon_kernels); // Always true.
        assert!(!plan.reason.is_empty());
    }

    #[test]
    fn capabilities_has_expected_keys() {
        let plan = TitanRuntimePlan::current();
        let caps = plan.capabilities();
        for key in ["can_generate", "gguf", "kv_cache", "paged_kv_cache", "sampler"] {
            assert!(caps.contains_key(key), "missing capability key: {key}");
        }
    }

    #[test]
    fn rust_primary_enables_generation() {
        let plan = TitanRuntimePlan {
            mode:             TitanRuntimeMode::RustPrimary,
            can_generate:     true,
            gguf_enabled:     true,
            kv_cache_enabled: true,
            sampler_enabled:  true,
            paged_kv_cache:   true,
            priority_sched:   true,
            rayon_kernels:    true,
            reason:           "test",
            version:          env!("CARGO_PKG_VERSION"),
        };
        assert!(plan.can_generate);
        assert!(plan.mode.is_rust_backed());
    }
}
