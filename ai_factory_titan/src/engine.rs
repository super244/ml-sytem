//! Titan engine descriptor — authoritative capability manifest (v0.3.0).

use serde::Serialize;

use crate::tensor::{QuantizationFormat, QuantizedTensorLayout};

// ─── Cache Strategy ────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CacheStrategy {
    PagedKv,
    SlidingWindow,
    RadixTree, // prefix sharing — future
}

// ─── Acceleration ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize)]
pub struct NativeAcceleration {
    pub rust_fallback:      bool,
    pub cpp_kernels:        bool,
    pub metal_backend:      bool,
    pub cuda_backend:       bool,
    pub rayon_parallel:     bool,
    pub paged_kv_cache:     bool,
    pub priority_scheduler: bool,
    pub bf16_compute:       bool,
    pub fp8_compute:        bool,
    /// AdamW CPU optimizer built in.
    pub cpu_adamw:          bool,
    /// Fused RMSNorm+SiLU kernel available.
    pub fused_norm_silu:    bool,
}

// ─── Engine Descriptor ────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize)]
pub struct TitanEngineDescriptor {
    pub architecture:            &'static str,
    pub decode_model:            &'static str,
    pub version:                 &'static str,
    pub max_context_tokens:      usize,
    pub max_batch_tokens:        usize,
    pub max_batch_sequences:     usize,
    pub cache_strategy:          CacheStrategy,
    pub scheduler_queue_depth:   usize,
    pub priority_bands:          u8,
    pub runtime_env:             &'static str,
    pub gguf_support:            bool,
    pub kv_cache:                bool,
    pub sampler_stack:           Vec<&'static str>,
    pub supported_quantizations: Vec<QuantizationFormat>,
    pub default_layout:          QuantizedTensorLayout,
    pub acceleration:            NativeAcceleration,
}

impl TitanEngineDescriptor {
    /// Default descriptor for local (laptop / workstation) deployment.
    pub fn local_default() -> Self {
        Self {
            architecture:         "llm-runtime",
            decode_model:         "llama.cpp-inspired",
            version:              env!("CARGO_PKG_VERSION"),
            max_context_tokens:   131_072, // 128 k context window
            max_batch_tokens:     8_192,
            max_batch_sequences:  64,
            cache_strategy:       CacheStrategy::PagedKv,
            scheduler_queue_depth: 256,
            priority_bands:       3,
            runtime_env:          "AI_FACTORY_TITAN_RUNTIME",
            gguf_support:         true,
            kv_cache:             true,
            sampler_stack: vec![
                "argmax",
                "temperature",
                "top_k",
                "top_p",
                "min_p",
                "repetition_penalty",
                "typical_p",
            ],
            supported_quantizations: vec![
                QuantizationFormat::Q4_0,
                QuantizationFormat::Q4K,
                QuantizationFormat::Q5K,
                QuantizationFormat::Q6K,
                QuantizationFormat::Q8_0,
                QuantizationFormat::F16,
                QuantizationFormat::BF16,
            ],
            default_layout: QuantizedTensorLayout::for_format(QuantizationFormat::Q4K),
            acceleration: NativeAcceleration {
                rust_fallback:      true,
                cpp_kernels:        cfg!(feature = "cpp"),
                metal_backend:      cfg!(feature = "metal"),
                cuda_backend:       cfg!(feature = "cuda"),
                rayon_parallel:     true,
                paged_kv_cache:     true,
                priority_scheduler: true,
                bf16_compute:       true,
                fp8_compute:        cfg!(feature = "cuda"),
                cpu_adamw:          true,
                fused_norm_silu:    true,
            },
        }
    }

    /// Cloud profile for A100 deployment.
    pub fn cloud_a100() -> Self {
        let mut d = Self::local_default();
        d.max_context_tokens  = 131_072;
        d.max_batch_tokens    = 32_768;
        d.max_batch_sequences = 512;
        d.default_layout = QuantizedTensorLayout::for_format(QuantizationFormat::BF16);
        d
    }

    /// Hopper (H100/H200) profile.
    pub fn cloud_h100() -> Self {
        let mut d = Self::cloud_a100();
        d.acceleration.fp8_compute = cfg!(feature = "cuda");
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_version_is_set() {
        let d = TitanEngineDescriptor::local_default();
        assert!(!d.version.is_empty());
        assert!(d.max_context_tokens >= 8192);
    }

    #[test]
    fn local_default_has_expected_capabilities() {
        let d = TitanEngineDescriptor::local_default();
        assert_eq!(d.decode_model, "llama.cpp-inspired");
        assert!(d.acceleration.rust_fallback);
        assert!(d.supported_quantizations.len() >= 4);
        assert!(d.gguf_support);
        assert!(d.kv_cache);
        assert!(d.acceleration.rayon_parallel);
        assert!(d.acceleration.paged_kv_cache);
        assert!(d.acceleration.cpu_adamw);
        assert!(d.acceleration.fused_norm_silu);
    }
}
