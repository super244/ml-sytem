use serde::Serialize;

use crate::tensor::{QuantizationFormat, QuantizedTensorLayout};

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheStrategy {
    PagedKv,
    SlidingWindow,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeAcceleration {
    pub rust_fallback: bool,
    pub cpp_kernels: bool,
    pub metal_backend: bool,
    pub cuda_backend: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct TitanEngineDescriptor {
    pub architecture: &'static str,
    pub decode_model: &'static str,
    pub max_context_tokens: usize,
    pub max_batch_tokens: usize,
    pub cache_strategy: CacheStrategy,
    pub scheduler_queue_depth: usize,
    pub runtime_env: &'static str,
    pub gguf_support: bool,
    pub kv_cache: bool,
    pub sampler_stack: Vec<&'static str>,
    pub supported_quantizations: Vec<QuantizationFormat>,
    pub default_layout: QuantizedTensorLayout,
    pub acceleration: NativeAcceleration,
}

impl TitanEngineDescriptor {
    pub fn local_default() -> Self {
        Self {
            architecture: "llm-runtime",
            decode_model: "llama.cpp-inspired",
            max_context_tokens: 8192,
            max_batch_tokens: 2048,
            cache_strategy: CacheStrategy::PagedKv,
            scheduler_queue_depth: 64,
            runtime_env: "AI_FACTORY_TITAN_RUNTIME",
            gguf_support: true,
            kv_cache: true,
            sampler_stack: vec![
                "argmax",
                "temperature",
                "top_k",
                "top_p",
                "repetition_penalty",
            ],
            supported_quantizations: vec![
                QuantizationFormat::Q4_0,
                QuantizationFormat::Q4K,
                QuantizationFormat::Q8_0,
                QuantizationFormat::F16,
            ],
            default_layout: QuantizedTensorLayout::for_format(QuantizationFormat::Q4K),
            acceleration: NativeAcceleration {
                rust_fallback: true,
                cpp_kernels: cfg!(feature = "cpp"),
                metal_backend: cfg!(feature = "metal"),
                cuda_backend: cfg!(feature = "cuda"),
            },
        }
    }
}
