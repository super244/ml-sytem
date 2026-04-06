//! Titan compute core — public API surface (v0.3.0).

pub mod backend;
#[cfg(feature = "cpp")]
pub mod cpp;
pub mod cpu_kernels;
#[cfg(feature = "cuda")]
pub mod cuda_kernels;
pub mod detect;
pub mod engine;
pub mod gguf;
pub mod kv_cache;
#[cfg(feature = "metal")]
pub mod metal_kernels;
pub mod python;
pub mod quantization;
pub mod runtime;
pub mod sampler;
pub mod scheduler;
pub mod telemetry;
pub mod tensor;

// ─── Core compute ─────────────────────────────────────────────────────────────

pub use cpu_kernels::{
    adamw_step_f32, batch_softmax_f32, dot_f32, fused_rms_norm_silu_f32, gelu_f32,
    gelu_exact_f32, matmul_f32, rms_norm_f32, silu_f32, softmax_f32, vec_add_f32, vec_mul_f32,
};

// ─── Backend ──────────────────────────────────────────────────────────────────

pub use backend::{BackendKind, TitanBackend};

// ─── CUDA (optional) ──────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
pub use cuda_kernels::{detect_cuda_gpus, ComputeCapability, CudaKernelCache, GpuArchitecture};

// ─── Metal (optional) ─────────────────────────────────────────────────────────

#[cfg(feature = "metal")]
pub use metal_kernels::{
    detect_metal_capabilities, AppleSiliconGeneration, MetalCapabilities, MetalKernelCache,
};

// ─── Hardware detection ───────────────────────────────────────────────────────

pub use detect::{detect_hardware, HardwareProfile};

// ─── Engine ───────────────────────────────────────────────────────────────────

pub use engine::{CacheStrategy, NativeAcceleration, TitanEngineDescriptor};

// ─── File format (GGUF) ───────────────────────────────────────────────────────

pub use gguf::{parse_gguf_header, GgufHeader};

// ─── KV cache ─────────────────────────────────────────────────────────────────

pub use kv_cache::{KvCache, KvCacheConfig, KvCacheStats};

// ─── Quantization ─────────────────────────────────────────────────────────────

pub use quantization::{q4_arrow_column, quantized_weight_schema};

// ─── Runtime ──────────────────────────────────────────────────────────────────

pub use runtime::{TitanRuntimeMode, TitanRuntimePlan};

// ─── Sampler ──────────────────────────────────────────────────────────────────

pub use sampler::{sample_token, SamplerConfig};

// ─── Scheduler ────────────────────────────────────────────────────────────────

pub use scheduler::{GpuTask, Priority, SchedulerStatus, TitanScheduler};

// ─── Tensor ───────────────────────────────────────────────────────────────────

pub use tensor::{BlockQ4_0, BlockQ8_0, QuantizationFormat, QuantizedTensorLayout, TensorShape};
