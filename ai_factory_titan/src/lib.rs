//! Titan compute core — public API surface (v0.5.0).
//!
//! This version includes:
//! - M5 Ultra (80-core GPU) support with 1228 GB/s bandwidth and dual-chip
//! - Blackwell/GB200 support with FP8, 5th gen Tensor Cores, and secure CUDA
//! - RTX 50-series (5090/5080) optimizations
//! - Async kernel execution with work stealing and task prioritization
//! - Distributed training primitives with NCCL-compatible collectives
//! - Enhanced Python bindings with zero-copy tensor support
//! - Next-gen memory management with unified memory pools
//! - Hardware-specific kernel auto-selection

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

// New modules for v0.4
pub mod async_runtime;
pub mod distributed;
pub mod memory;
pub mod profiling;

// ─── Core compute ─────────────────────────────────────────────────────────────

pub use cpu_kernels::{
    adamw_step_f32, batch_softmax_f32, dot_f32, fused_rms_norm_silu_f32, gelu_f32,
    gelu_exact_f32, matmul_f32, rms_norm_f32, silu_f32, softmax_f32, vec_add_f32, vec_mul_f32,
};

// ─── Backend ──────────────────────────────────────────────────────────────────

pub use backend::{BackendKind, TitanBackend};

// ─── CUDA (optional) ──────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
pub use cuda_kernels::{
    detect_cuda_gpus, ComputeCapability, CudaKernelCache, GpuArchitecture,
    CudaStreamPool, GradientAccumulator,
};

// ─── Metal (optional) ─────────────────────────────────────────────────────────

#[cfg(feature = "metal")]
pub use metal_kernels::{
    detect_metal_capabilities, AppleSiliconGeneration, MetalCapabilities, MetalKernelCache,
    AsyncMatmulBatch, flash_attention_metal, fused_rms_norm_silu_metal,
    matmul_f32_metal, matmul_q4_0_metal, softmax_f32_metal,
    MultiGpuMetalContext,
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

pub use quantization::{
    q4_arrow_column, quantized_weight_schema,
};

// ─── Runtime ──────────────────────────────────────────────────────────────────

pub use runtime::{TitanRuntimeMode, TitanRuntimePlan};

// ─── Sampler ──────────────────────────────────────────────────────────────────

pub use sampler::{sample_token, SamplerConfig};

// ─── Scheduler ────────────────────────────────────────────────────────────────

pub use scheduler::{GpuTask, Priority, SchedulerStatus, TitanScheduler};

// ─── Tensor ───────────────────────────────────────────────────────────────────

pub use tensor::{BlockQ4_0, BlockQ8_0, QuantizationFormat, QuantizedTensorLayout, TensorShape};

// ─── Async Runtime ────────────────────────────────────────────────────────────

pub use async_runtime::{TitanExecutor, TaskHandle, WorkStealingScheduler};

// ─── Distributed ────────────────────────────────────────────────────────────

pub use distributed::{DistributedConfig, TitanCommunicator, all_reduce, broadcast};

// ─── Memory ───────────────────────────────────────────────────────────────────

pub use memory::{MemoryPool, UnifiedMemoryAllocator};
#[cfg(feature = "cuda")]
pub use memory::DeviceMemory;

// ─── Profiling ────────────────────────────────────────────────────────────────

pub use profiling::{KernelProfiler, PerformanceCounter, TimingReport};

/// Titan engine version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Check if the ultimate feature set is enabled
pub fn is_ultimate_build() -> bool {
    cfg!(feature = "ultimate")
}

/// Get a summary of enabled features
pub fn enabled_features() -> Vec<&'static str> {
    let mut features = vec![];
    if cfg!(feature = "metal") {
        features.push("metal");
    }
    if cfg!(feature = "cuda") {
        features.push("cuda");
    }
    if cfg!(feature = "cpp") {
        features.push("cpp");
    }
    if cfg!(feature = "python") {
        features.push("python");
    }
    if cfg!(feature = "async-runtime") {
        features.push("async-runtime");
    }
    if cfg!(feature = "distributed") {
        features.push("distributed");
    }
    if cfg!(feature = "supercharged") {
        features.push("supercharged");
    }
    features
}
