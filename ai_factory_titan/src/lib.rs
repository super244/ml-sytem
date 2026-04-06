pub mod backend;
pub mod cpu_kernels;
#[cfg(feature = "cpp")]
pub mod cpp;
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

pub use backend::{BackendKind, TitanBackend};
pub use cpu_kernels::{
    dot_f32, matmul_f32, vec_add_f32, vec_mul_f32, rms_norm_f32, softmax_f32, silu_f32
};

#[cfg(feature = "cuda")]
pub use cuda_kernels::{CudaKernelCache, detect_cuda_gpus, GpuArchitecture, ComputeCapability};

#[cfg(feature = "metal")]
pub use metal_kernels::{detect_metal_capabilities, MetalCapabilities, MetalKernelCache, AppleSiliconGeneration};

pub use detect::{detect_hardware, HardwareProfile};
pub use engine::{CacheStrategy, NativeAcceleration, TitanEngineDescriptor};
pub use gguf::{parse_gguf_header, GgufHeader};
pub use kv_cache::{KvCache, KvCacheConfig, KvCacheStats};
pub use quantization::{q4_arrow_column, quantized_weight_schema};
pub use runtime::{TitanRuntimeMode, TitanRuntimePlan};
pub use sampler::{SamplerConfig, sample_token};
pub use scheduler::{GpuTask, TitanScheduler};
pub use tensor::{QuantizationFormat, QuantizedTensorLayout, TensorShape};
