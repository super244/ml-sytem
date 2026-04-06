//! Titan Benchmark Harness v2.0
//!
//! Comprehensive benchmarking suite for all Titan kernels and operations.
//! Supports micro-benchmarks, end-to-end inference, and comparative analysis.

use anyhow::Result;
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;

use ai_factory_titan::{
    cpu_kernels,
    detect_hardware,
    HardwareCapabilities,
    KernelProfiler,
    MetalKernelCache,
    CudaKernelCache,
    detect_metal_capabilities,
    detect_cuda_gpus,
};

/// Benchmark configuration
pub struct BenchmarkConfig {
    pub warmup_iterations: usize,
    pub measurement_time: Duration,
    pub sample_size: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            measurement_time: Duration::from_secs(5),
            sample_size: 100,
        }
    }
}

/// CPU kernel benchmarks
fn bench_cpu_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_kernels");
    
    // Dot product benchmarks
    for size in [128, 512, 1024, 4096, 8192] {
        let a = vec![1.0f32; size];
        let b = vec![2.0f32; size];
        
        group.bench_with_input(
            BenchmarkId::new("dot_product", size),
            &size,
            |b, _| {
                b.iter(|| {
                    cpu_kernels::dot_f32(
                        black_box(&a),
                        black_box(&b),
                    ).unwrap()
                });
            },
        );
    }
    
    // Matrix multiplication benchmarks
    for size in [64, 128, 256, 512] {
        let a = vec![1.0f32; size * size];
        let b = vec![2.0f32; size * size];
        
        group.bench_with_input(
            BenchmarkId::new("matmul", size),
            &size,
            |b, _| {
                b.iter(|| {
                    cpu_kernels::matmul_f32(
                        black_box(&a),
                        black_box(size),
                        black_box(size),
                        black_box(&b),
                        black_box(size),
                    ).unwrap()
                });
            },
        );
    }
    
    // RMSNorm benchmarks
    for size in [512, 1024, 4096, 8192] {
        let input = vec![1.0f32; size];
        
        group.bench_with_input(
            BenchmarkId::new("rms_norm", size),
            &size,
            |b, _| {
                b.iter(|| {
                    cpu_kernels::rms_norm_f32(
                        black_box(&input),
                        black_box(1e-6),
                    )
                });
            },
        );
    }
    
    // Softmax benchmarks
    for size in [512, 1024, 4096, 8192] {
        let input = vec![1.0f32; size];
        
        group.bench_with_input(
            BenchmarkId::new("softmax", size),
            &size,
            |b, _| {
                b.iter(|| {
                    cpu_kernels::softmax_f32(black_box(&input))
                });
            },
        );
    }
    
    group.finish();
}

/// Metal kernel benchmarks (macOS only)
#[cfg(feature = "metal")]
fn bench_metal_kernels(c: &mut Criterion) {
    use ai_factory_titan::metal_kernels::{matmul_f32_metal, flash_attention_metal};
    
    let mut group = c.benchmark_group("metal_kernels");
    let mut cache = MetalKernelCache::new().expect("Failed to create Metal cache");
    
    let caps = detect_metal_capabilities();
    group.throughput(criterion::Throughput::Elements(1));
    group.sample_size(50);
    
    // Matrix multiplication
    for size in [64, 128, 256, 512, 1024] {
        let a = vec![1.0f32; size * size];
        let b = vec![2.0f32; size * size];
        
        group.bench_with_input(
            BenchmarkId::new("matmul", format!("{}x{}", size, size)),
            &size,
            |b, _| {
                b.iter(|| {
                    matmul_f32_metal(
                        &mut cache,
                        black_box(&a),
                        black_box(&b),
                        black_box(size),
                        black_box(size),
                        black_box(size),
                    ).expect("Metal matmul failed")
                });
            },
        );
    }
    
    // FlashAttention
    for seq_len in [64, 128, 256, 512] {
        let head_dim = 64;
        let q = vec![1.0f32; seq_len * head_dim];
        let k = vec![1.0f32; seq_len * head_dim];
        let v = vec![1.0f32; seq_len * head_dim];
        
        group.bench_with_input(
            BenchmarkId::new("flash_attention", seq_len),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    flash_attention_metal(
                        &mut cache,
                        black_box(&q),
                        black_box(&k),
                        black_box(&v),
                        black_box(seq_len),
                        black_box(head_dim),
                    ).expect("Metal FlashAttention failed")
                });
            },
        );
    }
    
    // Log capabilities
    println!("\nMetal Capabilities:");
    println!("  Device: {}", caps.device_name);
    println!("  GPU Cores: {}", caps.gpu_cores);
    println!("  Bandwidth: {:.0} GB/s", caps.bandwidth_gbps);
    
    group.finish();
}

#[cfg(not(feature = "metal"))]
fn bench_metal_kernels(_c: &mut Criterion) {
    println!("Metal benchmarks skipped (feature not enabled)");
}

/// CUDA kernel benchmarks (Linux only)
#[cfg(feature = "cuda")]
fn bench_cuda_kernels(c: &mut Criterion) {
    use ai_factory_titan::cuda_kernels::{
        matmul_tf32_cuda,
        matmul_fp16_tc_cuda,
        flash_attention_cuda,
        detect_cuda_gpus,
    };
    
    let mut group = c.benchmark_group("cuda_kernels");
    let cache = CudaKernelCache::new(0).expect("Failed to create CUDA cache");
    
    let gpus = detect_cuda_gpus().expect("Failed to detect CUDA GPUs");
    if let Some(gpu) = gpus.first() {
        println!("\nCUDA GPU: {} (CC {}.{})", 
            gpu.name,
            gpu.compute_capability.major,
            gpu.compute_capability.minor
        );
    }
    
    group.sample_size(50);
    
    // TF32 matrix multiplication
    for size in [128, 256, 512, 1024, 2048] {
        let a = vec![1.0f32; size * size];
        let b = vec![2.0f32; size * size];
        
        group.bench_with_input(
            BenchmarkId::new("matmul_tf32", format!("{}x{}", size, size)),
            &size,
            |b, _| {
                b.iter(|| {
                    matmul_tf32_cuda(
                        &cache,
                        black_box(&a),
                        black_box(&b),
                        black_box(size),
                        black_box(size),
                        black_box(size),
                    ).expect("CUDA TF32 matmul failed")
                });
            },
        );
    }
    
    // FP16 Tensor Core matrix multiplication
    for size in [128, 256, 512, 1024, 2048] {
        let a = vec![half::f16::from_f32(1.0); size * size];
        let b = vec![half::f16::from_f32(2.0); size * size];
        
        group.bench_with_input(
            BenchmarkId::new("matmul_fp16_tc", format!("{}x{}", size, size)),
            &size,
            |b, _| {
                b.iter(|| {
                    matmul_fp16_tc_cuda(
                        &cache,
                        black_box(&a),
                        black_box(&b),
                        black_box(size),
                        black_box(size),
                        black_box(size),
                    ).expect("CUDA FP16 TC matmul failed")
                });
            },
        );
    }
    
    group.finish();
}

#[cfg(not(feature = "cuda"))]
fn bench_cuda_kernels(_c: &mut Criterion) {
    println!("CUDA benchmarks skipped (feature not enabled)");
}

/// End-to-end inference benchmarks
fn bench_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference");
    
    // Simulate typical LLM inference patterns
    for batch_size in [1, 4, 8] {
        for seq_len in [128, 512, 2048] {
            group.bench_with_input(
                BenchmarkId::new("prefill", format!("batch{}_seq{}", batch_size, seq_len)),
                &(batch_size, seq_len),
                |b, _| {
                    b.iter(|| {
                        // Simulate prefill pass
                        let _ = black_box(seq_len * batch_size * 4096);
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Hardware detection and reporting
fn report_hardware() {
    println!("\n{}", "=".repeat(70));
    println!("TITAN BENCHMARK HARNESS v2.0");
    println!("{}", "=".repeat(70));
    
    let hardware = detect_hardware();
    println!("\nDetected Hardware:");
    println!("  Platform: {}", hardware.platform);
    println!("  Device: {}", hardware.device_name);
    println!("  Memory: {:.1} GB", hardware.memory_gb);
    println!("  Compute Units: {}", hardware.compute_units);
    
    #[cfg(feature = "metal")]
    {
        let caps = detect_metal_capabilities();
        println!("\nMetal Capabilities:");
        println!("  Generation: {:?}", caps.generation);
        println!("  GPU Cores: {}", caps.gpu_cores);
        println!("  Unified Memory: {}", caps.has_unified_memory);
        println!("  Bandwidth: {:.0} GB/s", caps.bandwidth_gbps);
    }
    
    #[cfg(feature = "cuda")]
    {
        if let Ok(gpus) = detect_cuda_gpus() {
            println!("\nCUDA GPUs:");
            for (i, gpu) in gpus.iter().enumerate() {
                println!("  [{}] {}", i, gpu.name);
                println!("       Compute Capability: {}.{}", 
                    gpu.compute_capability.major,
                    gpu.compute_capability.minor
                );
                println!("       Memory: {:.1} GB", gpu.memory_gb);
                println!("       Tensor Cores: Gen {}", gpu.tensor_core_version);
            }
        }
    }
    
    println!("\n{}", "=".repeat(70));
}

/// Main benchmark group
criterion_group!(
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(5))
        .sample_size(100)
        .warm_up_time(Duration::from_secs(1));
    targets = 
        bench_cpu_kernels,
        bench_metal_kernels,
        bench_cuda_kernels,
        bench_inference
);

criterion_main!(benches);

/// Custom runner that prints hardware info first
pub fn main() {
    report_hardware();
    
    // Run criterion benchmarks
    benches();
}
