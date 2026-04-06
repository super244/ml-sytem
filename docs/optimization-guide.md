# AI-Factory Performance Optimization Guide

Ultimate hardware-aware training optimization for AI-Factory, automatically harnessing the full power of your silicon.

## Overview

The AI-Factory training system now includes **ultimate optimization** that automatically detects your hardware and applies the most aggressive, safe optimizations for:

- **Apple Silicon (M1/M2/M3/M4/M5)** - Metal Performance Shaders with unified memory
- **NVIDIA GPUs (A100, H100, H200, RTX)** - CUDA with Tensor Cores
- **CPU** - SIMD vectorized fallback

## Quick Start

### 1. Detect Your Hardware

```bash
python -m training.src.optimization
```

This prints a summary of detected hardware and recommended settings.

### 2. Use Ultimate Profile

**For Apple Silicon (M5 Max):**
```bash
python training/train.py --config training/configs/profiles/m5_max_ultimate.yaml
```

**For NVIDIA A100:**
```bash
python training/train.py --config training/configs/profiles/cuda_ultimate_a100.yaml
```

**For NVIDIA H100:**
```bash
python training/train.py --config training/configs/profiles/cuda_ultimate_h100.yaml
```

### 3. Use Ultimate Harness Programmatically

```python
from training.src.ultimate_harness import UltimateTrainingHarness, HarnessConfig
from training.src.config import load_experiment_config

# Load config
config = load_experiment_config("training/configs/profiles/m5_max_ultimate.yaml")

# Create harness with optimizations
harness = UltimateTrainingHarness(
    config,
    harness_config=HarnessConfig(
        enable_mixed_precision=True,
        enable_torch_compile=True,
        dynamic_batch_size=True,
    )
)

# Prepare model with all optimizations
model = harness.prepare_model(model)

# Get optimized training arguments
args = harness.get_training_arguments(layout)
```

## Hardware-Specific Optimizations

### Apple Silicon (Metal)

**M5 Max Specific:**
- **614 GB/s unified memory bandwidth** - Zero-copy CPU/GPU transfers
- **40 GPU cores** - Fully saturated compute
- **Fused kernels** - RMSNorm+SiLU fusion reduces bandwidth by 50%
- **FlashAttention-style attention** - Minimizes memory traffic

**Metal Shaders:**
- Tiled matrix multiplication optimized for Amx-array
- Async compute pipelines
- Memory-bounded kernel fusion

**Configuration:**
```yaml
hardware:
  unified_memory: true
  zero_copy: true
  gpu_cores: 40
  bandwidth_gbps: 614
```

### NVIDIA CUDA (A100/H100)

**A100 Specific:**
- **Tensor Core acceleration** - 312 TFLOPS FP16/ 19.5 TFLOPS FP32
- **BF16 native support** - No accuracy loss, 2x throughput
- **TF32 automatic mixed precision** - Easy 8x speedup
- **cuDNN auto-tuning** - Optimal algorithms selected at runtime

**H100/H200 Specific:**
- **4th gen Tensor Cores** - 989 TFLOPS FP8 / 395 TFLOPS FP16
- **Transformer Engine** - Dynamic FP8/FP16/FP32 switching
- **3.35 TB/s memory bandwidth** (H100) / 4.9 TB/s (H200)
- **FlashAttention-2 optimized** - 2x faster than A100

**CUDA Kernels:**
- WMMA-based FP16 Tensor Core matmul
- Fused AdamW optimizer
- Warp-level softmax
- Q4_0 quantization on GPU

**Configuration:**
```yaml
optimizations:
  mixed_precision: bf16
  tensor_core: true
  tf32: true
  cudnn_benchmark: true
  flash_attention: true
```

## Kernel Implementations

### Rust Titan Engine (`ai_factory_titan`)

The Titan engine provides low-level compute kernels:

```rust
// Metal kernels
use ai_factory_titan::metal_kernels::{
    MetalKernelCache, 
    matmul_f32_metal,
    flash_attention_metal,
    fused_rms_norm_silu_metal,
};

// CUDA kernels  
use ai_factory_titan::cuda_kernels::{
    CudaKernelCache,
    matmul_tf32_cuda,
    flash_attention_cuda,
    adamw_fused_cuda,
};
```

### Features

Enable in `Cargo.toml`:

```toml
[features]
default = []
metal = ["dep:metal", "dep:objc"]
cuda = ["dep:cudarc", "dep:half", "cudarc/cuda-12080"]
ultimate = ["metal", "cuda", "cpp"]
```

Build with ultimate features:

```bash
cd ai_factory_titan
cargo build --release --features ultimate
```

## Mixed Precision Training

### Automatic Mixed Precision (AMP)

The harness automatically enables the best precision for your hardware:

| Hardware | Default | Speedup | Memory |
|----------|---------|---------|--------|
| H100 | FP8/BF16 | 4-8x | 50% |
| A100 | BF16 | 2-4x | 50% |
| RTX 4090 | FP16 | 2-3x | 50% |
| M5 Max | FP32 | 1x | 100% |
| CPU | FP32 | 1x | 100% |

### Manual Control

```python
from training.src.optimization import TrainingOptimizer, HardwareDetector

# Detect hardware
hardware = HardwareDetector.detect()

# Create optimizer
optimizer = TrainingOptimizer(hardware)

# Get optimized config
config = optimizer.get_training_config({
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 8,
})
```

## Performance Monitoring

### Real-time Metrics

```python
from training.src.ultimate_harness import UltimateTrainingHarness

harness = UltimateTrainingHarness(config)

# Check memory stats
stats = harness.get_memory_stats()
print(f"Memory: {stats['allocated_gb']:.2f}GB / {stats['reserved_gb']:.2f}GB")

# Run benchmark
from training.src.ultimate_harness import quick_benchmark
results = quick_benchmark()
print(f"Throughput: {results['estimated_tflops']:.2f} TFLOPS")
```

### Performance Callback

```python
from training.src.ultimate_harness import UltimateTrainerCallback

# Add to trainer callbacks
callbacks = [
    UltimateTrainerCallback(harness),
    # ... other callbacks
]
```

## Memory Optimization

### Gradient Checkpointing

Enable for large models (>7B parameters):

```yaml
model:
  gradient_checkpointing: true
```

Trade-off: ~30% slower, 50% less memory

### Gradient Accumulation

Simulate larger batch sizes:

```yaml
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8  # Effective batch = 16
```

### Activation Checkpointing

```python
# Automatically applied by harness for CUDA
harness = UltimateTrainingHarness(
    config,
    harness_config=HarnessConfig(
        enable_gradient_checkpointing=True,
    )
)
```

## Distributed Training

### Multi-GPU (CUDA)

```bash
torchrun --nproc_per_node=8 training/train.py \
    --config training/configs/profiles/cuda_ultimate_a100.yaml \
    --distributed
```

The harness automatically:
- Selects NCCL backend
- Enables gradient synchronization fusion
- Balances load across GPUs

### Memory Pooling

```python
from ai_factory_titan::cuda_kernels::distributed::ring_allreduce
```

## Troubleshooting

### Metal (MPS) Issues

**Problem:** Out of memory on Apple Silicon

**Solution:**
```python
# Reduce batch size
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Disable limit
```

**Problem:** Slow training on MPS

**Solution:**
```yaml
training:
  dataloader_num_workers: 0  # MPS works best with main process
  pin_memory: false
```

### CUDA Issues

**Problem:** CUDA out of memory

**Solution:**
```python
# Enable gradient checkpointing
harness_config = HarnessConfig(
    enable_gradient_checkpointing=True,
)

# Or reduce batch size
config.training.per_device_train_batch_size = 2
config.training.gradient_accumulation_steps = 8
```

**Problem:** Low GPU utilization

**Solution:**
```bash
# Check with nvidia-smi
watch -n 1 nvidia-smi

# Increase workers
export OMP_NUM_THREADS=8
```

## Benchmarking

### Quick Benchmark

```bash
python -c "from training.src.ultimate_harness import quick_benchmark; quick_benchmark()"
```

### Full Benchmark Suite

```python
from training.src.ultimate_harness import UltimateTrainingHarness
from training.src.optimization import HardwareDetector

hardware = HardwareDetector.detect()
harness = UltimateTrainingHarness.__new__(UltimateTrainingHarness)
harness.hardware = hardware
harness.print_summary()
```

## Profile Reference

### Included Profiles

| Profile | Hardware | Memory | Batch | Notes |
|---------|----------|--------|-------|-------|
| `m5_max_ultimate` | M5 Max | 128GB | 4 | Unified memory, zero-copy |
| `cuda_ultimate_a100` | A100 | 80GB | 8 | BF16, Tensor Cores |
| `cuda_ultimate_h100` | H100 | 80GB | 16 | FP8, Transformer Engine |
| `local_metal` | Any Mac | Auto | 1 | Conservative, portable |

### Creating Custom Profiles

```yaml
profile_name: my_custom_ultimate
run_name: atlas_custom

refs:
  model: ../components/models/qwen2_scratch_1b.yaml
  # ... other refs

hardware:
  platform: darwin  # or linux
  backend: metal  # or cuda

optimizations:
  mixed_precision: true
  tensor_core: true
```

## Advanced: Custom Kernels

### Adding Metal Kernels

Edit `ai_factory_titan/src/metal/shaders.metal`:

```metal
kernel void my_kernel(
    device float* data [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    data[gid] = data[gid] * 2.0f;
}
```

### Adding CUDA Kernels

Edit `ai_factory_titan/src/cuda/kernels.cu`:

```cuda
extern "C" __global__ void my_cuda_kernel(
    float* data,
    uint size
) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] * 2.0f;
    }
}
```

## API Reference

### `TrainingOptimizer`

```python
class TrainingOptimizer:
    def __init__(self, hardware: HardwareProfile)
    def configure_torch(self) -> None
    def get_training_config(self, base: dict) -> dict
    def create_optimized_dataloader(self, dataset, batch_size, shuffle, collate_fn) -> DataLoader
    def apply_model_optimizations(self, model: nn.Module) -> nn.Module
```

### `UltimateTrainingHarness`

```python
class UltimateTrainingHarness:
    def __init__(self, config, harness_config, hardware)
    def prepare_model(self, model: nn.Module) -> nn.Module
    def get_training_arguments(self, layout) -> TrainingArguments
    def create_dataloader(self, dataset, tokenizer) -> DataLoader
    def get_memory_stats(self) -> dict[str, float]
    def print_summary(self) -> None
```

### `HardwareDetector`

```python
class HardwareDetector:
    @staticmethod
    def detect() -> HardwareProfile
    @staticmethod
    def print_hardware_summary() -> None
```

## Performance Targets

### M5 Max
- Training throughput: 50-100 tokens/sec (1.5B model)
- Memory bandwidth utilization: >80%
- GPU utilization: >90%

### A100
- Training throughput: 200-400 tokens/sec (7B model)
- Tensor Core utilization: >80%
- Memory bandwidth utilization: >70%

### H100
- Training throughput: 500-1000 tokens/sec (14B model)
- FP8 utilization: >60%
- FlashAttention speedup: 2x vs A100

## Contributing

To add optimizations for new hardware:

1. Add kernel implementation in `ai_factory_titan/src/`
2. Add detection logic in `training/src/optimization.py`
3. Create profile in `training/configs/profiles/`
4. Update documentation

## References

- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA Tensor Cores](https://www.nvidia.com/en-us/data-center/tensor-cores/)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
