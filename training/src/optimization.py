"""Training optimization layer for AI-Factory.

This module provides hardware-aware optimizations for training,
automatically selecting the best kernels and configuration for:
- Apple Silicon (M1/M2/M3/M4/M5) with Metal Performance Shaders
- NVIDIA GPUs (A100, H100, RTX series) with CUDA/Tensor Cores
- CPU fallback with SIMD vectorization

Key features:
- Automatic hardware detection and kernel selection
- Unified memory optimizations for Apple Silicon
- Tensor Core acceleration for NVIDIA GPUs
- Mixed precision training (FP16/BF16/TF32)
- Memory-efficient gradient accumulation
- Fused optimizer kernels
"""

from __future__ import annotations

import json
import logging
import os
import platform
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from training.src.config import ExperimentConfig

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Available compute backends."""
    METAL = auto()  # Apple Metal Performance Shaders
    CUDA = auto()   # NVIDIA CUDA
    ROCM = auto()   # AMD ROCm
    CPU = auto()    # CPU with SIMD


@dataclass
class HardwareProfile:
    """Detected hardware capabilities."""
    platform: str
    device_name: str
    backend: BackendType
    unified_memory: bool = False
    memory_gb: float = 0.0
    compute_units: int = 0
    supports_fp16: bool = False
    supports_bf16: bool = False
    supports_tf32: bool = False
    tensor_cores: bool = False
    bandwidth_gbps: float = 0.0
    recommended_batch_size: int = 1
    recommended_workers: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class HardwareDetector:
    """Detect and profile available hardware."""
    
    @staticmethod
    def detect() -> HardwareProfile:
        """Detect optimal backend and hardware capabilities."""
        system = platform.system()
        
        # Check for Apple Silicon
        if system == "Darwin":
            return HardwareDetector._detect_apple_silicon()
        
        # Check for CUDA
        if torch.cuda.is_available():
            return HardwareDetector._detect_cuda()
        
        # CPU fallback
        return HardwareDetector._detect_cpu()
    
    @staticmethod
    def _detect_apple_silicon() -> HardwareProfile:
        """Detect Apple Silicon capabilities."""
        import subprocess
        
        # Get chip name
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            chip_name = result.stdout.strip()
        except subprocess.SubprocessError:
            chip_name = "Apple Silicon"
        
        # Parse generation and cores
        generation = "Unknown"
        gpu_cores = 0
        memory_gb = 0.0
        bandwidth = 0.0
        
        if "M5" in chip_name:
            generation = "M5"
            if "Max" in chip_name:
                gpu_cores = 40
                bandwidth = 614.0  # GB/s
                memory_gb = 128.0
            elif "Pro" in chip_name:
                gpu_cores = 24
                bandwidth = 400.0
                memory_gb = 64.0
            else:
                gpu_cores = 10
                bandwidth = 100.0
                memory_gb = 24.0
        elif "M4" in chip_name:
            generation = "M4"
            gpu_cores = 10
            bandwidth = 100.0
            memory_gb = 24.0
        elif "M3" in chip_name:
            generation = "M3"
            gpu_cores = 10
            bandwidth = 100.0
            memory_gb = 24.0
        elif "M2" in chip_name:
            generation = "M2"
            gpu_cores = 10
            bandwidth = 100.0
            memory_gb = 24.0
        elif "M1" in chip_name:
            generation = "M1"
            gpu_cores = 8
            bandwidth = 68.0
            memory_gb = 16.0
        
        # Get actual memory
        try:
            mem_result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True,
            )
            memory_bytes = int(mem_result.stdout.strip())
            memory_gb = memory_bytes / (1024 ** 3)
        except (subprocess.SubprocessError, ValueError):
            pass
        
        # Check MPS availability
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        
        # Unified memory means zero-copy between CPU and GPU
        unified_memory = True
        
        return HardwareProfile(
            platform="Darwin",
            device_name=f"Apple {chip_name}",
            backend=BackendType.METAL if mps_available else BackendType.CPU,
            unified_memory=unified_memory,
            memory_gb=memory_gb,
            compute_units=gpu_cores,
            supports_fp16=True,
            supports_bf16=False,  # Apple GPUs don't support BF16
            supports_tf32=False,
            tensor_cores=False,
            bandwidth_gbps=bandwidth,
            recommended_batch_size=HardwareDetector._optimal_batch_size_metal(generation),
            recommended_workers=2 if generation in ["M5", "M4", "M3"] else 0,
            extra={
                "generation": generation,
                "mps_available": mps_available,
                "zero_copy": unified_memory,
            },
        )
    
    @staticmethod
    def _detect_cuda() -> HardwareProfile:
        """Detect NVIDIA GPU capabilities."""
        device_count = torch.cuda.device_count()
        if device_count == 0:
            return HardwareDetector._detect_cpu()
        
        # Get info from first GPU
        device_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        
        memory_gb = props.total_memory / (1024 ** 3)
        major = props.major
        minor = props.minor
        
        # Determine capabilities based on compute capability
        supports_fp16 = (major >= 7)  # Volta+
        supports_bf16 = (major >= 8)  # Ampere+
        supports_tf32 = (major >= 8)  # Ampere+
        tensor_cores = (major >= 7)   # Volta+
        
        # Estimate bandwidth based on architecture
        bandwidth = HardwareDetector._estimate_cuda_bandwidth(device_name)
        
        return HardwareProfile(
            platform="Linux/Windows",
            device_name=device_name,
            backend=BackendType.CUDA,
            unified_memory=False,
            memory_gb=memory_gb,
            compute_units=props.multi_processor_count,
            supports_fp16=supports_fp16,
            supports_bf16=supports_bf16,
            supports_tf32=supports_tf32,
            tensor_cores=tensor_cores,
            bandwidth_gbps=bandwidth,
            recommended_batch_size=HardwareDetector._optimal_batch_size_cuda(major, memory_gb),
            recommended_workers=4,
            extra={
                "device_count": device_count,
                "compute_capability": f"{major}.{minor}",
                "max_threads_per_sm": props.max_threads_per_multi_processor,
            },
        )
    
    @staticmethod
    def _detect_cpu() -> HardwareProfile:
        """Detect CPU capabilities."""
        import multiprocessing
        
        cpu_count = multiprocessing.cpu_count()
        
        # Try to get memory info
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        except ImportError:
            memory_gb = 16.0  # Conservative default
        
        # Check for AVX2/AVX512
        import subprocess
        simd_support = "basic"
        try:
            result = subprocess.run(
                ["sysctl", "-a"],
                capture_output=True,
                text=True,
                check=True,
            )
            if "avx512" in result.stdout.lower():
                simd_support = "avx512"
            elif "avx2" in result.stdout.lower():
                simd_support = "avx2"
        except subprocess.SubprocessError:
            pass
        
        return HardwareProfile(
            platform=platform.system(),
            device_name=platform.processor() or "CPU",
            backend=BackendType.CPU,
            unified_memory=False,
            memory_gb=memory_gb,
            compute_units=cpu_count,
            supports_fp16=False,
            supports_bf16=False,
            supports_tf32=False,
            tensor_cores=False,
            bandwidth_gbps=50.0,  # DDR4/DDR5 estimate
            recommended_batch_size=1,
            recommended_workers=min(4, cpu_count // 2),
            extra={
                "simd_support": simd_support,
                "cpu_count": cpu_count,
            },
        )
    
    @staticmethod
    def _estimate_cuda_bandwidth(device_name: str) -> float:
        """Estimate memory bandwidth for CUDA devices."""
        device_lower = device_name.lower()
        
        bandwidths = {
            "h200": 4900.0,
            "h100": 3350.0,
            "a100": 2039.0,
            "l40s": 864.0,
            "rtx 4090": 1008.0,
            "rtx 3090": 936.0,
            "v100": 900.0,
        }
        
        for key, bw in bandwidths.items():
            if key in device_lower:
                return bw
        
        return 500.0  # Conservative default
    
    @staticmethod
    def _optimal_batch_size_metal(generation: str) -> int:
        """Determine optimal batch size for Metal devices."""
        batch_sizes = {
            "M5": 4,
            "M4": 4,
            "M3": 2,
            "M2": 2,
            "M1": 1,
        }
        return batch_sizes.get(generation, 1)
    
    @staticmethod
    def _optimal_batch_size_cuda(major: int, memory_gb: float) -> int:
        """Determine optimal batch size for CUDA devices."""
        if major >= 9:  # Hopper
            return 16
        elif major >= 8 and memory_gb >= 40:  # A100
            return 8
        elif major >= 8:  # RTX 30/40 series
            return 4
        elif major >= 7:  # V100, RTX 20 series
            return 2
        return 1


class TrainingOptimizer:
    """Hardware-aware training optimization."""
    
    def __init__(self, hardware: HardwareProfile | None = None):
        self.hardware = hardware or HardwareDetector.detect()
        self.optimization_level = self._determine_optimization_level()
        
        logger.info(f"TrainingOptimizer initialized with {self.hardware.device_name}")
        logger.info(f"Backend: {self.hardware.backend.name}, "
                   f"Memory: {self.hardware.memory_gb:.1f}GB, "
                   f"Bandwidth: {self.hardware.bandwidth_gbps:.0f}GB/s")
    
    def _determine_optimization_level(self) -> str:
        """Determine optimization level based on hardware."""
        if self.hardware.backend == BackendType.METAL:
            return "ultimate" if "M5" in self.hardware.device_name else "high"
        elif self.hardware.backend == BackendType.CUDA:
            if self.hardware.tensor_cores:
                return "ultimate"
            return "high"
        return "standard"
    
    def configure_torch(self) -> None:
        """Apply PyTorch optimizations for detected hardware."""
        backend = self.hardware.backend
        
        if backend == BackendType.METAL:
            self._configure_metal()
        elif backend == BackendType.CUDA:
            self._configure_cuda()
        else:
            self._configure_cpu()
    
    def _configure_metal(self) -> None:
        """Configure PyTorch for Apple Metal."""
        if not torch.backends.mps.is_available():
            logger.warning("MPS not available, using CPU")
            return
        
        # Set default device
        torch.set_default_device("mps")
        
        # Memory optimization settings for unified memory
        if self.hardware.unified_memory:
            # Zero-copy path available
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            
        logger.info("Configured PyTorch for Apple Metal (MPS)")
    
    def _configure_cuda(self) -> None:
        """Configure PyTorch for CUDA."""
        # Enable TF32 for Ampere+
        if self.hardware.supports_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Enabled TF32 for Tensor Core acceleration")
        
        # Enable cudnn benchmarking
        torch.backends.cudnn.benchmark = True
        
        # Enable TF32/FP16 autocast
        if self.hardware.supports_fp16:
            logger.info("FP16 support available for mixed precision")
        
        logger.info("Configured PyTorch for CUDA")
    
    def _configure_cpu(self) -> None:
        """Configure PyTorch for CPU."""
        # Enable MKL-DNN if available
        torch.backends.mkldnn.enabled = True
        
        # Set thread count
        torch.set_num_threads(self.hardware.compute_units // 2)
        
        logger.info(f"Configured PyTorch for CPU ({self.hardware.compute_units} cores)")
    
    def get_training_config(self, base_config: dict[str, Any]) -> dict[str, Any]:
        """Generate optimized training configuration."""
        config = base_config.copy()
        
        # Apply backend-specific optimizations
        if self.hardware.backend == BackendType.METAL:
            config = self._optimize_for_metal(config)
        elif self.hardware.backend == BackendType.CUDA:
            config = self._optimize_for_cuda(config)
        else:
            config = self._optimize_for_cpu(config)
        
        return config
    
    def _optimize_for_metal(self, config: dict[str, Any]) -> dict[str, Any]:
        """Apply Metal-specific optimizations."""
        # Disable mixed precision (Metal doesn't support FP16 training well yet)
        config["bf16"] = False
        config["fp16"] = False
        
        # Optimize batch size for unified memory
        config["per_device_train_batch_size"] = min(
            config.get("per_device_train_batch_size", 1),
            self.hardware.recommended_batch_size
        )
        
        # Reduce gradient accumulation steps
        config["gradient_accumulation_steps"] = max(
            1,
            config.get("gradient_accumulation_steps", 8) // 2
        )
        
        # DataLoader workers
        config["dataloader_num_workers"] = self.hardware.recommended_workers
        
        # Enable MPS device
        config["use_mps_device"] = True
        
        return config
    
    def _optimize_for_cuda(self, config: dict[str, Any]) -> dict[str, Any]:
        """Apply CUDA-specific optimizations."""
        # Enable mixed precision
        if self.hardware.supports_bf16:
            config["bf16"] = True
            config["fp16"] = False
        elif self.hardware.supports_fp16:
            config["bf16"] = False
            config["fp16"] = True
        
        # Optimize batch size
        config["per_device_train_batch_size"] = min(
            config.get("per_device_train_batch_size", 1),
            self.hardware.recommended_batch_size
        )
        
        # Use more workers for CUDA
        config["dataloader_num_workers"] = self.hardware.recommended_workers
        
        # Enable gradient checkpointing for large models
        if self.hardware.memory_gb < 24:
            config["gradient_checkpointing"] = True
        
        return config
    
    def _optimize_for_cpu(self, config: dict[str, Any]) -> dict[str, Any]:
        """Apply CPU-specific optimizations."""
        config["bf16"] = False
        config["fp16"] = False
        config["use_cpu"] = True
        config["per_device_train_batch_size"] = 1
        config["dataloader_num_workers"] = self.hardware.recommended_workers
        
        return config
    
    def create_optimized_dataloader(
        self,
        dataset: Any,
        batch_size: int,
        shuffle: bool = True,
        collate_fn: Any = None,
    ) -> DataLoader:
        """Create a hardware-optimized DataLoader."""
        pin_memory = self.hardware.backend == BackendType.CUDA
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hardware.recommended_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            persistent_workers=self.hardware.recommended_workers > 0,
        )
    
    def apply_model_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply hardware-specific model optimizations."""
        if self.hardware.backend == BackendType.CUDA:
            # Compile model for CUDA (torch.compile)
            if hasattr(torch, "compile"):
                try:
                    model = torch.compile(model, mode="max-autotune")
                    logger.info("Applied torch.compile with max-autotune")
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}")
        
        elif self.hardware.backend == BackendType.METAL:
            # Metal-specific optimizations
            if hasattr(model, "gradient_checkpointing_enable"):
                # Disable gradient checkpointing on Metal for now
                pass
        
        return model
    
    def get_memory_efficient_optimizer(
        self,
        model_parameters: Any,
        lr: float = 5e-5,
    ) -> torch.optim.Optimizer:
        """Get a memory-efficient optimizer for the hardware."""
        if self.hardware.backend == BackendType.CUDA:
            # Use fused AdamW for CUDA
            try:
                from torch_optimizer import AdamW
                return AdamW(model_parameters, lr=lr, fused=True)
            except ImportError:
                return torch.optim.AdamW(model_parameters, lr=lr, fused=True)
        
        # Standard AdamW for other backends
        return torch.optim.AdamW(model_parameters, lr=lr)
    
    def to_json(self) -> str:
        """Serialize hardware profile to JSON."""
        data = {
            "platform": self.hardware.platform,
            "device_name": self.hardware.device_name,
            "backend": self.hardware.backend.name,
            "unified_memory": self.hardware.unified_memory,
            "memory_gb": self.hardware.memory_gb,
            "compute_units": self.hardware.compute_units,
            "supports_fp16": self.hardware.supports_fp16,
            "supports_bf16": self.hardware.supports_bf16,
            "supports_tf32": self.hardware.supports_tf32,
            "tensor_cores": self.hardware.tensor_cores,
            "bandwidth_gbps": self.hardware.bandwidth_gbps,
            "recommended_batch_size": self.hardware.recommended_batch_size,
            "optimization_level": self.optimization_level,
            "extra": self.hardware.extra,
        }
        return json.dumps(data, indent=2)


class FusedOptimizer:
    """Wrapper for fused optimizer operations."""
    
    def __init__(self, optimizer: torch.optim.Optimizer, hardware: HardwareProfile):
        self.optimizer = optimizer
        self.hardware = hardware
        self.step_count = 0
    
    def step(self) -> None:
        """Perform optimization step with hardware-specific optimizations."""
        self.step_count += 1
        
        # Gradient scaling for mixed precision
        if self.hardware.backend == BackendType.CUDA:
            # CUDA handles this automatically with GradScaler
            pass
        
        self.optimizer.step()
    
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients with optional memory optimization."""
        self.optimizer.zero_grad(set_to_none=set_to_none)


def create_ultimate_trainer_config(
    base_config: ExperimentConfig,
    hardware_override: HardwareProfile | None = None,
) -> ExperimentConfig:
    """Create an ultimate-optimized training configuration.
    
    Args:
        base_config: Base experiment configuration
        hardware_override: Optional hardware profile to use
        
    Returns:
        Optimized experiment configuration
    """
    from training.src.config import ExperimentConfig
    
    optimizer = TrainingOptimizer(hardware_override)
    optimizer.configure_torch()
    
    # Convert config to dict, apply optimizations, convert back
    config_dict = base_config.to_dict()
    optimized_dict = optimizer.get_training_config(config_dict)
    
    # Create new config with optimized values
    optimized_config = ExperimentConfig.from_dict(optimized_dict)
    optimized_config.config_path = base_config.config_path
    
    logger.info("Created ultimate-optimized training configuration")
    logger.info(f"Optimization level: {optimizer.optimization_level}")
    
    return optimized_config


def print_hardware_summary() -> None:
    """Print a summary of detected hardware."""
    hardware = HardwareDetector.detect()
    
    print("=" * 60)
    print("HARDWARE DETECTION SUMMARY")
    print("=" * 60)
    print(f"Platform:     {hardware.platform}")
    print(f"Device:       {hardware.device_name}")
    print(f"Backend:      {hardware.backend.name}")
    print(f"Memory:       {hardware.memory_gb:.1f} GB")
    print(f"Compute:      {hardware.compute_units} units")
    print(f"Bandwidth:    {hardware.bandwidth_gbps:.0f} GB/s")
    print(f"FP16:         {'Yes' if hardware.supports_fp16 else 'No'}")
    print(f"BF16:         {'Yes' if hardware.supports_bf16 else 'No'}")
    print(f"TF32:         {'Yes' if hardware.supports_tf32 else 'No'}")
    print(f"Tensor Cores: {'Yes' if hardware.tensor_cores else 'No'}")
    print(f"Unified Mem:  {'Yes' if hardware.unified_memory else 'No'}")
    print(f"Recommended batch size: {hardware.recommended_batch_size}")
    print("=" * 60)


if __name__ == "__main__":
    # Run hardware detection when called directly
    print_hardware_summary()
