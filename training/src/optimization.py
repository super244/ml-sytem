"""Training optimization layer v3.0 for AI-Factory.

Re-exports hardware detection and optimizers from `ai_factory.core.runtime`.
Training-specific helpers (`create_supercharged_trainer_config`, CLI) remain here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ai_factory.core.runtime.optimization import (
    AutoTuner,
    HardwareDetector,
    TrainingOptimizer,
)
from ai_factory.core.schemas import BackendType, HardwareProfile

if TYPE_CHECKING:
    from training.src.config import ExperimentConfig

logger = logging.getLogger(__name__)

__all__ = [
    "AutoTuner",
    "BackendType",
    "HardwareDetector",
    "HardwareProfile",
    "TrainingOptimizer",
    "create_supercharged_trainer_config",
    "print_hardware_summary",
]


def create_supercharged_trainer_config(
    base_config: "ExperimentConfig",
    hardware_override: HardwareProfile | None = None,
) -> "ExperimentConfig":
    """Create a supercharged-optimized training configuration."""
    from training.src.config import ExperimentConfig

    optimizer = TrainingOptimizer(hardware_override)
    optimizer.configure_torch()

    config_dict = base_config.to_dict()
    optimized_dict = optimizer.get_training_config(config_dict)

    optimized_dict["use_titan"] = optimizer.titan_config.enabled
    optimized_dict["flash_attention"] = optimizer.hardware.flash_attention
    optimized_dict["dynamic_batch_scaling"] = True

    optimized_config = ExperimentConfig.from_dict(optimized_dict)
    optimized_config.config_path = base_config.config_path

    logger.info("Created supercharged training configuration v3.0")
    return optimized_config


def print_hardware_summary() -> None:
    """Print a comprehensive hardware summary."""
    hardware = HardwareDetector.detect()
    optimizer = TrainingOptimizer(hardware)
    summary = optimizer.get_optimization_summary()

    print("=" * 70)
    print("TITAN v3.0 HARDWARE DETECTION & OPTIMIZATION SUMMARY")
    print("=" * 70)
    print(f"Platform:       {hardware.platform}")
    print(f"Device:         {hardware.device_name}")
    print(f"Backend:        {hardware.backend.upper()}")
    print(f"Memory:         {hardware.memory_gb:.1f} GB")
    print(f"Compute Units:  {hardware.compute_units}")
    print(f"Bandwidth:      {hardware.bandwidth_gbps:.0f} GB/s")
    print(f"Capability:     {hardware.capability_score:.0f}/100")
    print(f"Optimization:   {optimizer.optimization_level}")
    print("-" * 70)
    print("CAPABILITIES:")
    print(f"  FP16:           {'Yes' if hardware.supports_fp16 else 'No'}")
    print(f"  BF16:           {'Yes' if hardware.supports_bf16 else 'No'}")
    print(f"  TF32:           {'Yes' if hardware.supports_tf32 else 'No'}")
    print(f"  FP8:            {'Yes' if hardware.supports_fp8 else 'No'}")
    print(f"  FP4:            {'Yes' if hardware.supports_fp4 else 'No'}")
    print(f"  Tensor Cores:   {'Yes' if hardware.tensor_cores else 'No'}")
    print(f"  Matrix Cores:   {'Yes' if hardware.matrix_cores else 'No'}")
    print(f"  Flash Attention:{'Yes' if hardware.flash_attention else 'No'}")
    print(f"  Unified Memory: {'Yes' if hardware.unified_memory else 'No'}")
    print(f"  PyTorch 2.x:    {'Yes' if hardware.pytorch_compile else 'No'}")
    print(f"  Titan Engine:   {'Yes' if hardware.titan_available else 'No'}", end="")
    if hardware.titan_available:
        print(f" v{hardware.titan_version}")
    else:
        print()
    print("-" * 70)
    print("RECOMMENDATIONS:")
    recs = summary["recommendations"]
    print(f"  Batch Size:     {recs['batch_size']}")
    print(f"  Workers:        {recs['workers']}")
    print(f"  Mixed Prec:     {recs['mixed_precision']}")
    print(f"  Grad Checkpt:   {'Yes' if recs['gradient_checkpointing'] else 'No'}")
    print("=" * 70)


if __name__ == "__main__":
    print_hardware_summary()
