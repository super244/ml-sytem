"""Ultimate Training Harness for AI-Factory.

This module provides the ultimate training harness that automatically:
- Detects hardware capabilities (Apple Silicon, NVIDIA GPUs)
- Selects optimal kernel backends (Metal, CUDA, CPU)
- Configures mixed precision training (FP16/BF16/TF32)
- Applies memory-efficient optimizations
- Monitors performance and adjusts dynamically

The harness integrates with the Titan engine for maximum performance.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from transformers import TrainerCallback, TrainingArguments

from training.src.optimization import (
    BackendType,
    HardwareDetector,
    HardwareProfile,
    TrainingOptimizer,
)

if TYPE_CHECKING:
    from training.src.config import ExperimentConfig
    from training.src.tracking import Tracker

logger = logging.getLogger(__name__)


@dataclass
class HarnessConfig:
    """Configuration for the ultimate training harness."""
    enable_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = False
    enable_torch_compile: bool = True
    enable_memory_profiling: bool = False
    dynamic_batch_size: bool = True
    auto_recovery: bool = True
    max_retries: int = 3
    performance_log_interval: int = 100
    extra: dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """Monitor training performance and resource utilization."""
    
    def __init__(self, hardware: HardwareProfile, log_interval: int = 100):
        self.hardware = hardware
        self.log_interval = log_interval
        self.step_count = 0
        self.start_time = time.time()
        self.step_times: list[float] = []
        self.memory_history: list[tuple[int, float]] = []
    
    def on_step_begin(self) -> None:
        """Call at the beginning of each training step."""
        self.step_start = time.time()
    
    def on_step_end(self) -> None:
        """Call at the end of each training step."""
        step_time = time.time() - self.step_start
        self.step_times.append(step_time)
        self.step_count += 1
        
        # Track memory
        if self.hardware.backend == BackendType.CUDA:
            memory_used = torch.cuda.memory_allocated() / (1024**3)  # GB
            self.memory_history.append((self.step_count, memory_used))
        
        # Log performance periodically
        if self.step_count % self.log_interval == 0:
            self._log_performance()
    
    def _log_performance(self) -> None:
        """Log current performance metrics."""
        if not self.step_times:
            return
        
        avg_step_time = sum(self.step_times[-self.log_interval:]) / min(
            len(self.step_times), self.log_interval
        )
        throughput = 1.0 / avg_step_time if avg_step_time > 0 else 0
        
        logger.info(
            f"Step {self.step_count}: "
            f"{throughput:.2f} steps/sec, "
            f"{avg_step_time*1000:.1f}ms/step"
        )
        
        if self.memory_history:
            recent_memory = self.memory_history[-1][1]
            logger.info(f"Memory: {recent_memory:.2f}GB / {self.hardware.memory_gb:.1f}GB")


class UltimateTrainingHarness:
    """Ultimate training harness with hardware-aware optimizations."""
    
    def __init__(
        self,
        config: ExperimentConfig,
        harness_config: HarnessConfig | None = None,
        hardware: HardwareProfile | None = None,
    ):
        self.config = config
        self.harness_config = harness_config or HarnessConfig()
        self.hardware = hardware or HardwareDetector.detect()
        self.optimizer = TrainingOptimizer(self.hardware)
        self.performance_monitor = PerformanceMonitor(
            self.hardware,
            self.harness_config.performance_log_interval,
        )
        
        # Mixed precision state
        self.scaler: GradScaler | None = None
        self.autocast_enabled = False
        
        # Setup
        self._setup()
    
    def _setup(self) -> None:
        """Initialize the harness."""
        logger.info("Initializing Ultimate Training Harness")
        logger.info(f"Hardware: {self.hardware.device_name}")
        logger.info(f"Backend: {self.hardware.backend.name}")
        
        # Configure PyTorch
        self.optimizer.configure_torch()
        
        # Setup mixed precision
        if self.harness_config.enable_mixed_precision:
            self._setup_mixed_precision()
        
        # Log configuration
        self._log_configuration()
    
    def _setup_mixed_precision(self) -> None:
        """Setup mixed precision training."""
        if self.hardware.backend == BackendType.CUDA:
            if self.hardware.supports_fp16 or self.hardware.supports_bf16:
                self.scaler = GradScaler()
                self.autocast_enabled = True
                dtype = torch.bfloat16 if self.hardware.supports_bf16 else torch.float16
                logger.info(f"Enabled mixed precision ({dtype})")
        
        elif self.hardware.backend == BackendType.METAL:
            # Metal doesn't support FP16 training well yet
            logger.info("Mixed precision disabled for Metal")
            self.autocast_enabled = False
    
    def _log_configuration(self) -> None:
        """Log harness configuration."""
        config_summary = {
            "mixed_precision": self.autocast_enabled,
            "gradient_checkpointing": self.harness_config.enable_gradient_checkpointing,
            "torch_compile": self.harness_config.enable_torch_compile,
            "dynamic_batch_size": self.harness_config.dynamic_batch_size,
        }
        logger.info(f"Harness config: {config_summary}")
    
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare model for training with optimizations."""
        logger.info("Preparing model with ultimate optimizations")
        
        # Apply model optimizations
        model = self.optimizer.apply_model_optimizations(model)
        
        # Enable gradient checkpointing if configured
        if self.harness_config.enable_gradient_checkpointing:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing")
        
        # Move to device
        device = self._get_device()
        model = model.to(device)
        
        # Compile if enabled and supported
        if self.harness_config.enable_torch_compile:
            model = self._compile_model(model)
        
        return model
    
    def _get_device(self) -> torch.device:
        """Get the optimal device for training."""
        if self.hardware.backend == BackendType.CUDA:
            return torch.device("cuda:0")
        elif self.hardware.backend == BackendType.METAL:
            return torch.device("mps")
        return torch.device("cpu")
    
    def _compile_model(self, model: nn.Module) -> nn.Module:
        """Compile model with optimal settings."""
        if hasattr(torch, "compile"):
            try:
                mode = "max-autotune" if self.hardware.backend == BackendType.CUDA else "reduce-overhead"
                compiled_model = torch.compile(model, mode=mode)
                logger.info(f"Compiled model with mode={mode}")
                return compiled_model
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
        return model
    
    def create_dataloader(self, dataset: Any, tokenizer: Any) -> Any:
        """Create optimized dataloader."""
        from training.src.collators import WeightedDataCollator
        
        batch_size = self.config.training.per_device_train_batch_size
        
        collator = WeightedDataCollator(
            tokenizer=tokenizer,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        )
        
        return self.optimizer.create_optimized_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
        )
    
    def get_training_arguments(self, layout: Any) -> TrainingArguments:
        """Generate optimized training arguments."""
        # Get base optimized config
        base_dict = self._build_training_args_dict(layout)
        optimized_dict = self.optimizer.get_training_config(base_dict)
        
        # Apply harness-specific optimizations
        if self.harness_config.enable_torch_compile:
            optimized_dict["torch_compile"] = True
        
        # Filter to only valid TrainingArguments parameters
        import inspect
        valid_params = set(inspect.signature(TrainingArguments.__init__).parameters)
        filtered_dict = {k: v for k, v in optimized_dict.items() if k in valid_params}
        
        return TrainingArguments(**filtered_dict)
    
    def _build_training_args_dict(self, layout: Any) -> dict[str, Any]:
        """Build base training arguments dictionary."""
        training = self.config.training
        
        return {
            "output_dir": str(layout.checkpoints_dir),
            "run_name": self.config.run_name,
            "num_train_epochs": training.num_train_epochs,
            "max_steps": training.max_steps,
            "learning_rate": training.learning_rate,
            "weight_decay": training.weight_decay,
            "warmup_ratio": training.warmup_ratio,
            "lr_scheduler_type": training.lr_scheduler_type,
            "per_device_train_batch_size": training.per_device_train_batch_size,
            "per_device_eval_batch_size": training.per_device_eval_batch_size,
            "gradient_accumulation_steps": training.gradient_accumulation_steps,
            "max_grad_norm": training.max_grad_norm,
            "logging_steps": training.logging_steps,
            "eval_steps": training.eval_steps,
            "save_steps": training.save_steps,
            "save_total_limit": training.save_total_limit,
            "bf16": False,
            "fp16": False,
            "save_strategy": training.save_strategy,
            "load_best_model_at_end": training.load_best_model_at_end,
            "report_to": self.config.logging.report_to,
            "dataloader_num_workers": training.dataloader_num_workers,
            "remove_unused_columns": False,
            "optim": "adamw_torch",
            "logging_dir": str(layout.logs_dir),
            "logging_first_step": self.config.logging.logging_first_step,
            "group_by_length": training.group_by_length,
            "save_safetensors": training.save_safetensors,
            "seed": self.config.seed,
            "deepspeed": self.config.runtime.deepspeed_config,
            "torch_compile": False,
        }
    
    def wrap_forward(self, forward_fn: Callable) -> Callable:
        """Wrap forward pass with autocast if enabled."""
        if not self.autocast_enabled:
            return forward_fn
        
        def wrapped_forward(*args, **kwargs):
            dtype = torch.bfloat16 if self.hardware.supports_bf16 else torch.float16
            with autocast(device_type=self._get_device().type, dtype=dtype, enabled=True):
                return forward_fn(*args, **kwargs)
        
        return wrapped_forward
    
    def training_step_hook(self) -> None:
        """Call at the beginning of each training step."""
        self.performance_monitor.on_step_begin()
    
    def post_step_hook(self) -> None:
        """Call at the end of each training step."""
        self.performance_monitor.on_step_end()
    
    def save_checkpoint(self, trainer: Any, path: Path) -> None:
        """Save optimized checkpoint."""
        trainer.save_model(path)
        
        # Save harness state
        harness_state = {
            "hardware_profile": {
                "device_name": self.hardware.device_name,
                "backend": self.hardware.backend.name,
            },
            "performance_stats": {
                "steps": self.performance_monitor.step_count,
                "avg_step_time_ms": (
                    sum(self.performance_monitor.step_times) / len(self.performance_monitor.step_times) * 1000
                    if self.performance_monitor.step_times else 0
                ),
            },
        }
        
        state_path = path / "harness_state.json"
        with open(state_path, "w") as f:
            json.dump(harness_state, f, indent=2)
        
        logger.info(f"Saved checkpoint with harness state to {path}")
    
    def get_memory_stats(self) -> dict[str, float]:
        """Get current memory statistics."""
        stats = {}
        
        if self.hardware.backend == BackendType.CUDA:
            stats["allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
            stats["reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
            stats["max_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
        
        elif self.hardware.backend == BackendType.METAL:
            # Metal memory tracking is limited
            stats["allocated_gb"] = 0.0  # Would need IOKit access
        
        return stats
    
    def print_summary(self) -> None:
        """Print training harness summary."""
        print("=" * 70)
        print("ULTIMATE TRAINING HARNESS SUMMARY")
        print("=" * 70)
        print(f"Hardware: {self.hardware.device_name}")
        print(f"Backend:  {self.hardware.backend.name}")
        print(f"Memory:   {self.hardware.memory_gb:.1f} GB")
        print(f"Compute:  {self.hardware.compute_units} units")
        print(f"Bandwidth: {self.hardware.bandwidth_gbps:.0f} GB/s")
        print("-" * 70)
        print("Configuration:")
        print(f"  Mixed Precision: {self.autocast_enabled}")
        print(f"  Gradient Checkpointing: {self.harness_config.enable_gradient_checkpointing}")
        print(f"  Torch Compile: {self.harness_config.enable_torch_compile}")
        print(f"  Dynamic Batch Size: {self.harness_config.dynamic_batch_size}")
        print("=" * 70)


class UltimateTrainerCallback(TrainerCallback):
    """Callback to integrate harness with transformers Trainer."""
    
    def __init__(self, harness: UltimateTrainingHarness):
        self.harness = harness
    
    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.harness.training_step_hook()
    
    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.harness.post_step_hook()
    
    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        # Log memory stats on save
        memory_stats = self.harness.get_memory_stats()
        if memory_stats:
            logger.info(f"Memory at save: {memory_stats}")


def build_ultimate_trainer_with_harness(
    config: ExperimentConfig,
    model: nn.Module,
    args: TrainingArguments,
    train_dataset: Any,
    eval_dataset: Any | None,
    data_collator: Any,
    tokenizer: Any,
    callbacks: list[Any],
    layout: Any,
) -> Any:
    """Build ultimate trainer with full optimization harness.
    
    Args:
        config: Experiment configuration
        model: Model to train
        args: Training arguments
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        data_collator: Data collator
        tokenizer: Tokenizer
        callbacks: Additional callbacks
        layout: Run layout
        
    Returns:
        Optimized trainer instance
    """
    from training.src.trainer import MathTrainer
    
    # Create harness
    harness_config = HarnessConfig(
        enable_mixed_precision=True,
        enable_gradient_checkpointing=config.model.use_4bit or config.model.use_8bit,
        enable_torch_compile=config.runtime.torch_compile,
    )
    harness = UltimateTrainingHarness(config, harness_config)
    
    # Print harness summary
    harness.print_summary()
    
    # Prepare model
    model = harness.prepare_model(model)
    
    # Add harness callback
    harness_callback = UltimateTrainerCallback(harness)
    all_callbacks = callbacks + [harness_callback]
    
    # Determine if we need preference optimization trainer
    if config.preference.enabled:
        from training.src.trainer import build_ultimate_trainer
        # Use existing build_ultimate_trainer for preference optimization
        return build_ultimate_trainer(
            config=config,
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=all_callbacks,
        )
    
    # Build standard trainer with harness
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "sequential_training": (config.data.curriculum_learning and config.data.sequential_curriculum),
        "callbacks": all_callbacks,
    }
    
    # Attach tokenizer/processor
    import inspect
    _TRAINER_PARAMETERS = set(inspect.signature(MathTrainer.__init__).parameters)
    
    if "processing_class" in _TRAINER_PARAMETERS:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in _TRAINER_PARAMETERS:
        trainer_kwargs["tokenizer"] = tokenizer
    
    return MathTrainer(**trainer_kwargs)


def quick_benchmark(hardware: HardwareProfile | None = None) -> dict[str, float]:
    """Quick benchmark to verify optimizations are working.
    
    Returns benchmark results including throughput metrics.
    """
    hardware = hardware or HardwareDetector.detect()
    
    print("Running quick benchmark...")
    
    # Simple matmul benchmark
    device = torch.device("cpu")
    if hardware.backend == BackendType.CUDA:
        device = torch.device("cuda:0")
    elif hardware.backend == BackendType.METAL:
        device = torch.device("mps")
    
    # Create test tensors
    size = 2048
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Warmup
    for _ in range(10):
        _ = torch.matmul(a, b)
    
    if hardware.backend == BackendType.CUDA:
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    iterations = 100
    
    for _ in range(iterations):
        _ = torch.matmul(a, b)
        if hardware.backend == BackendType.CUDA:
            torch.cuda.synchronize()
    
    elapsed = time.time() - start
    
    # Calculate metrics
    flops_per_matmul = 2 * size ** 3  # 2 * N^3 for matmul
    total_flops = flops_per_matmul * iterations
    tflops = total_flops / elapsed / 1e12
    
    results = {
        "device": str(device),
        "matrix_size": size,
        "iterations": iterations,
        "total_time_sec": elapsed,
        "avg_time_ms": elapsed / iterations * 1000,
        "estimated_tflops": tflops,
    }
    
    print(f"Benchmark results: {results}")
    return results


if __name__ == "__main__":
    # Run harness diagnostic
    harness = UltimateTrainingHarness.__new__(UltimateTrainingHarness)
    harness.hardware = HardwareDetector.detect()
    harness.harness_config = HarnessConfig()
    harness.optimizer = TrainingOptimizer(harness.hardware)
    harness.performance_monitor = PerformanceMonitor(harness.hardware)
    harness.autocast_enabled = False
    harness.scaler = None
    
    harness.print_summary()
    quick_benchmark(harness.hardware)
