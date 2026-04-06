#!/usr/bin/env python3
"""Titan Ultimate Training Script v2.0

Comprehensive training script with:
- Automatic hardware optimization (M5 Ultra, Blackwell, etc.)
- Distributed training with FSDP/DDP
- 8-bit optimizers and mixed precision
- PyTorch 2.x compile integration
- Flash Attention 2
- Gradient checkpointing
- LoRA/QLoRA fine-tuning

Usage:
    python -m training.scripts.train_ultimate --config configs/ultimate.yaml
    python -m training.scripts.train_ultimate --model meta-llama/Llama-2-7b --dataset math

Environment Variables:
    WORLD_SIZE: Number of processes for distributed training
    RANK: Process rank
    LOCAL_RANK: Local GPU rank
    MASTER_ADDR: Master node address for distributed training
    MASTER_PORT: Master node port
    CUDA_VISIBLE_DEVICES: GPU selection
    PYTORCH_CUDA_ALLOC_CONF: CUDA memory configuration
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Add project to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from training.src.data import create_dataloader  # noqa: E402
from training.src.modeling import load_model, load_tokenizer  # noqa: E402
from training.src.optimization import (  # noqa: E402
    HardwareDetector,
    OptimizerType,
    TrainingOptimizer,
    print_hardware_summary,
)
from training.src.tracking import ExperimentTracker  # noqa: E402
from training.src.trainer import TitanTrainer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_distributed() -> tuple[int, int, int]:
    """Initialize distributed training.

    Returns:
        Tuple of (world_size, rank, local_rank)
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        torch.cuda.set_device(local_rank)
        logger.info(f"Initialized distributed: rank={rank}/{world_size}, local_rank={local_rank}")

    return world_size, rank, local_rank


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Titan Ultimate Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Model name or path",
    )
    parser.add_argument(
        "--model_revision",
        type=str,
        default="main",
        help="Model revision (for HuggingFace)",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name or path",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Dataset split to use",
    )

    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size per device (auto if not specified)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps",
    )

    # Optimization arguments
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (auto-tuned if not specified)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adamw", "adamw_fused", "adamw_8bit", "lion", "sgd"],
        default="adamw_fused",
        help="Optimizer type",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["linear", "cosine", "constant"],
        default="linear",
        help="Learning rate scheduler",
    )

    # Precision arguments
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 mixed precision",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 mixed precision",
    )
    parser.add_argument(
        "--fp8",
        action="store_true",
        help="Use FP8 (requires Hopper/Blackwell)",
    )

    # Efficiency arguments
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--flash_attention",
        action="store_true",
        default=True,
        help="Use Flash Attention 2",
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Use PyTorch 2.x compile",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="max-autotune",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode",
    )

    # LoRA arguments
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA fine-tuning",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=64,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout",
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        help="Use QLoRA (4-bit quantization)",
    )

    # Distributed arguments
    parser.add_argument(
        "--distributed_strategy",
        type=str,
        default="ddp",
        choices=["ddp", "fsdp", "none"],
        help="Distributed training strategy",
    )
    parser.add_argument(
        "--fsdp_sharding",
        type=str,
        default="full_shard",
        choices=["full_shard", "shard_grad_op", "no_shard"],
        help="FSDP sharding strategy",
    )

    # Logging arguments
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="titan-training",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )

    # Hardware detection
    parser.add_argument(
        "--hardware_summary",
        action="store_true",
        help="Print hardware summary and exit",
    )
    parser.add_argument(
        "--auto_tune",
        action="store_true",
        default=True,
        help="Auto-tune hyperparameters for hardware",
    )

    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()

    # Print hardware summary if requested
    if args.hardware_summary:
        print_hardware_summary()
        return

    # Setup distributed training
    world_size, rank, local_rank = setup_distributed()
    is_main_process = rank == 0

    # Detect hardware and create optimizer
    hardware = HardwareDetector.detect()
    optimizer_wrapper = TrainingOptimizer(hardware)
    optimizer_wrapper.configure_torch()

    if is_main_process:
        logger.info(f"Hardware: {hardware.device_name}")
        logger.info(f"Optimization level: {optimizer_wrapper.optimization_level}")
        logger.info(f"World size: {world_size}")

    # Auto-tune batch size if not specified
    if args.batch_size is None and args.auto_tune:
        args.batch_size = hardware.recommended_batch_size
        if is_main_process:
            logger.info(f"Auto-tuned batch size: {args.batch_size}")

    # Auto-tune learning rate if not specified
    if args.learning_rate is None and args.auto_tune:
        args.learning_rate = optimizer_wrapper.autotuner.tune_learning_rate()
        if is_main_process:
            logger.info(f"Auto-tuned learning rate: {args.learning_rate:.2e}")

    # Apply automatic precision settings if not explicitly specified
    if not args.bf16 and not args.fp16 and not args.fp8:
        if hardware.supports_bf16:
            args.bf16 = True
        elif hardware.supports_fp16:
            args.fp16 = True
        if hardware.supports_fp8:
            args.fp8 = True

    # Load model and tokenizer
    if is_main_process:
        logger.info(f"Loading model: {args.model}")

    model_kwargs = {
        "torch_dtype": torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
        "use_flash_attention_2": args.flash_attention,
    }

    if args.use_qlora:
        model_kwargs["load_in_4bit"] = True
        model_kwargs["bnb_4bit_compute_dtype"] = torch.bfloat16 if args.bf16 else torch.float16
        model_kwargs["bnb_4bit_use_double_quant"] = True
        model_kwargs["bnb_4bit_quant_type"] = "nf4"

    model = load_model(args.model, **model_kwargs)
    tokenizer = load_tokenizer(args.model)

    # Apply LoRA if enabled
    if args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        if is_main_process:
            logger.info(f"Applied LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
            model.print_trainable_parameters()

    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if is_main_process:
            logger.info("Enabled gradient checkpointing")

    # Apply PyTorch compile
    if args.torch_compile and hardware.pytorch_compile:
        model = optimizer_wrapper.apply_model_optimizations(model)

    # Setup distributed
    if world_size > 1:
        if args.distributed_strategy == "fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

            model = FSDP(
                model,
                auto_wrap_policy=transformer_auto_wrap_policy,
                mixed_precision=torch.bfloat16 if args.bf16 else None,
                device_id=local_rank,
            )
        elif args.distributed_strategy == "ddp":
            model = DDP(model, device_ids=[local_rank])

    # Create dataloaders
    train_dataloader = create_dataloader(
        args.dataset,
        tokenizer,
        split=args.dataset_split,
        batch_size=args.batch_size,
        num_workers=hardware.recommended_workers,
    )

    # Setup optimizer
    optimizer_type = OptimizerType(args.optimizer.replace("_", "_").upper())
    optimizer = optimizer_wrapper.get_memory_efficient_optimizer(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer_type=optimizer_type,
    )

    # Setup learning rate scheduler
    total_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
    warmup_steps = min(args.warmup_steps, total_steps // 10)

    if args.scheduler == "linear":
        from transformers import get_linear_schedule_with_warmup

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    elif args.scheduler == "cosine":
        from transformers import get_cosine_schedule_with_warmup

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    else:
        scheduler = None

    # Setup experiment tracker
    tracker = ExperimentTracker(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "model": args.model,
            "hardware": hardware.to_dict(),
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "optimizer": args.optimizer,
            "precision": "fp8" if args.fp8 else ("bf16" if args.bf16 else "fp16" if args.fp16 else "fp32"),
        },
        enabled=is_main_process,
    )

    # Create trainer
    trainer = TitanTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp8" if args.fp8 else ("bf16" if args.bf16 else "fp16" if args.fp16 else None),
        tracker=tracker,
    )

    # Training loop
    if is_main_process:
        logger.info(f"Starting training for {args.num_epochs} epochs")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Warmup steps: {warmup_steps}")

    global_step = 0
    for epoch in range(args.num_epochs):
        if is_main_process:
            logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")

        epoch_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            loss = trainer.training_step(batch)
            epoch_loss += loss

            global_step += 1

            # Logging
            if global_step % args.logging_steps == 0 and is_main_process:
                avg_loss = epoch_loss / (step + 1)
                lr = scheduler.get_last_lr()[0] if scheduler else args.learning_rate
                logger.info(f"Step {global_step}/{total_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
                tracker.log(
                    {
                        "train/loss": avg_loss,
                        "train/learning_rate": lr,
                        "train/epoch": epoch,
                    },
                    step=global_step,
                )

            # Evaluation
            if global_step % args.eval_steps == 0:
                pass  # Add evaluation logic

            # Save checkpoint
            if global_step % args.save_steps == 0 and is_main_process:
                output_path = Path(args.output_dir) / f"checkpoint-{global_step}"
                trainer.save_checkpoint(output_path)
                logger.info(f"Saved checkpoint to {output_path}")

    # Final save
    if is_main_process:
        final_path = Path(args.output_dir) / "final"
        trainer.save_checkpoint(final_path)
        logger.info(f"Training complete! Final model saved to {final_path}")

    # Cleanup distributed
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
