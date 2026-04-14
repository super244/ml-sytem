#!/usr/bin/env python3
"""
Ultimate Teacher-Student Training Launcher

This script launches the ultimate upgraded training with:
- Teacher-student knowledge distillation using Qwen2.5-7B
- Full parameter training (not just LoRA)
- Enhanced dataset integration
- Ultimate optimization harness
"""

import argparse
import logging
import sys
from pathlib import Path

# Add training src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from training.train import main as train_main


def parse_args():
    parser = argparse.ArgumentParser(description="Launch ultimate teacher-student training with Qwen2.5-7B teacher")
    parser.add_argument(
        "--config",
        default="training/configs/ultimate_teacher_student.yaml",
        help="Path to the ultimate teacher-student config file",
    )
    parser.add_argument("--resume-from-checkpoint", default=None, help="Resume from specific checkpoint")
    parser.add_argument("--resume-from-latest-checkpoint", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Validate configs and setup without training")
    parser.add_argument("--validate-model-load", action="store_true", help="Load model after validation and exit")
    parser.add_argument("--distributed", action="store_true", help="Launch in distributed mode")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs for distributed training")
    parser.add_argument(
        "--teacher-model", default="Qwen/Qwen2.5-7B-Instruct", help="Teacher model for knowledge distillation"
    )
    parser.add_argument("--temperature", type=float, default=2.0, help="Temperature for knowledge distillation")
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight for teacher loss vs student loss")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)

    logger.info("🚀 Launching ULTIMATE Teacher-Student Training!")
    logger.info(f"📚 Config: {args.config}")
    logger.info(f"👨‍🏫 Teacher Model: {args.teacher_model}")
    logger.info(f"🌡️ Temperature: {args.temperature}")
    logger.info(f"⚖️ Alpha: {args.alpha}")

    # Override sys.argv for train_main
    sys.argv = [
        "train.py",
        "--config",
        args.config,
    ]

    if args.resume_from_checkpoint:
        sys.argv.extend(["--resume-from-checkpoint", args.resume_from_checkpoint])
    if args.resume_from_latest_checkpoint:
        sys.argv.append("--resume-from-latest-checkpoint")
    if args.dry_run:
        sys.argv.append("--dry-run")
    if args.validate_model_load:
        sys.argv.append("--validate-model-load")
    if args.distributed:
        sys.argv.append("--distributed")
        sys.argv.extend(["--num-gpus", str(args.num_gpus)])

    logger.info("🎯 Starting training pipeline...")
    train_main()


if __name__ == "__main__":
    main()
