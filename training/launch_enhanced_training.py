#!/usr/bin/env python3
"""
Simple Training Script for Enhanced Model

This script trains your existing fine-tuned model with full parameter training
to improve accuracy beyond the base model.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add training src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from training.train import main as train_main

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train enhanced model with full parameter training"
    )
    parser.add_argument(
        "--config",
        default="training/configs/enhance_existing_model_simple.yaml",
        help="Path to enhanced training config file"
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Resume from specific checkpoint"
    )
    parser.add_argument(
        "--resume-from-latest-checkpoint",
        action="store_true",
        help="Resume from latest checkpoint"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configs and setup without training"
    )
    parser.add_argument(
        "--validate-model-load",
        action="store_true",
        help="Load model after validation and exit"
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Launch in distributed mode"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for distributed training"
    )
    
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
    
    logger.info("🚀 Starting Enhanced Model Training!")
    logger.info(f"📚 Config: {args.config}")
    logger.info("🎯 Goal: Improve accuracy beyond base model")
    
    # Override sys.argv for train_main
    sys.argv = [
        "train.py",
        "--config", args.config,
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
