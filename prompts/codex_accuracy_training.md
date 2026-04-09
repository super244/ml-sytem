# Codex Prompt: Ultimate Accuracy-Boosting Training Suite

## Objective
Create a final training implementation that maximizes evaluation accuracy on mathematical reasoning tasks. This should be the "production-grade" training suite that directly targets the gap between low training loss and actual evaluation performance.

## Context

### Current State
- **Base Model**: `Qwen/Qwen2.5-Math-1.5B-Instruct`
- **Current Best Trained Model**: `atlas_boss_teacher_student_final-20260407-042110` (BOSS Qwen3.5)
- **Current Training Loss**: 15.53 (excellent)
- **Current Evaluation Accuracy**: ~20% (needs improvement)
- **Gap**: Model overfits to training format, doesn't generalize to evaluation

### Previous Training Approaches Tried
1. Standard LoRA fine-tuning (loss: 1750, accuracy: 0%)
2. Optimized LoRA with teacher guidance (loss: 300, accuracy: 40%)
3. BOSS conversational with Qwen3.5-32B (loss: 15.5, accuracy: 20%)

## Requirements

### 1. Training Architecture
Create a new trainer class `AccuracyBoostTrainer` that extends `transformers.Trainer` with:

```python
class AccuracyBoostTrainer(Trainer):
    """
    Production-grade trainer targeting evaluation accuracy through:
    - Answer correctness verification during training
    - Format-aware loss weighting
    - Hard negative mining from evaluation failures
    - Curriculum learning by difficulty
    """
```

### 2. Key Features to Implement

**A. Answer Verification During Training**
- Extract predicted answers from model outputs during forward pass
- Compare against ground truth using symbolic verification (sympy for math)
- Apply higher loss weight to examples where answer is wrong
- Use `ai_factory.core.answers.verify_prediction()` for verification

**B. Format-Regularization Loss**
- Add auxiliary loss term that penalizes format deviations
- Reward `\boxed{answer}` format
- Penalize missing final answers or incorrect formatting
- Weight: 0.1 * format_loss + 0.9 * token_loss

**C. Hard Negative Mining**
- After each epoch, run model on evaluation set
- Identify examples where model fails
- Add these to training batch with higher weight (2x)
- Keep a "failure bank" of top 100 hardest examples

**D. Curriculum Learning**
- Start with easy examples (single-step math)
- Progress to medium (multi-step)
- End with hard (complex reasoning)
- Use difficulty scoring from dataset metadata

**E. Answer-Aware Token Weighting**
- Apply 5x higher weight to tokens within `\boxed{}`
- Apply 2x weight to "final answer" keywords
- Standard weight to reasoning steps

### 3. Training Configuration

Create config file `training/configs/profiles/codex_accuracy_final.yaml`:

```yaml
run_name: codex_accuracy_final
base_model: artifacts/runs/atlas_boss_teacher_student_final-20260407-042110/published/final_adapter  # Start from BOSS model

training:
  num_train_epochs: 5
  learning_rate: 0.00005  # Lower LR for fine-tuning on top of BOSS
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 64
  warmup_ratio: 0.15
  weight_decay: 0.01
  
  # Accuracy-specific settings
  eval_steps: 50
  save_steps: 50
  logging_steps: 5
  
  # Loss weighting
  answer_verification_weight: 2.0
  format_regularization_weight: 0.1
  hard_negative_weight: 3.0
  answer_token_weight: 5.0
  
lora:
  r: 32  # Higher rank for final accuracy push
  alpha: 64
  dropout: 0.05
  target_modules: [q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj]
  use_rslora: true
  
data:
  train_file: data/processed/train.jsonl
  eval_file: data/processed/eval.jsonl
  max_length: 4096
  pack: true
  curriculum: true
  difficulty_field: difficulty_score
```

### 4. Implementation Files to Create

**File 1: `training/src/accuracy_boost_trainer.py`**
- Main trainer implementation
- Override `compute_loss()` to add accuracy-aware loss components
- Override `evaluation_loop()` to collect hard negatives
- Implement `verify_answers()` method

**File 2: `training/src/answer_verification.py`**
- Utility for verifying mathematical answers
- Symbolic comparison using sympy
- Numeric tolerance handling
- LaTeX parsing for boxed answers

**File 3: `training/src/hard_negative_bank.py`**
- Class to manage hardest examples
- Priority queue based on loss magnitude
- Persistence across epochs
- Sampling strategy for training mix

### 5. Training Data Processing

Create data collator that:
- Identifies answer spans in target sequences
- Creates token-level weight masks (5x for answers, 1x for rest)
- Adds difficulty metadata for curriculum
- Packs examples efficiently

### 6. Expected Improvements

Based on the architecture:
- **Target Loss**: < 10 (improve from 15.5)
- **Target Accuracy**: > 60% (improve from 20%)
- **Mechanism**: Direct accuracy feedback loop during training

## Code Structure Template

```python
# training/src/accuracy_boost_trainer.py

import torch
import torch.nn as nn
from transformers import Trainer
from typing import Dict, List, Optional, Tuple
import numpy as np

class AccuracyBoostTrainer(Trainer):
    def __init__(
        self,
        answer_verification_weight: float = 2.0,
        format_regularization_weight: float = 0.1,
        hard_negative_weight: float = 3.0,
        answer_token_weight: float = 5.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.answer_weight = answer_verification_weight
        self.format_weight = format_regularization_weight
        self.hard_neg_weight = hard_negative_weight
        self.answer_token_weight = answer_token_weight
        self.hard_negative_bank = HardNegativeBank(max_size=100)
        
    def compute_loss(self, model, inputs, return_outputs=False):
        # Standard forward pass
        outputs = model(**inputs)
        token_loss = outputs.loss
        
        # Extract predictions and verify answers
        predictions = self.extract_predictions(outputs.logits)
        verified_weights = self.verify_batch_answers(predictions, inputs["labels"])
        
        # Apply answer verification weighting
        weighted_loss = token_loss * verified_weights.mean()
        
        # Add format regularization
        format_loss = self.compute_format_loss(outputs.logits, inputs["labels"])
        
        # Add hard negative weighting
        hard_neg_mask = self.get_hard_negative_mask(inputs)
        
        # Token-level weighting for answer spans
        token_weights = self.compute_token_weights(inputs["labels"])
        
        # Combined loss
        total_loss = (
            self.answer_weight * weighted_loss + 
            self.format_weight * format_loss +
            (token_loss * token_weights).mean()
        )
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def extract_predictions(self, logits: torch.Tensor) -> List[str]:
        """Decode logits to answer strings"""
        pass
    
    def verify_batch_answers(
        self, 
        predictions: List[str], 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Verify answers and return per-example weights"""
        # Use ai_factory.core.answers.verify_prediction()
        # Return 2.0 for wrong answers, 1.0 for correct
        pass
    
    def compute_format_loss(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Loss component encouraging \boxed{} format"""
        pass
    
    def compute_token_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """Create weight mask for answer tokens"""
        # Identify \boxed{...} spans
        # Return 5.0 for answer tokens, 1.0 for others
        pass
    
    def evaluation_loop(self, *args, **kwargs):
        """Override to collect hard negatives"""
        metrics = super().evaluation_loop(*args, **kwargs)
        self.update_hard_negative_bank()
        return metrics
```

## Evaluation Criteria

The implementation should:
1. Train for 5 epochs on the BOSS model checkpoint
2. Achieve training loss < 10
3. When evaluated, show improved answer correctness
4. Maintain fast training speed (< 20 min total)
5. Be fully compatible with existing evaluation pipeline

## Output

Generate complete, runnable Python code for:
1. `training/src/accuracy_boost_trainer.py`
2. `training/src/answer_verification.py`
3. `training/src/hard_negative_bank.py`
4. `training/configs/profiles/codex_accuracy_final.yaml`
5. Update `training/train.py` to use the new trainer when config specifies `accuracy_boost: true`

Include imports, type hints, docstrings, and comments explaining the accuracy-boosting strategy.
