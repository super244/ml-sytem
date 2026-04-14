"""
Teacher-Student Training Implementation for Ultimate Upgrade

This module implements knowledge distillation where a teacher model (Qwen3-4B)
trains a student model using both hard labels (ground truth) and soft labels
(teacher predictions) with temperature scaling.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer import Trainer

logger = logging.getLogger(__name__)


class TeacherStudentConfig:
    """Configuration for teacher-student training."""

    def __init__(
        self,
        teacher_model_name: str = "Qwen/Qwen2.5-7B-Instruct",  # Using Qwen2.5-7B as teacher
        temperature: float = 2.0,
        alpha: float = 0.7,  # Weight for teacher loss vs student loss
        use_kl_divergence: bool = True,
        teacher_cache_dir: str | None = None,
        max_teacher_sequence_length: int = 2048,
        teacher_batch_size: int = 4,
    ):
        self.teacher_model_name = teacher_model_name
        self.temperature = temperature
        self.alpha = alpha
        self.use_kl_divergence = use_kl_divergence
        self.teacher_cache_dir = teacher_cache_dir
        self.max_teacher_sequence_length = max_teacher_sequence_length
        self.teacher_batch_size = teacher_batch_size


class TeacherModel:
    """Wrapper for teacher model with efficient inference."""

    def __init__(self, config: TeacherStudentConfig, device: str = "auto"):
        self.config = config
        self.device = self._resolve_device(device)
        self.model = None
        self.tokenizer = None
        self._load_teacher()

    def _resolve_device(self, device: str) -> str:
        """Resolve the best device for teacher model."""
        if device == "auto":
            # Force CPU for stability
            return "cpu"
        return device

    def _load_teacher(self):
        """Load the teacher model and tokenizer."""
        logger.info(f"Loading teacher model: {self.config.teacher_model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.teacher_model_name,
            trust_remote_code=True,
            cache_dir=self.config.teacher_cache_dir,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with optimizations for inference
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.teacher_model_name,
            trust_remote_code=True,
            cache_dir=self.config.teacher_cache_dir,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map=self.device,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        # Enable optimizations
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

        logger.info(f"Teacher model loaded on {self.device}")

    def get_teacher_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Get teacher model logits for given inputs."""
        with torch.no_grad():
            # Truncate if necessary
            max_len = min(input_ids.shape[1], self.config.max_teacher_sequence_length)
            input_ids = input_ids[:, :max_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :max_len]

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            return outputs.logits

    def generate_teacher_labels(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 512,
    ) -> torch.Tensor:
        """Generate teacher predictions for training."""
        with torch.no_grad():
            # Truncate if necessary
            max_len = min(input_ids.shape[1], self.config.max_teacher_sequence_length - max_new_tokens)
            input_ids = input_ids[:, :max_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :max_len]

            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
            )

            # Get logits for generated tokens
            generated_ids = outputs.sequences
            teacher_logits = []

            for i, sequence in enumerate(generated_ids):
                # Get logits for each generated token
                for step in range(len(input_ids[i]), len(sequence)):
                    step_outputs = self.model(
                        input_ids=sequence[:step].unsqueeze(0),
                        return_dict=True,
                    )
                    teacher_logits.append(step_outputs.logits[0, -1, :])

            if teacher_logits:
                return torch.stack(teacher_logits)
            else:
                # Fallback to regular forward pass
                return self.get_teacher_logits(input_ids, attention_mask)


class TeacherStudentTrainer(Trainer):
    """Custom trainer implementing teacher-student knowledge distillation."""

    def __init__(self, teacher_config: TeacherStudentConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_config = teacher_config
        self.teacher = None
        self._initialize_teacher()

    def _initialize_teacher(self):
        """Initialize the teacher model."""
        self.teacher = TeacherModel(
            self.teacher_config, device=self.args.device if hasattr(self.args, "device") else "auto"
        )
        logger.info("Teacher model initialized for knowledge distillation")

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """
        Compute the combined loss: alpha * teacher_loss + (1-alpha) * student_loss
        """
        # Get labels
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("Labels must be provided for teacher-student training")

        # Student forward pass
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        # Standard student loss (cross-entropy with ground truth)
        student_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        student_loss = student_loss_fct(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))

        try:
            # Get teacher logits for distillation
            teacher_logits = self.teacher.get_teacher_logits(inputs["input_ids"], inputs.get("attention_mask"))

            # Ensure teacher logits have same shape as student logits
            if teacher_logits.shape != student_logits.shape:
                # Truncate or pad to match student logits shape
                min_seq_len = min(teacher_logits.shape[1], student_logits.shape[1])
                teacher_logits = teacher_logits[:, :min_seq_len, : student_logits.shape[2]]
                student_logits_truncated = student_logits[:, :min_seq_len, :]
            else:
                student_logits_truncated = student_logits

            # Student logits with temperature (teacher used directly in MSE below)
            student_logits_temp = student_logits_truncated / self.teacher_config.temperature

            # Knowledge distillation loss using MSE for stability
            distill_loss = F.mse_loss(student_logits_temp, teacher_logits)

            # Combine losses
            total_loss = self.teacher_config.alpha * distill_loss + (1 - self.teacher_config.alpha) * student_loss

        except Exception as e:
            logger.warning(f"Teacher distillation failed, using only student loss: {e}")
            total_loss = student_loss
            distill_loss = torch.tensor(0.0, device=student_loss.device)

        # Log losses for monitoring
        if self.state.global_step % 100 == 0:
            logger.info(
                f"Step {self.state.global_step}: "
                f"Student Loss: {student_loss.item():.4f}, "
                f"Distill Loss: {distill_loss.item():.4f}, "
                f"Total Loss: {total_loss.item():.4f}, "
                f"Alpha: {self.teacher_config.alpha}"
            )

        return (
            (total_loss, {"student_loss": student_loss, "distill_loss": distill_loss}) if return_outputs else total_loss
        )

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to include teacher-student metrics."""
        result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # Add teacher-student specific metrics
        if hasattr(self, "teacher"):
            result[f"{metric_key_prefix}_teacher_model"] = self.teacher.config.teacher_model_name
            result[f"{metric_key_prefix}_temperature"] = self.teacher.config.temperature
            result[f"{metric_key_prefix}_alpha"] = self.teacher.config.alpha

        return result


def create_teacher_student_trainer(
    model: nn.Module,
    args,
    train_dataset,
    eval_dataset=None,
    tokenizer=None,
    data_collator=None,
    teacher_config: TeacherStudentConfig | None = None,
    **kwargs,
) -> TeacherStudentTrainer:
    """Factory function to create a teacher-student trainer."""
    if teacher_config is None:
        teacher_config = TeacherStudentConfig()

    # Remove conflicting arguments from kwargs
    kwargs.pop("tokenizer", None)
    kwargs.pop("data_collator", None)

    return TeacherStudentTrainer(
        teacher_config=teacher_config,
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer
        data_collator=data_collator,
        **kwargs,
    )
