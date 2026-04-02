from __future__ import annotations

from typing import Any

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SequentialSampler
from transformers import Trainer
from trl import DPOConfig, DPOTrainer, ORPOConfig, ORPOTrainer


class MathTrainer(Trainer):
    """Custom trainer for math tasks with weighted loss for curriculum learning."""

    def __init__(self, *args, sequential_training: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequential_training = sequential_training

    def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **kwargs: Any) -> Any:
        sample_weight = inputs.pop("sample_weight", None)
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if labels is None or logits is None:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            return (loss, outputs) if return_outputs else loss

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.shape)
        token_mask = shift_labels.ne(-100)
        per_example_loss = (token_loss * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1)
        if sample_weight is not None:
            sample_weight = sample_weight.to(per_example_loss.device)
            loss = (per_example_loss * sample_weight).mean()
        else:
            loss = per_example_loss.mean()
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self) -> DataLoader:
        if not self.sequential_training:
            return super().get_train_dataloader()
        if self.train_dataset is None:
            raise ValueError("Trainer requires a train_dataset.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=SequentialSampler(self.train_dataset),
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def build_ultimate_trainer(
    config: Any,
    model: Any,
    args: Any,
    train_dataset: Any,
    eval_dataset: Any,
    data_collator: Any,
    tokenizer: Any,
    callbacks: list[Any],
) -> Any:
    """Builds the appropriate trainer (SFT, DPO, ORPO) based on experiment config."""
    if not config.preference.enabled:
        return MathTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            sequential_training=(config.data.curriculum_learning and config.data.sequential_curriculum),
            callbacks=callbacks,
        )

    # Preference Optimization (DPO/ORPO)
    if config.preference.loss_type == "orpo":
        orpo_args = ORPOConfig(
            **args.to_dict(),
            beta=config.preference.beta,
            max_prompt_length=config.preference.max_prompt_length,
            max_length=config.preference.max_prompt_length + config.preference.max_target_length,
        )
        return ORPOTrainer(
            model=model,
            args=orpo_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks,
        )

    # Default to DPO
    dpo_args = DPOConfig(
        **args.to_dict(),
        beta=config.preference.beta,
        loss_type=config.preference.loss_type,
        max_prompt_length=config.preference.max_prompt_length,
        max_length=config.preference.max_prompt_length + config.preference.max_target_length,
        label_smoothing=config.preference.label_smoothing,
    )
    return DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )
