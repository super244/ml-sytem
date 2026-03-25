from __future__ import annotations

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SequentialSampler
from transformers import Trainer


class MathTrainer(Trainer):
    def __init__(self, *args, sequential_training: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequential_training = sequential_training

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
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
