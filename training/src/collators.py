from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from transformers import DataCollatorForSeq2Seq


class WeightedDataCollator:
    def __init__(self, tokenizer: Any, label_pad_token_id: int = -100, pad_to_multiple_of: int | None = 8):
        self.base = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=pad_to_multiple_of,
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        rows = [deepcopy(feature) for feature in features]
        sample_weights = [float(row.pop("sample_weight", 1.0) or 1.0) for row in rows]
        batch = self.base(rows)
        batch["sample_weight"] = torch.tensor(sample_weights, dtype=torch.float32)
        return batch
