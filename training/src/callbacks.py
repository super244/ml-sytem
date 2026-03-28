from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from training.src.tracking import CompositeTracker


class JsonlMetricsCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.output_path = Path(output_dir)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs: dict[str, Any] | None = None, **kwargs):
        if not logs:
            return
        payload = {
            "step": state.global_step,
            "epoch": state.epoch,
            **logs,
        }
        with self.output_path.open("a") as handle:
            handle.write(json.dumps(payload) + "\n")


class TrackerCallback(TrainerCallback):
    def __init__(self, tracker: CompositeTracker):
        self.tracker = tracker

    def on_log(self, args, state, control, logs: dict[str, Any] | None = None, **kwargs):
        if not logs:
            return
        self.tracker.log_metrics(logs, step=state.global_step)
