from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class ModelSpec:
    name: str
    base_model: str
    adapter_path: str | None = None
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    label: str | None = None
    description: str | None = None
    tags: list[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        if self.adapter_path is None:
            return True
        return Path(self.adapter_path).exists()


def load_registry_from_yaml(path: str | Path) -> dict[str, ModelSpec]:
    payload = yaml.safe_load(Path(path).read_text()) or {}
    return {
        item["name"]: ModelSpec(**item)
        for item in payload.get("models", [])
    }


def resolve_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def build_quant_config(spec: ModelSpec) -> BitsAndBytesConfig | None:
    if spec.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=resolve_dtype(spec.dtype),
        )
    if spec.load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any


class MathModelRuntime:
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self._loaded: LoadedModel | None = None
        self._lock = Lock()

    def is_available(self) -> bool:
        return self.spec.available

    def load(self) -> LoadedModel:
        if self._loaded is not None:
            return self._loaded
        with self._lock:
            if self._loaded is not None:
                return self._loaded
            if not self.is_available():
                raise FileNotFoundError(
                    f"Adapter path for model '{self.spec.name}' was not found: {self.spec.adapter_path}"
                )
            tokenizer = AutoTokenizer.from_pretrained(
                self.spec.base_model,
                trust_remote_code=self.spec.trust_remote_code,
                use_fast=False,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            model = AutoModelForCausalLM.from_pretrained(
                self.spec.base_model,
                trust_remote_code=self.spec.trust_remote_code,
                device_map="auto",
                torch_dtype=resolve_dtype(self.spec.dtype),
                quantization_config=build_quant_config(self.spec),
            )
            if self.spec.adapter_path:
                model = PeftModel.from_pretrained(model, self.spec.adapter_path)
            model.eval()
            self._loaded = LoadedModel(model=model, tokenizer=tokenizer)
            return self._loaded


class MathModelRegistry:
    def __init__(self, specs: dict[str, ModelSpec]):
        self._runtimes = {name: MathModelRuntime(spec) for name, spec in specs.items()}

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "label": runtime.spec.label or name,
                "description": runtime.spec.description,
                "base_model": runtime.spec.base_model,
                "adapter_path": runtime.spec.adapter_path,
                "available": runtime.is_available(),
                "tags": runtime.spec.tags,
            }
            for name, runtime in self._runtimes.items()
        ]

    def get_runtime(self, name: str) -> MathModelRuntime:
        if name not in self._runtimes:
            raise KeyError(f"Unknown model variant: {name}")
        return self._runtimes[name]
