from __future__ import annotations

import torch
import yaml
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference.app.model_catalog import normalize_model_record

if TYPE_CHECKING:
    from transformers import BitsAndBytesConfig


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
    parameter_size_b: float | None = None
    parameter_size_label: str | None = None
    quantization: str | None = None
    tier: str | None = None
    scale_tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        if not self.adapter_path:
            return True
        return Path(self.adapter_path).exists()


def _build_model_spec(item: dict[str, Any]) -> ModelSpec:
    model_fields = {field.name for field in fields(ModelSpec)}
    spec_payload: dict[str, Any] = {key: value for key, value in item.items() if key in model_fields}
    extra_payload = {key: value for key, value in item.items() if key not in model_fields}
    metadata = dict(spec_payload.get("metadata") or {})
    if extra_payload:
        metadata.update(extra_payload)
    if metadata:
        spec_payload["metadata"] = metadata
    return ModelSpec(**spec_payload)


def load_registry_from_yaml(path: str | Path) -> dict[str, ModelSpec]:
    payload = yaml.safe_load(Path(path).read_text()) or {}
    return {item["name"]: _build_model_spec(item) for item in payload.get("models", [])}


def resolve_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def build_quant_config(spec: ModelSpec) -> BitsAndBytesConfig | None:
    from transformers import BitsAndBytesConfig

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
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self.spec.base_model,
                trust_remote_code=self.spec.trust_remote_code,
                use_fast=False,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            # Metal-specific optimizations
            device_map = "auto"
            if torch.backends.mps.is_available():
                device_map = "mps"  # Force Metal device
                torch_dtype = torch.float32  # Full precision for Metal
            else:
                torch_dtype = resolve_dtype(self.spec.dtype)

            model = AutoModelForCausalLM.from_pretrained(
                self.spec.base_model,
                trust_remote_code=self.spec.trust_remote_code,
                device_map=device_map,
                torch_dtype=torch_dtype,
                quantization_config=build_quant_config(self.spec),
                low_cpu_mem_usage=True,  # Memory optimization
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
            normalize_model_record(
                {
                    **asdict(runtime.spec),
                    "name": name,
                    "base_model": runtime.spec.base_model,
                },
                source="runtime",
                available=runtime.is_available(),
            )
            for name, runtime in self._runtimes.items()
        ]

    def get_runtime(self, name: str) -> MathModelRuntime:
        if name not in self._runtimes:
            raise KeyError(f"Unknown model variant: {name}")
        return self._runtimes[name]
