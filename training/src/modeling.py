from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from training.src.config import ExperimentConfig


def resolve_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def build_quantization_config(config: ExperimentConfig) -> BitsAndBytesConfig | None:
    model_config = config.model
    if model_config.use_full_precision:
        return None
    if model_config.use_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_config.double_quant,
            bnb_4bit_compute_dtype=resolve_dtype(model_config.bnb_compute_dtype),
        )
    if model_config.use_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_tokenizer(config: ExperimentConfig) -> Any:
    tokenizer_name = config.model.tokenizer_name or config.model.base_model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=config.model.trust_remote_code,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model_for_training(config: ExperimentConfig) -> Any:
    quantization_config = build_quantization_config(config)
    dtype = resolve_dtype(config.model.bnb_compute_dtype)
    method = (config.lora.method or "qlora").lower()
    if method in {"full", "sft"} and quantization_config is not None:
        raise ValueError("Full/SFT training requires use_4bit=false and use_8bit=false.")

    model = AutoModelForCausalLM.from_pretrained(
        config.model.base_model_name,
        trust_remote_code=config.model.trust_remote_code,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=dtype,
        low_cpu_mem_usage=config.runtime.low_cpu_mem_usage,
    )
    model.config.use_cache = False
    if config.model.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.model.gradient_checkpointing,
        )

    if method in {"full", "sft"}:
        return model
    if method not in {"lora", "qlora"}:
        raise ValueError(f"Unsupported adapter method: {config.lora.method}")

    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias=config.lora.bias,
        task_type=config.lora.task_type,
        target_modules=config.lora.target_modules,
    )
    peft_model = get_peft_model(model, lora_config)
    return peft_model


def export_merged_model(model: Any, output_dir: str) -> str | None:
    if not hasattr(model, "merge_and_unload"):
        return None
    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir)
    return output_dir


def trainable_parameter_report(model: Any) -> dict[str, float]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_ratio": trainable / total if total else 0.0,
    }
