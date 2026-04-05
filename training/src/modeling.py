from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available

from training.src.config import ExperimentConfig
from training.src.scaling import resolve_scratch_architecture

logger = logging.getLogger(__name__)


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


def resolve_tokenizer_reference(config: ExperimentConfig) -> str:
    if config.model.tokenizer_path:
        tokenizer_path = Path(config.model.tokenizer_path).expanduser()
        if tokenizer_path.exists():
            return str(tokenizer_path)
    if config.model.tokenizer_name:
        return config.model.tokenizer_name
    return config.model.base_model_name


def load_tokenizer(config: ExperimentConfig) -> Any:
    tokenizer_name = resolve_tokenizer_reference(config)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=config.model.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"
    return tokenizer


def resolve_attention_implementation(config: ExperimentConfig) -> str | None:
    if not config.model.use_flash_attention:
        return None
    if is_flash_attn_2_available():
        return "flash_attention_2"
    logger.warning(
        "FlashAttention2 was requested for %s, but it is unavailable in the current environment. "
        "Falling back to the default attention implementation.",
        config.model.base_model_name,
    )
    return None


def build_model_from_scratch(config: ExperimentConfig, tokenizer: Any | None = None) -> Any:
    architecture, estimated_parameters = resolve_scratch_architecture(
        model_type=config.model.model_type or "",
        architecture_overrides=config.model.architecture,
        target_parameters=config.model.target_parameters,
    )
    if tokenizer is not None:
        architecture.setdefault("vocab_size", len(tokenizer))
        if tokenizer.pad_token_id is not None:
            architecture.setdefault("pad_token_id", tokenizer.pad_token_id)
        if tokenizer.bos_token_id is not None:
            architecture.setdefault("bos_token_id", tokenizer.bos_token_id)
        if tokenizer.eos_token_id is not None:
            architecture.setdefault("eos_token_id", tokenizer.eos_token_id)

    model_type = config.model.model_type or architecture.get("model_type")
    if not model_type:
        raise ValueError("Scratch model initialization requires model.model_type.")
    if estimated_parameters is not None:
        logger.info(
            "Resolved scratch architecture for %s target %s -> hidden=%s layers=%s estimated_parameters=%s",
            model_type,
            config.model.target_parameters,
            architecture.get("hidden_size"),
            architecture.get("num_hidden_layers"),
            estimated_parameters,
        )

    hf_config = AutoConfig.for_model(model_type, **architecture)
    attention_implementation = resolve_attention_implementation(config)
    if attention_implementation:
        hf_config._attn_implementation = attention_implementation
        hf_config.attn_implementation = attention_implementation

    model = AutoModelForCausalLM.from_config(
        hf_config,
        trust_remote_code=config.model.trust_remote_code,
    )
    model.config.use_cache = False
    if config.model.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def load_model_for_training(config: ExperimentConfig, tokenizer: Any | None = None) -> Any:
    if config.model.initialization.lower() == "scratch":
        if build_quantization_config(config) is not None:
            raise ValueError("Scratch training requires use_4bit=false and use_8bit=false.")
        method = (config.lora.method or "full").lower()
        if method not in {"full", "sft"}:
            raise ValueError("Scratch initialization currently supports method=full or method=sft.")
        return build_model_from_scratch(config, tokenizer=tokenizer)

    quantization_config = build_quantization_config(config)
    dtype = resolve_dtype(config.model.bnb_compute_dtype)
    method = (config.lora.method or "qlora").lower()
    if method in {"full", "sft"} and quantization_config is not None:
        raise ValueError("Full/SFT training requires use_4bit=false and use_8bit=false.")

    model = AutoModelForCausalLM.from_pretrained(
        config.model.base_model_name,
        trust_remote_code=config.model.trust_remote_code,
        quantization_config=quantization_config,
        device_map=config.model.device_map,
        torch_dtype=dtype,
        low_cpu_mem_usage=config.runtime.low_cpu_mem_usage,
        attn_implementation=resolve_attention_implementation(config),
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
