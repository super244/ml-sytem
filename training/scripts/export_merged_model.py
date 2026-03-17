from __future__ import annotations

import argparse
import json
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM

from training.src.config import load_experiment_config
from training.src.modeling import build_quantization_config, load_tokenizer, resolve_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a merged base+adapter model from a training config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    tokenizer = load_tokenizer(config)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.base_model_name,
        trust_remote_code=config.model.trust_remote_code,
        device_map="cpu",
        torch_dtype=resolve_dtype(config.model.bnb_compute_dtype),
        quantization_config=build_quantization_config(config),
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    merged = model.merge_and_unload()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
