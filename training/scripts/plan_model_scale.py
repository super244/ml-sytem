from __future__ import annotations

import argparse
import json

import yaml

from training.src.scaling import format_parameter_count, resolve_scratch_architecture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan a scratch model architecture from a target parameter count.")
    parser.add_argument("--target-parameters", required=True, help="Examples: 750m, 1.3b, 2b, 3b")
    parser.add_argument("--model-type", default="qwen2")
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--max-position-embeddings", type=int, default=4096)
    parser.add_argument("--rope-theta", type=float, default=1_000_000.0)
    parser.add_argument("--tie-word-embeddings", action="store_true")
    parser.add_argument("--yaml-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    architecture, estimated_parameters = resolve_scratch_architecture(
        model_type=args.model_type,
        target_parameters=args.target_parameters,
        architecture_overrides={
            "vocab_size": args.vocab_size,
            "max_position_embeddings": args.max_position_embeddings,
            "rope_theta": args.rope_theta,
            "tie_word_embeddings": args.tie_word_embeddings,
        },
    )
    if args.yaml_only:
        print(yaml.safe_dump({"target_parameters": args.target_parameters, "architecture": architecture}, sort_keys=False))
        return
    payload = {
        "target_parameters": args.target_parameters,
        "estimated_parameters": estimated_parameters,
        "estimated_parameters_human": format_parameter_count(int(estimated_parameters or 0)),
        "model_type": args.model_type,
        "architecture": architecture,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
