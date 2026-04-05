from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast

from ai_factory.core.artifacts import write_json
from training.src.config import load_experiment_config
from training.src.data import build_training_text, load_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a local BPE tokenizer from a training profile.")
    parser.add_argument("--config", required=True, help="Training profile used to source text documents.")
    parser.add_argument("--output-dir", required=True, help="Directory where the tokenizer will be saved.")
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


def _iter_texts(config_path: str, max_records: int | None = None) -> tuple[Iterator[str], list[str], int]:
    config = load_experiment_config(config_path)
    data_files = [path for path in [config.data.train_file, config.data.eval_file, config.data.test_file] if path]

    def generator() -> Iterator[str]:
        emitted = 0
        split_map = {"train": config.data.train_file, "eval": config.data.eval_file, "test": config.data.test_file}
        for split_name, file_path in split_map.items():
            if not file_path:
                continue
            for record in load_records(file_path, split=split_name):
                yield build_training_text(record, config.data)
                emitted += 1
                if max_records is not None and emitted >= max_records:
                    return

    return generator(), data_files, len(data_files)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_experiment_config(args.config)
    vocab_size = args.vocab_size or int(config.model.architecture.get("vocab_size", 49152))
    texts, data_files, file_count = _iter_texts(args.config, max_records=args.max_records)

    special_tokens = [
        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
    ]

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=args.min_frequency,
        show_progress=True,
        special_tokens=special_tokens,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> <bos> $B <eos>",
        special_tokens=[("<bos>", tokenizer.token_to_id("<bos>")), ("<eos>", tokenizer.token_to_id("<eos>"))],
    )

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    fast_tokenizer.model_max_length = config.data.max_length
    fast_tokenizer.save_pretrained(str(output_dir))

    summary = {
        "config": str(Path(args.config).resolve()),
        "output_dir": str(output_dir),
        "vocab_size": len(fast_tokenizer),
        "min_frequency": args.min_frequency,
        "source_files": data_files,
        "source_file_count": file_count,
        "max_records": args.max_records,
    }
    write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
