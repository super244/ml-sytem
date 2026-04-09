# Codex Accuracy Final

## Intent
This profile targets the full Codex accuracy push by starting from the already-trained BOSS/Qwen3.5 adapter, consuming the ~2GB `data/processed/normalized_all.jsonl` corpus, and running a 5-epoch LoRA-based fine-tune with the evaluation-informed curriculum described in the codex prompt.

## Launch
Use the existing training entry point with the following config:

```bash
python -m training.train --config training/configs/profiles/codex_accuracy_final.yaml
```

### Dataset assumptions
- Training data: `data/processed/normalized_all.jsonl`
- Evaluation data: `data/processed/eval.jsonl`
- Test data: `data/processed/test.jsonl`

## Notes
- Teacher model: `Qwen/Qwen3.5-32B-Instruct-AWQ` (cached under `cache/teacher_model/codex_accuracy_final`)
- The profile preserves the LoRA math adapter while pushing gradient accumulation and curriculum weighting for lengthy runs.
