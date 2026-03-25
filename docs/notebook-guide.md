# Notebook Guide

The notebook lab is generated from `notebooks/build_notebooks.py` so the project always has a reproducible exploration surface.

## Coverage

- dataset landscape and quality audit
- calculus and olympiad generator exploration
- tokenizer and prompt exploration
- base-vs-fine-tuned inference
- evaluation win cases
- LoRA experiment tracking
- public dataset normalization workflow
- prompt optimization
- self-consistency and reranking
- verifier analysis
- benchmark slicing
- error-driven retraining
- run artifact exploration

## Refresh Command

```bash
python3 notebooks/build_notebooks.py
```

Regenerate notebooks after changing builders, schemas, or core workflows so the lab stays aligned with the codebase.
