# Public Dataset Adapters

The repository treats public datasets as first-class inputs, but it does **not** vendor large public corpora directly into git. Instead, it includes:

- a registry of public math datasets that are useful for calculus and advanced reasoning
- downloader and normalization scripts
- support for filtering those sources into the same canonical schema used by the custom corpora
- usage, weighting, and benchmark metadata for each registry entry

This keeps the repo practical while still making public data part of the workflow.

Core files:

- `registry.yaml`
- `download_public_datasets.py`
- `normalize_public_datasets.py`

Registry entries can now carry:

- dataset family
- expected topic
- reasoning style
- usage intent
- default source weight
- benchmark tags
- split strategy
- loader/filter metadata

If you have network access and the necessary licenses, you can download and normalize the public datasets, then point `data/prepare_dataset.py` at `data/public/normalized/*.jsonl`.
