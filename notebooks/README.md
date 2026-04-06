# Notebook Lab

The notebook lab is generated from `build_notebooks.py` so the research workbench stays reproducible and easy to refresh as the system evolves.

## Included Notebooks

- `00_dataset_landscape.ipynb`
- `01_calculus_generator_lab.ipynb`
- `02_transformer_exploration.ipynb`
- `03_base_vs_finetuned_inference.ipynb`
- `04_evaluation_win_case_browser.ipynb`
- `05_lora_experiment_board.ipynb`
- `06_public_dataset_normalization.ipynb`
- `07_dataset_quality_audit.ipynb`
- `08_prompt_optimization_lab.ipynb`
- `09_reranking_self_consistency.ipynb`
- `10_verifier_analysis.ipynb`
- `11_benchmark_slice_analysis.ipynb`
- `12_error_driven_retraining.ipynb`
- `13_run_artifact_explorer.ipynb`
- `14_colab_training_pipeline.ipynb`

## Regeneration

```bash
python3 notebooks/build_notebooks.py
```

The builder writes notebooks with a shared structure so they are readable in version control and straightforward to extend.
