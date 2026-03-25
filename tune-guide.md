Below are exact setup steps for a clean machine.

**1. Clone + Python env**
```bash
git clone <your-repo-url> atlas-math-lab
cd atlas-math-lab

python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

**2. Env vars**
```bash
cp .env.example .env
```
Optional: edit `.env` if you want `ARTIFACTS_DIR` somewhere else.

**3. Frontend deps**
```bash
cd frontend
npm install
cd ..
```

**4. Sanity check the workspace**
```bash
python scripts/doctor.py
```

**5. Build data (synthetic + processed corpus)**
```bash
python data/generator/generate_calculus_datasets.py --config data/configs/generation.yaml
python data/prepare_dataset.py --config data/configs/processing.yaml
python data/tools/validate_dataset.py --input data/processed/train.jsonl --manifest data/processed/manifest.json
```
Optional (requires network): normalize public datasets
```bash
python data/public/normalize_public_datasets.py --registry data/public/registry.yaml
```

**6. Dry-run training (validates config + wiring without long runs)**
```bash
python -m training.train --config training/configs/profiles/baseline_qlora.yaml --dry-run
```

**7. Tune the model (LoRA/QLoRA profiles)**
Start with one of these:
```bash
python -m training.train --config training/configs/profiles/baseline_qlora.yaml
# or
python -m training.train --config training/configs/profiles/calculus_specialist.yaml
# or
python -m training.train --config training/configs/profiles/fast_iteration_small_model.yaml
```
Then inspect what you just produced:
```bash
python scripts/latest_run.py
```
Artifacts land in `artifacts/runs/...` and packaged model assets in `artifacts/models/...`.

**8. Start the API**
Terminal A:
```bash
uvicorn inference.app.main:app --host 127.0.0.1 --port 8000 --reload
```

**9. Start the app (UI)**
Terminal B:
```bash
cd frontend
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 npm run dev
```
Open:
- `http://localhost:3000/` (solve)
- `http://localhost:3000/compare`
- `http://localhost:3000/datasets`
- `http://localhost:3000/benchmarks`
- `http://localhost:3000/runs`

**10. Evaluate (optional, but recommended)**
```bash
python -m evaluation.evaluate --config evaluation/configs/base_vs_finetuned.yaml
```

If you tell me what GPU/OS you’re running (Mac CPU-only vs NVIDIA CUDA), I can point you to the best training profile + any environment tweaks to avoid the common `bitsandbytes`/CUDA pitfalls.