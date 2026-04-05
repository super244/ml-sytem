# 📊 AI-Factory Data Layer

The data layer transforms heterogeneous AI corpora into a unified research-grade schema with reproducible packs. Built around canonical `v2` records, lineage tracking, deduplication, contamination checks, quality scoring, and domain-specific slices.
Processed corpora now ship in two parallel forms: JSONL splits for portability and `corpus.sqlite` for indexed local querying and DB-backed training reads.

## 🏗️ **Architecture Overview**

```
data/
├── adapters/          # Dataset registry & normalization
├── builders/           # Corpus assembly & pack construction
├── configs/           # Processing & generation configurations
├── custom/            # Custom dataset families & manifests
├── processed/         # Normalized datasets & packs
├── quality/           # Quality scoring & analysis
├── reports/           # Dataset cards & analytics
├── synthesis/         # Synthetic data generation
└── tools/            # CLI utilities & validation
```

## 📦 **Main Packages**

### **🔌 Adapters** (`adapters/`)
- **Registry**: Central dataset catalog with metadata
- **Normalization**: Convert heterogeneous formats to v2 schema
- **Loaders**: Support for local, HuggingFace, S3, and web sources

### **🏗️ Builders** (`builders/`)
- **Corpus Builder**: Assemble multi-source datasets
- **Pack Registry**: Create reproducible training/evaluation packs
- **Version Control**: Track dataset lineage and versions

### **⭐ Quality** (`quality/`)
- **Difficulty Estimation**: Automatic complexity scoring
- **Contamination Detection**: Cross-dataset overlap analysis
- **Quality Metrics**: Content quality and coverage assessment
- **Failure Mining**: Extract and analyze error cases

### **🔧 Tools** (`tools/`)
- **Preview**: Quick dataset inspection
- **Validation**: Schema and format verification
- **Export**: Subset extraction and format conversion
- **Audit**: Comprehensive quality reports

## 🎯 **Multi-Domain Support**

### **Mathematics Domain**
- **Calculus**: Derivatives, integrals, limits
- **Algebra**: Linear, abstract, computational algebra
- **Olympiad**: Competition mathematics and proofs
- **Statistics**: Probability, statistical analysis

### **Code Domain** (Extensible)
- **Python**: Algorithm implementation and debugging
- **JavaScript**: Web development and DOM manipulation
- **System Design**: Architecture and optimization

### **Reasoning Domain** (Extensible)
- **Logic**: Formal logic and syllogisms
- **Pattern Recognition**: Sequence and pattern analysis
- **Causal Reasoning**: Cause-effect relationships

## ⚙️ **Configuration System**

### **Processing Config** (`data/configs/processing.yaml`)
```yaml
sources:
  - kind: local
    path: "data/custom/*.jsonl"
    sample_ratio: 1.0
  - kind: huggingface
    dataset: "openai/gsm8k"
    split: "train"
    sample_ratio: 0.5
  - kind: composite
    name: "mathematics_mix"
    sources: ["derivatives", "integrals"]
    ratios: [0.6, 0.4]
emit_sqlite: true
sqlite_output_path: corpus.sqlite
```

### **Generation Config** (`data/configs/generation.yaml`)
```yaml
custom_total_size_gb: 2.0
dataset_specs:
  - id: custom_derivative_mastery
    target_share: 0.1666666667
```

## 📋 **Dataset Families**

### **Mathematics**
- `custom_derivative_mastery` - Calculus derivatives
- `custom_integral_arena` - Integration problems
- `custom_limits_series_lab` - Limits and series
- `custom_olympiad_reasoning_studio` - Competition math

### **Synthetic Generation**
- **Template-based**: Parameterized problem generation
- **Curriculum-based**: Progressive difficulty sequences
- **Domain-specific**: Tailored to mathematical subdomains

## 🚀 **Core Commands**

### **Data Processing**
```bash
# Generate synthetic datasets
python data/generator/generate_calculus_datasets.py --config data/configs/generation.yaml

# Normalize and process datasets
python data/prepare_dataset.py --config data/configs/processing.yaml

# Validate processed datasets
python data/tools/validate_dataset.py --input data/processed/*.jsonl
python data/tools/validate_dataset.py --input data/processed/corpus.sqlite --split train
```

### **Quality Analysis**
```bash
# Preview dataset statistics
python data/tools/preview_dataset.py --input data/processed/train.jsonl
python data/tools/preview_dataset.py --input data/processed/corpus.sqlite --split train

# Audit dataset quality
python data/tools/audit_dataset.py --input data/processed/normalized_all.jsonl

# Mine failure cases from evaluations
python data/mine_failure_cases.py --input evaluation/results/latest/per_example.jsonl
```

### **Export & Utilities**
```bash
# Export subset for testing
python data/tools/export_subset.py --input data/processed/train.jsonl --output test_subset.jsonl --limit 100

# Build benchmark packs
python data/tools/build_benchmark_pack.py --input data/processed/normalized_all.jsonl

# Deduplicate near-duplicates
python data/tools/deduplicate_simhash.py --input data/processed/train.jsonl
```

## 📈 **Quality Metrics**

### **Automated Scoring**
- **Difficulty**: Algorithmic complexity estimation
- **Quality**: Content coherence and completeness
- **Diversity**: Topic and format variety
- **Contamination**: Cross-dataset overlap detection

### **Human Validation**
- **Expert Review**: Domain expert validation
- **Inter-annotator Agreement**: Consistency scoring
- **Error Analysis**: Systematic error identification

## 🔍 **Schema (v2)**

### **Universal Record**
```json
{
  "schema_version": "v2",
  "id": "unique_identifier",
  "domain": "mathematics",
  "subdomain": "calculus",
  "question": "What is the derivative of x²?",
  "solution": "Using the power rule, d/dx(x²) = 2x",
  "final_answer": "2x",
  "difficulty": "medium",
  "metadata": {
    "generator": "synthetic",
    "quality_score": 0.95
  }
}
```

## 🎯 **Best Practices**

### **Data Quality**
1. **Validation**: Always validate after processing
2. **Deduplication**: Remove near-duplicates to prevent overfitting
3. **Balance**: Maintain difficulty and topic distribution
4. **Documentation**: Include clear lineage and metadata

### **Processing Pipeline**
1. **Source Integration**: Combine multiple data sources
2. **Normalization**: Convert to unified v2 schema
3. **Quality Control**: Apply quality filters and scoring
4. **Pack Creation**: Build training/evaluation packs
5. **Validation**: Final verification and testing

## 📚 **Examples & Tutorials**

See the `notebooks/` directory for:
- `00_dataset_landscape.ipynb` - Dataset overview
- `01_calculus_generator_lab.ipynb` - Synthetic generation
- `07_dataset_quality_audit.ipynb` - Quality analysis

## 🔗 **Integration**

The data layer integrates seamlessly with:
- **Training**: Automatic pack loading for training pipelines
- **Evaluation**: Benchmark pack creation for testing
- **Inference**: Real-time data validation and formatting
- **Monitoring**: Data quality metrics and alerts

SQLite note:

- `data/processed/corpus.sqlite` is for corpus rows and dataset inspection
- `artifacts/control_plane/control_plane.db` is a separate SQLite database for orchestration state, runs, leases, and events
