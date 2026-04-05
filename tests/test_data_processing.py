import json
from pathlib import Path

from ai_factory.core.io import write_jsonl
from ai_factory.core.schemas import DatasetBuildInfo
from data.builders import corpus_builder
from data.builders.corpus_builder import ProcessingConfig, coerce_source_specs, load_source_records, normalize_record
from data.builders.pack_registry import build_derived_packs
from data.tools.preview_dataset import preview_rows


def test_normalize_record_accepts_legacy_step_strings() -> None:
    normalized = normalize_record(
        {
            "question": "Evaluate int_0^1 x dx.",
            "solution": "The antiderivative is x^2/2. Final Answer: 1/2",
            "step_checks": "x^2/2||1/2",
            "difficulty": "hard",
        },
        default_source="unit_test",
    )
    assert normalized is not None
    assert normalized["quality_score"] > 0
    assert len(normalized["step_checks"]) == 2


def test_build_derived_packs(tmp_path: Path) -> None:
    rows = [
        {
            "id": "a",
            "question": "Q",
            "solution": "S",
            "difficulty": "hard",
            "topic": "calculus",
            "source": "custom_derivative_mastery",
            "pack_id": "core_train_mix",
            "reasoning_style": "chain_of_thought",
            "step_checks": [{"kind": "substring", "value": "x"}],
            "dataset_split": "test",
        }
    ]
    summaries = build_derived_packs(rows, tmp_path, build=DatasetBuildInfo(build_id="unit-test"))
    ids = {item["id"] for item in summaries}
    assert "calculus_hard_pack" in ids
    assert "benchmark_holdout_pack" in ids
    assert all(item["manifest_path"].endswith("manifest.json") for item in summaries)
    assert all(item["card_path"].endswith("card.md") for item in summaries)
    manifest = json.loads((tmp_path / "calculus_hard_pack" / "manifest.json").read_text())
    assert manifest["build"]["build_id"] == "unit-test"
    assert manifest["metadata"]["card_path"].endswith("card.md")


def test_coerce_source_specs_flattens_composite_ratios() -> None:
    specs = coerce_source_specs(
        [
            {"kind": "local", "path": "data/examples/math_reasoning_examples.jsonl", "sample_ratio": 0.5},
            {
                "kind": "composite",
                "sample_ratio": 0.5,
                "version": "bundle-v1",
                "sources": [
                    {"kind": "local", "path": "data/custom/custom_derivative_mastery.jsonl", "sample_ratio": 0.5},
                    {"kind": "local", "path": "data/custom/custom_integral_arena.jsonl"},
                ],
            },
        ]
    )

    assert [spec.kind for spec in specs] == ["local", "local", "local"]
    assert [spec.sample_ratio for spec in specs] == [0.5, 0.25, 0.5]
    assert specs[1].version == "bundle-v1"


def test_build_corpus_tracks_source_versions_and_ratios(tmp_path: Path, monkeypatch) -> None:
    source_a = tmp_path / "source_a.jsonl"
    source_b = tmp_path / "source_b.jsonl"
    write_jsonl(
        source_a,
        [
            {"id": f"a{i}", "question": f"Q{i}", "solution": f"S{i}", "difficulty": "hard", "topic": "calculus"}
            for i in range(4)
        ],
    )
    write_jsonl(
        source_b,
        [
            {"id": f"b{i}", "question": f"QB{i}", "solution": f"SB{i}", "difficulty": "hard", "topic": "calculus"}
            for i in range(4)
        ],
    )

    config = ProcessingConfig(
        seed=7,
        eval_ratio=0.0,
        test_ratio=0.0,
        min_difficulty="easy",
        sources=[
            {"kind": "local", "path": str(source_a), "sample_ratio": 0.5, "version": "2026.03"},
            {
                "kind": "composite",
                "id": "bundle",
                "sample_ratio": 0.5,
                "version": "bundle-v1",
                "sources": [{"kind": "local", "path": str(source_b)}],
            },
        ],
        output_dir=str(tmp_path / "processed"),
        derived_packs=[],
    )
    config_path = tmp_path / "processing.yaml"
    config_path.write_text("seed: 7\n")
    monkeypatch.setattr(corpus_builder, "build_derived_packs", lambda *args, **kwargs: [])

    result = corpus_builder.build_corpus(config, config_path)
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    lineage_summary = json.loads(Path(result["lineage_summary_path"]).read_text())

    assert result["stats"]["all"]["num_records"] == 4
    assert len(result["source_summaries"]) == 2
    assert result["source_summaries"][0]["rows_selected"] == 2
    assert result["source_summaries"][1]["rows_selected"] == 2
    assert manifest["metadata"]["dataset_version"] == "v2"
    assert manifest["metadata"]["source_summaries"][0]["version"] == "2026.03"
    assert manifest["metadata"]["source_summaries"][1]["sample_ratio"] == 0.5
    assert manifest["metadata"]["lineage_summary_path"].endswith("lineage_summary.json")
    assert "processing_version=v1" in manifest["build"]["notes"]
    assert lineage_summary["total_records"] == 4
    assert lineage_summary["contamination"]["contaminated_records"] == 0
    assert lineage_summary["groups"][0]["record_count"] == 2


def test_web_source_loader_reads_jsonl_payload(monkeypatch) -> None:
    payload = '{"id":"web-1","question":"Q","solution":"S","difficulty":"hard","topic":"calculus"}\n'

    class FakeHeaders:
        def __init__(self, content_type: str) -> None:
            self._content_type = content_type

        def get_content_charset(self) -> str:
            return "utf-8"

        def get(self, key: str, default=None):
            if key.lower() == "content-type":
                return self._content_type
            return default

    class FakeResponse:
        def __init__(self, text: str) -> None:
            self._text = text
            self.headers = FakeHeaders("application/jsonl")

        def read(self) -> bytes:
            return self._text.encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(corpus_builder, "urlopen", lambda request: FakeResponse(payload))
    loaded, summaries, warnings = load_source_records(
        [corpus_builder.SourceSpec(id="web", kind="web", path="https://example.test/dataset.jsonl")],
        seed=1,
    )

    assert warnings == []
    assert summaries[0]["rows_selected"] == 1
    assert loaded[0][1]["id"] == "web-1"


def test_optional_source_load_failure_is_skipped(monkeypatch) -> None:
    def raise_runtime(spec):
        raise RuntimeError("missing dependency")

    monkeypatch.setattr(corpus_builder, "load_source_rows", raise_runtime)
    loaded, summaries, warnings = load_source_records(
        [corpus_builder.SourceSpec(id="s3-snapshot", kind="s3", path="s3://bucket/dataset.jsonl", optional=True)],
        seed=1,
    )

    assert loaded == []
    assert summaries[0]["status"] == "skipped"
    assert warnings and "Skipped optional source" in warnings[0]


def test_load_source_records_warns_on_empty_required_source(monkeypatch) -> None:
    monkeypatch.setattr(corpus_builder, "load_source_rows", lambda spec: [])
    loaded, summaries, warnings = load_source_records(
        [corpus_builder.SourceSpec(id="empty-local", kind="local", path="data/public/normalized/*.jsonl")],
        seed=1,
    )

    assert loaded == []
    assert summaries[0]["rows_loaded"] == 0
    assert warnings == ["Source 'empty-local' loaded 0 rows from data/public/normalized/*.jsonl."]


def test_load_source_records_warns_on_empty_optional_source(monkeypatch) -> None:
    monkeypatch.setattr(corpus_builder, "load_source_rows", lambda spec: [])
    loaded, summaries, warnings = load_source_records(
        [corpus_builder.SourceSpec(id="optional-empty", kind="local", path="data/public/normalized/*.jsonl", optional=True)],
        seed=1,
    )

    assert loaded == []
    assert summaries[0]["optional"] is True
    assert warnings == ["Optional source 'optional-empty' loaded 0 rows from data/public/normalized/*.jsonl."]


def test_preview_rows_includes_tokenization_preview() -> None:
    class FakeTokenizer:
        def tokenize(self, text: str):
            return text.replace("?", " ?").split()

    preview = preview_rows(
        [{"id": "1", "question": "Solve x + 1?", "solution": "x = 0", "difficulty": "hard", "topic": "calculus"}],
        fields=["question", "solution"],
        tokenizer=FakeTokenizer(),
        preview_length=3,
    )

    assert preview[0]["tokenization"]["fields"]["question"]["mode"] == "transformers"
    assert preview[0]["tokenization"]["fields"]["question"]["token_count"] >= 3
    assert preview[0]["tokenization"]["combined"]["token_preview"]
