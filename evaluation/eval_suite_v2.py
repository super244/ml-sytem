#!/usr/bin/env python3
"""Titan Evaluation Suite v2.0 - Comprehensive benchmark evaluation

Supports:
- Standard LLM benchmarks (MMLU, GSM8K, HumanEval, etc.)
- Custom task evaluation
- Distributed evaluation across multiple GPUs
- Async batch processing
- Real-time metrics streaming
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai_factory.core.runtime import HardwareDetector, TrainingOptimizer  # noqa: E402
from evaluation.benchmark_registry import BenchmarkRegistry  # noqa: E402
from evaluation.error_taxonomy import ErrorTaxonomy  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


class EvaluationEngine:
    """High-performance evaluation engine with hardware optimization."""

    def __init__(self, model_path: str, batch_size: int | None = None):
        self.hardware = HardwareDetector.detect()
        self.optimizer = TrainingOptimizer(self.hardware)
        self.optimizer.configure_torch()

        # Auto-tune batch size
        self.batch_size = batch_size or self.hardware.recommended_batch_size

        logger.info(f"Loading model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        model_kwargs = {
            "torch_dtype": torch.bfloat16 if self.hardware.supports_bf16 else torch.float16,
            "device_map": "auto",
        }

        if self.hardware.supports_fp8:
            model_kwargs["quantization_config"] = {"load_in_8bit": False}  # FP8 handled separately

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

        # Apply torch.compile for better performance
        if self.hardware.pytorch_compile:
            self.model = self.optimizer.apply_model_optimizations(self.model)

        self.registry = BenchmarkRegistry()
        self.error_taxonomy = ErrorTaxonomy()

    async def evaluate_benchmark(
        self,
        benchmark_name: str,
        num_samples: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate a single benchmark asynchronously."""
        benchmark = self.registry.get(benchmark_name)

        logger.info(f"Starting evaluation: {benchmark_name}")

        results = []
        dataset = benchmark.load_dataset()

        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        # Process in batches
        for i in range(0, len(dataset), self.batch_size):
            batch = dataset[i : i + self.batch_size]
            batch_results = await self._evaluate_batch(batch, benchmark)
            results.extend(batch_results)

        # Compute metrics
        metrics = benchmark.compute_metrics(results)

        return {
            "benchmark": benchmark_name,
            "num_samples": len(results),
            "metrics": metrics,
            "hardware": self.hardware.to_dict(),
        }

    async def _evaluate_batch(
        self,
        batch: list[dict],
        benchmark,
    ) -> list[dict]:
        """Evaluate a batch of examples."""
        results = []

        for example in batch:
            try:
                # Generate response
                inputs = self.tokenizer(
                    example["input"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to(self.model.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.7,
                        do_sample=True,
                    )

                response = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1] :],
                    skip_special_tokens=True,
                )

                # Evaluate
                result = benchmark.evaluate_example(example, response)
                results.append(result)

            except Exception as e:
                logger.error(f"Error evaluating example: {e}")
                self.error_taxonomy.classify_error(e, example)
                results.append({"correct": False, "error": str(e)})

        return results

    async def evaluate_all(
        self,
        benchmarks: list[str] | None = None,
        num_samples: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate all registered benchmarks."""
        if benchmarks is None:
            benchmarks = self.registry.list_benchmarks()

        all_results = {}

        for benchmark_name in benchmarks:
            try:
                result = await self.evaluate_benchmark(benchmark_name, num_samples)
                all_results[benchmark_name] = result
            except Exception as e:
                logger.error(f"Failed to evaluate {benchmark_name}: {e}")
                all_results[benchmark_name] = {"error": str(e)}

        return {
            "model": str(self.model.name_or_path),
            "hardware": self.hardware.to_dict(),
            "results": all_results,
            "summary": self._compute_summary(all_results),
        }

    def _compute_summary(self, results: dict) -> dict:
        """Compute overall evaluation summary."""
        total_correct = 0
        total_samples = 0

        for _benchmark, result in results.items():
            if "error" in result:
                continue

            metrics = result.get("metrics", {})
            if "accuracy" in metrics:
                total_correct += metrics["accuracy"] * result["num_samples"]
                total_samples += result["num_samples"]

        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0

        return {
            "overall_accuracy": overall_accuracy,
            "total_samples": total_samples,
            "num_benchmarks": len(results),
        }


def main():
    parser = argparse.ArgumentParser(description="Titan Evaluation Suite v2.0")
    parser.add_argument("--model", required=True, help="Model path or HuggingFace ID")
    parser.add_argument("--benchmarks", nargs="+", help="Benchmarks to run (default: all)")
    parser.add_argument("--num-samples", type=int, help="Number of samples per benchmark")
    parser.add_argument("--batch-size", type=int, help="Evaluation batch size")
    parser.add_argument("--output", default="evaluation_results.json", help="Output file")
    parser.add_argument("--hardware-summary", action="store_true", help="Print hardware summary and exit")

    args = parser.parse_args()

    # Print hardware summary if requested
    if args.hardware_summary:
        hardware = HardwareDetector.detect()
        print(json.dumps(hardware.to_dict(), indent=2))
        return

    # Initialize evaluation engine
    engine = EvaluationEngine(args.model, args.batch_size)

    # Run evaluation
    results = asyncio.run(engine.evaluate_all(args.benchmarks, args.num_samples))

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Model: {results['model']}")
    print(f"Hardware: {results['hardware']['device_name']}")
    print(f"Overall Accuracy: {results['summary']['overall_accuracy']:.2%}")
    print(f"Total Samples: {results['summary']['total_samples']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
