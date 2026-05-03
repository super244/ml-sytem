#!/usr/bin/env python3
"""Ensemble inference combining multiple specialized models."""

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class EnsembleInference:
    def __init__(self, model_configs: list[dict]):
        """Initialize ensemble with multiple models."""
        self.models = []
        self.tokenizers = []

        for config in model_configs:
            print(f"Loading model: {config['name']}")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config["base_model"], trust_remote_code=False, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load base model
            from typing import Any, cast
            model: Any = AutoModelForCausalLM.from_pretrained(
                config["base_model"],
                torch_dtype=torch.float32,
                device_map="auto" if torch.cuda.is_available() else {"": "cpu"},
            )

            # Load adapter if specified
            if config.get("adapter_path"):
                model = cast(Any, PeftModel.from_pretrained(model, config["adapter_path"], is_trainable=False))

            if torch.backends.mps.is_available():
                model = model.to("mps")

            model.eval()

            self.models.append(model)
            self.tokenizers.append(tokenizer)

    def predict(self, question: str, max_new_tokens: int = 512) -> list[dict]:
        """Get predictions from all models."""
        predictions = []

        for i, (model, tokenizer) in enumerate(zip(self.models, self.tokenizers, strict=True)):
            prompt = f"Question: {question}\n\nLet's solve this step by step.\n\n"

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            if torch.backends.mps.is_available():
                inputs = {k: v.to("mps") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append({"model_index": i, "response": response})

        return predictions

    def vote(self, question: str, max_new_tokens: int = 512) -> dict:
        """Get ensemble prediction by voting."""
        predictions = self.predict(question, max_new_tokens)

        # Extract final answers from responses
        answers = []
        for pred in predictions:
            response = pred["response"]
            # Simple extraction of boxed answer
            if "\\boxed{" in response:
                answer = response.split("\\boxed{")[1].split("}")[0]
                answers.append(answer)

        # Vote on most common answer
        if answers:
            from collections import Counter

            vote_counts = Counter(answers)
            best_answer = vote_counts.most_common(1)[0][0]
            confidence = vote_counts[best_answer] / len(answers)
        else:
            best_answer = predictions[0]["response"]
            confidence = 1.0 / len(predictions)

        return {"answer": best_answer, "confidence": confidence, "predictions": predictions}


def main() -> None:
    """Run ensemble inference on evaluation set."""
    # Define ensemble models
    model_configs = [
        {
            "name": "Accuracy Metal Shaders",
            "base_model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
            "adapter_path": "artifacts/runs/accuracy_ultimate_metal_shaders_95-20260408-161422/published/final_adapter",
        },
        {
            "name": "Accuracy Ultimate",
            "base_model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
            "adapter_path": "artifacts/runs/accuracy_ultimate_final-20260408-003835/published/final_adapter",
        },
        {
            "name": "Accuracy Perfect",
            "base_model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
            "adapter_path": "artifacts/runs/accuracy_perfect_final-20260408-014151/published/final_adapter",
        },
    ]

    # Initialize ensemble
    ensemble = EnsembleInference(model_configs)

    # Load evaluation questions
    eval_path = Path("data/processed/eval_combined.jsonl")
    questions = []
    with open(eval_path) as f:
        for line in f:
            data = json.loads(line)
            questions.append(data)

    print(f"Running ensemble inference on {len(questions)} questions...")

    # Run ensemble inference
    results = []
    for i, question_data in enumerate(questions[:10]):  # Test on first 10
        question = question_data["question"]
        print(f"Question {i + 1}/{min(10, len(questions))}")

        result = ensemble.vote(question)
        results.append(
            {
                "question": question,
                "ensemble_answer": result["answer"],
                "ensemble_confidence": result["confidence"],
                "ground_truth": question_data.get("final_answer", ""),
            }
        )

    # Save results
    output_path = Path("evaluation/results/ensemble_inference.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"Ensemble results saved to {output_path}")

    # Calculate accuracy
    correct = 0
    for result in results:
        if result["ensemble_answer"] == result["ground_truth"]:
            correct += 1

    accuracy = correct / len(results) if results else 0
    print(f"Ensemble accuracy: {accuracy:.1%}")


if __name__ == "__main__":
    main()
