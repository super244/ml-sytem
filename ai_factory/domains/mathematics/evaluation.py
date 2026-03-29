"""Mathematics domain evaluation suite."""

from pathlib import Path

from ai_factory.core.schemas import EvaluationSpec, MetricSpec


class MathEvaluationSuite:
    """Mathematics-specific evaluation capabilities."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self._metrics = self._load_math_metrics()
        self._benchmarks = self._load_math_benchmarks()
    
    def _load_math_metrics(self) -> dict[str, MetricSpec]:
        """Load mathematics-specific evaluation metrics."""
        return {
            "mathematical_accuracy": MetricSpec(
                name="mathematical_accuracy",
                description="Accuracy of final mathematical answers",
                type="accuracy",
                domain="mathematics"
            ),
            "step_correctness": MetricSpec(
                name="step_correctness", 
                description="Correctness of intermediate solution steps",
                type="accuracy",
                domain="mathematics"
            ),
            "verification_score": MetricSpec(
                name="verification_score",
                description="Agreement with mathematical verification",
                type="score",
                domain="mathematics",
                range=[0.0, 1.0]
            ),
            "calculus_fluency": MetricSpec(
                name="calculus_fluency",
                description="Quality of calculus reasoning and notation",
                type="score", 
                domain="mathematics",
                subdomain="calculus",
                range=[0.0, 1.0]
            ),
            "olympiad_reasoning": MetricSpec(
                name="olympiad_reasoning",
                description="Quality of olympiad-level mathematical reasoning",
                type="score",
                domain="mathematics", 
                subdomain="olympiad",
                range=[0.0, 1.0]
            )
        }
    
    def _load_math_benchmarks(self) -> dict[str, EvaluationSpec]:
        """Load mathematics benchmark specifications."""
        return {
            "mathematics_benchmark": EvaluationSpec(
                name="mathematics_benchmark",
                description="Comprehensive mathematics evaluation",
                domain="mathematics",
                datasets=["derivatives", "integrals", "limits_series", "olympiad_reasoning"],
                metrics=["mathematical_accuracy", "step_correctness", "verification_score"],
                splits=["test"],
                size=2000
            ),
            "calculus_specialist": EvaluationSpec(
                name="calculus_specialist",
                description="Calculus-focused evaluation",
                domain="mathematics",
                subdomain="calculus", 
                datasets=["derivatives", "integrals", "limits_series"],
                metrics=["mathematical_accuracy", "step_correctness", "calculus_fluency"],
                splits=["test"],
                size=1500
            ),
            "olympiad_reasoning": EvaluationSpec(
                name="olympiad_reasoning",
                description="Olympiad-level mathematical reasoning",
                domain="mathematics",
                subdomain="olympiad",
                datasets=["olympiad_reasoning"],
                metrics=["mathematical_accuracy", "step_correctness", "olympiad_reasoning"],
                splits=["test"],
                size=500
            )
        }
    
    def list_metrics(self) -> list[str]:
        """List all available mathematics metrics."""
        return list(self._metrics.keys())
    
    def list_benchmarks(self) -> list[str]:
        """List all available mathematics benchmarks."""
        return list(self._benchmarks.keys())
    
    def get_metric(self, name: str) -> MetricSpec:
        """Get a specific mathematics metric."""
        if name not in self._metrics:
            raise ValueError(f"Mathematics metric '{name}' not found")
        return self._metrics[name]
    
    def get_benchmark(self, name: str) -> EvaluationSpec:
        """Get a specific mathematics benchmark."""
        if name not in self._benchmarks:
            raise ValueError(f"Mathematics benchmark '{name}' not found")
        return self._benchmarks[name]
