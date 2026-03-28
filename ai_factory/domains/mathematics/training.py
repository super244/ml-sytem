"""Mathematics domain training profiles."""

from typing import Dict, List, Any
from pathlib import Path

from ai_factory.core.schemas import TrainingProfileSpec


class MathTrainingProfiles:
    """Mathematics-specific training configurations."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self._profiles = self._load_math_profiles()
    
    def _load_math_profiles(self) -> Dict[str, TrainingProfileSpec]:
        """Load mathematics training profiles."""
        return {
            "baseline_qlora": TrainingProfileSpec(
                name="baseline_qlora",
                description="Baseline QLoRA training for mathematics",
                domain="mathematics",
                training_method="qlora",
                datasets=["derivatives", "integrals", "limits_series"],
                config_path="training/configs/profiles/baseline_qlora.yaml",
                model_requirements={
                    "architecture": "transformer",
                    "min_parameters": 1_000_000_000,  # 1B
                    "math_specialization": True
                }
            ),
            "calculus_specialist": TrainingProfileSpec(
                name="calculus_specialist",
                description="Specialized calculus training",
                domain="mathematics",
                subdomain="calculus",
                training_method="qlora",
                datasets=["derivatives", "integrals", "limits_series", "multivariable"],
                config_path="training/configs/profiles/calculus_specialist.yaml",
                model_requirements={
                    "architecture": "transformer",
                    "min_parameters": 3_000_000_000,  # 3B
                    "math_specialization": True
                }
            ),
            "olympiad_reasoning": TrainingProfileSpec(
                name="olympiad_reasoning",
                description="Olympiad-level reasoning training",
                domain="mathematics",
                subdomain="olympiad",
                training_method="full_finetune",
                datasets=["olympiad_reasoning"],
                config_path="training/configs/profiles/olympiad_reasoning.yaml",
                model_requirements={
                    "architecture": "transformer",
                    "min_parameters": 7_000_000_000,  # 7B
                    "math_specialization": True,
                    "long_context": True
                }
            ),
            "mathematics_curriculum": TrainingProfileSpec(
                name="mathematics_curriculum",
                description="Curriculum-based mathematics training",
                domain="mathematics",
                training_method="qlora",
                datasets=["derivatives", "integrals", "limits_series", "olympiad_reasoning"],
                config_path="training/configs/profiles/mathematics_curriculum.yaml",
                curriculum_order=[
                    "derivatives",
                    "integrals", 
                    "limits_series",
                    "olympiad_reasoning"
                ],
                model_requirements={
                    "architecture": "transformer",
                    "min_parameters": 3_000_000_000,  # 3B
                    "math_specialization": True
                }
            )
        }
    
    def list_profiles(self) -> List[str]:
        """List all available mathematics training profiles."""
        return list(self._profiles.keys())
    
    def get_profile(self, name: str) -> TrainingProfileSpec:
        """Get a specific mathematics training profile."""
        if name not in self._profiles:
            raise ValueError(f"Mathematics training profile '{name}' not found")
        return self._profiles[name]
    
    def get_profiles_by_method(self, method: str) -> List[TrainingProfileSpec]:
        """Get training profiles by training method."""
        return [
            profile for profile in self._profiles.values()
            if profile.training_method == method
        ]
