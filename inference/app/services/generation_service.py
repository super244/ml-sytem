from __future__ import annotations

from inference.app.config import AppSettings
from inference.app.generation import GenerationParameters, MathGenerator


class GenerationService:
    def __init__(self, generator: MathGenerator, settings: AppSettings):
        self.generator = generator
        self.settings = settings

    def generate(self, params: GenerationParameters) -> dict:
        return self.generator.generate(params)

    def compare(self, primary: GenerationParameters, secondary: GenerationParameters) -> dict:
        return {
            "primary": self.generator.generate(primary),
            "secondary": self.generator.generate(secondary),
        }
