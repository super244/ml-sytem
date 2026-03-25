import httpx
import pytest

from inference.app.main import app


class DummyGenerationService:
    def generate(self, params):
        return {
            "model_variant": params.model_variant,
            "prompt": "prompt",
            "answer": "Final Answer: 2",
            "raw_text": "Final Answer: 2",
            "final_answer": "2",
            "reasoning_steps": ["step"],
            "selected_score": 2.0,
            "candidates": [],
            "verification": None,
            "structured": None,
            "cache_hit": False,
            "telemetry_id": None,
            "latency_s": 0.01,
            "prompt_preset": params.prompt_preset,
            "candidate_agreement": 1.0,
        }

    def compare(self, primary, secondary):
        return {
            "primary": self.generate(primary),
            "secondary": self.generate(secondary),
        }


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_generate_endpoint(monkeypatch):
    from inference.app.routers import generation as generation_router

    monkeypatch.setattr(generation_router, "get_generation_service", lambda: DummyGenerationService())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/v1/generate",
            json={
                "question": "What is 1+1?",
                "model_variant": "base",
                "compare_to_base": False,
                "prompt_preset": "atlas_rigorous",
                "temperature": 0.2,
                "top_p": 0.95,
                "max_new_tokens": 64,
                "show_reasoning": True,
                "difficulty_target": "easy",
                "num_samples": 1,
                "use_calculator": True,
                "solver_mode": "rigorous",
                "output_format": "text",
                "use_cache": True,
            },
        )
    assert response.status_code == 200
    assert response.json()["final_answer"] == "2"
