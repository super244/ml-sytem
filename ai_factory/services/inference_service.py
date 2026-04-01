import random
import time
from uuid import uuid4
from ai_factory.schemas.inference import CompletionRequest, CompletionResponse


SIMULATED_COMPLETIONS = [
    "The key insight is that we can decompose this problem into smaller subproblems using dynamic programming. First, we define our state as dp[i][j] representing the optimal solution for the subarray from index i to j.",
    "Based on the training data analysis, the model shows strong performance on mathematical reasoning tasks with an accuracy of 78.4% on GSM8K. The chain-of-thought approach significantly improves step-by-step problem solving.",
    "To implement the LoRA adapter, we need to identify the target modules in the transformer architecture. Typically, we apply low-rank decomposition to the query and value projection matrices in each attention layer.",
    "The gradient accumulation strategy with a batch size of 4 and accumulation steps of 8 gives us an effective batch size of 32, which provides stable training dynamics while fitting within the VRAM constraints.",
    "Looking at the loss curve, we can observe that the model converges smoothly after the warmup phase. The learning rate schedule follows a cosine decay pattern, reaching the minimum at step 9500.",
]


class InferenceService:
    async def generate_completion(self, request: CompletionRequest) -> CompletionResponse:
        completion_text = random.choice(SIMULATED_COMPLETIONS)
        tokens = len(completion_text.split())
        tokens = min(tokens, request.max_tokens)

        return CompletionResponse(
            id=str(uuid4()),
            model_id=request.model_id,
            prompt=request.prompt,
            completion=completion_text,
            tokens_generated=tokens,
            tokens_per_second=round(random.uniform(25.0, 65.0), 1),
            time_to_first_token_ms=round(random.uniform(50.0, 200.0), 1),
            confidence_score=round(random.uniform(0.7, 0.95), 3),
            finish_reason="stop",
        )
