from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Calculus 8B Inference API")


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.95
    stream: bool = False


# Placeholder for the loaded 8B model and tokenizer
# model = CalculusTransformer8B.from_pretrained("checkpoints/calculus-8b")
# tokenizer = CalculusTokenizer.from_pretrained("checkpoints/calculus-8b")


@app.post("/generate")
async def generate_solution(req: GenerateRequest):
    """
    Generate step-by-step calculus reasoning.
    Supports KV-caching in the underlying implementation.
    """
    if req.stream:
        raise HTTPException(status_code=400, detail="Streaming not yet implemented in this mock.")

    # Dummy inference generation showcasing the desired output format
    response_text = f"We are solving: {req.prompt}\n\nStep 1: Analyze the function.\nStep 2: Apply derivative rules.\nFinal Answer: Mock Response"

    return {"text": response_text, "usage": {"prompt_tokens": len(req.prompt.split()), "completion_tokens": 15}}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
