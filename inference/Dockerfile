FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY inference/ ./inference/
COPY ai_factory/ ./ai_factory/

EXPOSE 8000

CMD ["uvicorn", "inference.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
