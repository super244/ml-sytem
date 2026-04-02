FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    libxml2-dev \
    libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY training/ ./training/
COPY ai_factory/ ./ai_factory/
COPY data/ ./data/

# Default entrypoint for training jobs
ENTRYPOINT ["python", "-m", "training.train"]
