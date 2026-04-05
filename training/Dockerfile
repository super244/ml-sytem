FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    libxml2-dev \
    libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY ai_factory/ ./ai_factory/
COPY training/ ./training/
COPY data/ ./data/
COPY evaluation/ ./evaluation/
COPY configs/ ./configs/

RUN pip install --upgrade pip && pip install -e .

# Default entrypoint for training jobs
ENTRYPOINT ["python", "-m", "training.train"]
