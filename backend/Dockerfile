# AcousticSpace Backend - Week 4 (Deployment)
#
# Multi-stage build: install deps in a build layer, copy only what's
# needed into a slim runtime layer, so the final image isn't carrying
# pip's build cache / compiler toolchain around.

FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.12-slim

WORKDIR /app

# libsndfile1 is a runtime dependency of soundfile/librosa, not just a build-time one
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY app/ ./app/

# torch/transformers respect this to avoid over-subscribing CPU threads
# when multiple workers run in the same container
ENV OMP_NUM_THREADS=2
ENV TOKENIZERS_PARALLELISM=false

EXPOSE 8000

# --workers 1 by default: the model is loaded once per worker process,
# so more workers = more memory. Bump this via docker-compose/k8s once
# you've profiled actual load rather than guessing here.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
