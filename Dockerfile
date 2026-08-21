# backend/Dockerfile — full image: the pipeline (scrape/extract/clean/embed/
# cluster) and/or the API. Installs the full requirements.txt (hdbscan,
# scikit-learn, trafilatura, ...) since either mode can reach the clustering
# code paths. No torch: embeddings come from the Gemini API by default.
#
# Build from backend/ itself — the frontend is a fully independent Next.js
# app now (its own process, its own Dockerfile), not something this image
# needs to contain:
#
#   docker build -f backend/Dockerfile -t sl-news-backend backend/
#
# Secrets (.env) are deliberately NOT copied into the image — pass them at
# run time instead (see docker-compose.yml, or `docker run --env-file`).

# ── Stage 1: build wheels with a full toolchain ─────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: slim runtime — only the installed packages, no compilers ──
FROM python:3.13-slim AS runtime

# libgomp1: the OpenMP runtime hdbscan/scikit-learn's compiled extensions
# link against at import time — needed here even though it wasn't needed
# to build the wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

WORKDIR /app
COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TRANSFORMERS_VERBOSITY=error

EXPOSE 8000

# No default CMD tied to one mode — docker-compose.yml sets the command per
# service (`python main.py` for the scraper loop, `python main.py --api`
# for the API server). This default just makes `docker run <image>` with no
# override do *something* sensible standalone.
# Overridden per deployment:
#   API            uvicorn api:app --host 0.0.0.0 --port $PORT
#   one pipeline   python pipeline.py        (Cloud Run Job / cron)
#   scheduler loop python main.py            (always-on host)
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
