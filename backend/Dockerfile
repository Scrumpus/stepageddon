# syntax=docker/dockerfile:1.7

# Builder: install uv and resolve deps from uv.lock into a venv.
FROM python:3.12-slim-bookworm AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pinned uv release for reproducibility.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Runtime: slim image with just the venv + system audio libs.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# ffmpeg + libsndfile1 for librosa/yt-dlp; curl for HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY . .

RUN mkdir -p audio_storage \
    && useradd --create-home --uid 1000 app \
    && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Multi-worker uvicorn for handling concurrent generation requests.
# --proxy-headers + --forwarded-allow-ips=* so per-IP rate limiting sees the
# real client IP when running behind a reverse proxy.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--proxy-headers", "--forwarded-allow-ips=*"]
