# ── Stage 1: build wheels ─────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8050

# Non-root user
RUN useradd --create-home appuser
WORKDIR /app

# Install pre-built wheels
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
 && rm -rf /wheels

# Copy source
COPY --chown=appuser:appuser . .

# Create data dir with correct permissions
RUN mkdir -p data && chown appuser:appuser data

USER appuser

EXPOSE $PORT

CMD gunicorn \
    -k uvicorn.workers.UvicornWorker \
    -w 1 \
    -b "0.0.0.0:${PORT}" \
    --timeout 300 \
    --graceful-timeout 30 \
    main:app
