FROM python:3.11-slim

# ── System deps for OpenCV ────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps (layer caching) ───────────────────────────────────────
# Use tensorflow-cpu to roughly halve image size — no GPU on free tiers anyway.
COPY requirements.txt .
RUN sed 's/^tensorflow>=/tensorflow-cpu>=/' requirements.txt > requirements.cpu.txt \
    && pip install --no-cache-dir -r requirements.cpu.txt

# ── Application code + models ─────────────────────────────────────────
# Models live in models/ and are committed to the repo, so COPY bakes
# them into the image (no runtime volume needed — works on free PaaS).
COPY . .

RUN mkdir -p models metrics

# Many platforms (HF Spaces, Render, Railway) inject $PORT. Default 8000.
ENV PORT=7860
EXPOSE 7860

# Use shell form so $PORT is expanded at runtime.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
