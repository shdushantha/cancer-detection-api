from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import uvicorn

from app.routers import predict, gradcam, embeddings, uncertainty, metrics, full_analysis
from app.services.model_service import model_service

# EAGER_LOAD=1  → load all 3 models at startup (~1GB RAM; needs HF Spaces / >=2GB).
# EAGER_LOAD=0  → lazy-load each model on first request (low idle RAM; fits 512MB tiers).
EAGER_LOAD = os.getenv("EAGER_LOAD", "1") == "1"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Optionally load models on startup; always release on shutdown."""
    if EAGER_LOAD:
        print("🚀 Eager-loading all cancer detection models...")
        model_service.load_all_models()
        print("✅ All models loaded and ready.")
    else:
        print("💤 Lazy mode — models load on first request.")
    yield
    print("🛑 Shutting down — releasing models.")
    model_service.unload_all_models()

app = FastAPI(
    title="Cancer Detection API",
    description=(
        "Multi-cancer detection API exposing predictions, Grad-CAM heatmaps, "
        "feature embeddings, uncertainty estimation, performance metrics, and more "
        "for Lung, Skin, and Breast cancer models."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ───────────────────────────────────────
app.include_router(predict.router,       prefix="/api/v1", tags=["1. Prediction"])
app.include_router(gradcam.router,       prefix="/api/v1", tags=["2. Grad-CAM"])
app.include_router(embeddings.router,    prefix="/api/v1", tags=["3. Embeddings"])
app.include_router(uncertainty.router,   prefix="/api/v1", tags=["4. Uncertainty"])
app.include_router(metrics.router,       prefix="/api/v1", tags=["5. Metrics"])
app.include_router(full_analysis.router, prefix="/api/v1", tags=["6. Full Analysis"])

@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "online",
        "message": "Cancer Detection API is running.",
        "models_loaded": model_service.loaded_models(),
        "docs": "/docs",
    }

@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy", "models": model_service.loaded_models()}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
