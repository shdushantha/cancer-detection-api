from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from app.routers import predict, gradcam, embeddings, uncertainty, metrics, full_analysis
from app.services.model_service import model_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models on startup, release on shutdown."""
    print("🚀 Loading cancer detection models...")
    model_service.load_all_models()
    print("✅ All models loaded and ready.")
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
