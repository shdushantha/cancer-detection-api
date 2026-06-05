from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # ── Model paths ───────────────────────────────────────────
    LUNG_MODEL_PATH:   str = "models/lung_cancer_model_gradcam.keras"
    SKIN_MODEL_PATH:   str = "models/skin_cancer_b3_v3.keras"
    BREAST_MODEL_PATH: str = "models/breast_cancer_model_gradcam.keras"

    # ── Image sizes per model ─────────────────────────────────
    LUNG_IMG_SIZE:   tuple = (300, 300)
    SKIN_IMG_SIZE:   tuple = (224, 224)
    BREAST_IMG_SIZE: tuple = (300, 300)   # ← was (224,224) — breast model trained on 300x300

    # ── Class labels per model ────────────────────────────────
    LUNG_CLASSES:   list = ["lung_aca", "lung_n", "lung_scc"]
    SKIN_CLASSES:   list = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    BREAST_CLASSES: list = ["benign", "malignant", "normal"]

    # ── Grad-CAM layer names ──────────────────────────────────
    # top_activation = last activation in EfficientNetB3 backbone
    # Must match the layer name confirmed in debug_layers output
    LUNG_GRADCAM_LAYER:   str = "top_activation"   # ← was top_conv
    SKIN_GRADCAM_LAYER:   str = "top_activation"
    BREAST_GRADCAM_LAYER: str = "top_activation"   # ← was top_conv

    # ── Embedding layer ───────────────────────────────────────
    # head_gap = GlobalAveragePooling2D in the classification head
    LUNG_EMBED_LAYER:   str = "head_gap"            # ← was top_activation
    SKIN_EMBED_LAYER:   str = "head_gap"
    BREAST_EMBED_LAYER: str = "head_gap"            # ← was top_activation

    # ── Uncertainty ───────────────────────────────────────────
    MC_DROPOUT_PASSES:     int   = 30
    UNCERTAINTY_THRESHOLD: float = 0.10

    # ── TTA ───────────────────────────────────────────────────
    TTA_STEPS: int = 5

    class Config:
        env_file = ".env"

settings = Settings()

CANCER_TYPES = ["lung", "skin", "breast"]