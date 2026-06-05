from pydantic import BaseModel
from typing import Optional


# ── Shared ─────────────────────────────────────────────────
class CancerType(str):
    pass


# ── 1. Prediction ──────────────────────────────────────────
class PredictionResponse(BaseModel):
    cancer_type:       str
    prediction:        str
    confidence_scores: dict[str, float]
    confidence_percent: float
    top_prediction:    str


# ── 2. Grad-CAM ───────────────────────────────────────────
class GradCAMResponse(BaseModel):
    cancer_type:      str
    prediction:       str
    confidence_percent: float
    heatmap_base64:   str   # PNG image encoded as base64
    overlay_base64:   str   # Original image + heatmap overlay
    cam_scores:       list[list[float]]  # raw activation map (downsampled)


# ── 3. Embeddings ─────────────────────────────────────────
class EmbeddingResponse(BaseModel):
    cancer_type:      str
    prediction:       str
    embedding_dim:    int
    embedding_vector: list[float]


# ── 4. Uncertainty ────────────────────────────────────────
class UncertaintyResponse(BaseModel):
    cancer_type:         str
    mean_probabilities:  dict[str, float]
    uncertainty_std:     dict[str, float]
    prediction:          str
    confidence_percent:  float
    is_uncertain:        bool
    uncertainty_level:   str  # "low" | "medium" | "high"


# ── 5. Metrics ────────────────────────────────────────────
class ClassMetrics(BaseModel):
    precision: float
    recall:    float
    f1_score:  float
    support:   int

class MetricsResponse(BaseModel):
    cancer_type:            str
    accuracy:               float
    auc_roc:                float
    sensitivity:            float
    specificity:            float
    confusion_matrix:       list[list[int]]
    classification_report:  dict[str, ClassMetrics]
    note:                   Optional[str] = "Metrics are pre-computed on the held-out test set."


# ── 6. Full Analysis ──────────────────────────────────────
class FullAnalysisResponse(BaseModel):
    cancer_type:         str
    prediction:          str
    confidence_scores:   dict[str, float]
    confidence_percent:  float
    heatmap_base64:      str
    overlay_base64:      str
    embedding_dim:       int
    embedding_vector:    list[float]
    mean_probabilities:  dict[str, float]
    uncertainty_std:     dict[str, float]
    is_uncertain:        bool
    uncertainty_level:   str


# ── 7. Batch Prediction ───────────────────────────────────
class SingleBatchResult(BaseModel):
    filename:           str
    prediction:         str
    confidence_percent: float
    confidence_scores:  dict[str, float]
    error:              Optional[str] = None

class BatchPredictionResponse(BaseModel):
    cancer_type: str
    total:       int
    results:     list[SingleBatchResult]


# ── 8. Error ──────────────────────────────────────────────
class ErrorResponse(BaseModel):
    error:   str
    detail:  Optional[str] = None
