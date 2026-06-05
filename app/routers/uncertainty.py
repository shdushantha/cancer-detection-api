"""
POST /api/v1/{cancer_type}/uncertainty
Returns: Monte Carlo Dropout uncertainty estimation.

Fix: the original code called tf.nn.softmax on model output, but the
model already has a softmax output layer — applying softmax again
squashes all probabilities toward 1/n_classes, making uncertainty
estimates meaningless.  We now read model output directly.
"""
import numpy as np
import tensorflow as tf
from fastapi import APIRouter, UploadFile, File, HTTPException, Path, Query

from app.services.model_service import model_service
from app.utils.image_utils import load_image_from_bytes
from app.schemas.responses import UncertaintyResponse
from app.config import settings

router = APIRouter()

CANCER_PATH = Path(..., description="Cancer type", pattern="^(lung|skin|breast)$")


def mc_dropout_predict(
    model: tf.keras.Model,
    tensor: np.ndarray,
    n_passes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run N stochastic forward passes with dropout enabled (training=True).
    Returns (mean_probs, std_probs) each of shape (n_classes,).

    Note: model output is already softmax — do NOT apply softmax again.
    """
    predictions = []
    for _ in range(n_passes):
        # training=True keeps Dropout layers active during inference
        output = model(tensor, training=True)
        # output is already a probability distribution (softmax applied in model)
        probs = output.numpy()[0]
        predictions.append(probs)

    predictions = np.array(predictions)          # (n_passes, n_classes)
    return predictions.mean(axis=0), predictions.std(axis=0)


def uncertainty_level(max_std: float) -> str:
    if max_std < 0.05:
        return "low"
    elif max_std < 0.10:
        return "medium"
    return "high"


@router.post(
    "/{cancer_type}/uncertainty",
    response_model=UncertaintyResponse,
    summary="Monte Carlo Dropout uncertainty estimation",
)
async def get_uncertainty(
    cancer_type: str = CANCER_PATH,
    file: UploadFile = File(..., description="Medical image (JPEG/PNG)"),
    n_passes: int = Query(
        default=30,
        ge=5,
        le=100,
        description="Number of MC Dropout forward passes (5–100). Higher = more accurate but slower.",
    ),
):
    """
    Runs the model N times with dropout **enabled** during inference (Monte Carlo Dropout).
    The standard deviation across passes measures prediction uncertainty:
    - **Low std** → model is confident
    - **High std** → model is uncertain — consider manual review

    This is especially important in medical settings to flag ambiguous cases.
    """
    try:
        img_size = model_service.get_img_size(cancer_type)
        classes  = model_service.get_classes(cancer_type)
        model    = model_service.get_model(cancer_type)

        raw_bytes   = await file.read()
        image_array = load_image_from_bytes(raw_bytes, img_size)
        tensor      = model_service.preprocess(image_array, cancer_type)

        mean_probs, std_probs = mc_dropout_predict(model, tensor, n_passes)

        pred_idx = int(np.argmax(mean_probs))
        max_std  = float(std_probs.max())

        return UncertaintyResponse(
            cancer_type=cancer_type,
            mean_probabilities={c: round(float(p), 6) for c, p in zip(classes, mean_probs)},
            uncertainty_std={c: round(float(s), 6) for c, s in zip(classes, std_probs)},
            prediction=classes[pred_idx],
            confidence_percent=round(float(mean_probs[pred_idx]) * 100, 2),
            is_uncertain=bool(max_std > settings.UNCERTAINTY_THRESHOLD),
            uncertainty_level=uncertainty_level(max_std),
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Uncertainty estimation failed: {str(e)}")
