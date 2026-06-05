"""
POST /api/v1/{cancer_type}/predict
Returns: class label + confidence scores for all classes.
"""
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, Path

from app.services.model_service import model_service
from app.utils.image_utils import load_image_from_bytes
from app.schemas.responses import PredictionResponse, BatchPredictionResponse, SingleBatchResult
from app.config import CANCER_TYPES

router = APIRouter()

CANCER_PATH = Path(..., description="Cancer type", pattern="^(lung|skin|breast)$")


@router.post(
    "/{cancer_type}/predict",
    response_model=PredictionResponse,
    summary="Predict cancer class from a single image",
)
async def predict(
    cancer_type: str = CANCER_PATH,
    file: UploadFile = File(..., description="Medical image (JPEG/PNG)"),
):
    """
    Upload a medical image and receive:
    - Predicted class label
    - Confidence scores for every class
    - Top prediction with confidence %
    """
    try:
        img_size    = model_service.get_img_size(cancer_type)
        classes     = model_service.get_classes(cancer_type)
        model       = model_service.get_model(cancer_type)

        raw_bytes   = await file.read()
        image_array = load_image_from_bytes(raw_bytes, img_size)
        tensor      = model_service.preprocess(image_array, cancer_type)

        probs       = model.predict(tensor, verbose=0)[0]
        pred_idx    = int(np.argmax(probs))

        return PredictionResponse(
            cancer_type=cancer_type,
            prediction=classes[pred_idx],
            confidence_scores={cls: round(float(p), 6) for cls, p in zip(classes, probs)},
            confidence_percent=round(float(probs[pred_idx]) * 100, 2),
            top_prediction=classes[pred_idx],
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post(
    "/{cancer_type}/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Predict cancer class for multiple images",
)
async def predict_batch(
    cancer_type: str = CANCER_PATH,
    files: list[UploadFile] = File(..., description="Multiple medical images"),
):
    """
    Upload multiple images at once and receive predictions for each.
    Failed images are reported individually without failing the whole batch.
    """
    img_size = model_service.get_img_size(cancer_type)
    classes  = model_service.get_classes(cancer_type)
    model    = model_service.get_model(cancer_type)
    results  = []

    for f in files:
        try:
            raw       = await f.read()
            arr       = load_image_from_bytes(raw, img_size)
            tensor    = model_service.preprocess(arr, cancer_type)
            probs     = model.predict(tensor, verbose=0)[0]
            pred_idx  = int(np.argmax(probs))

            results.append(SingleBatchResult(
                filename=f.filename,
                prediction=classes[pred_idx],
                confidence_percent=round(float(probs[pred_idx]) * 100, 2),
                confidence_scores={c: round(float(p), 6) for c, p in zip(classes, probs)},
            ))
        except Exception as e:
            results.append(SingleBatchResult(
                filename=f.filename,
                prediction="",
                confidence_percent=0.0,
                confidence_scores={},
                error=str(e),
            ))

    return BatchPredictionResponse(
        cancer_type=cancer_type,
        total=len(results),
        results=results,
    )
