"""
POST /api/v1/{cancer_type}/full-analysis
Keras 3 / TF 2.20 compatible — uses updated compute_gradcam with correct tape pattern.
"""
import numpy as np
import tensorflow as tf
from fastapi import APIRouter, UploadFile, File, HTTPException, Path, Query

from app.services.model_service import model_service
from app.utils.image_utils import load_image_from_bytes, encode_image_to_base64
from app.schemas.responses import FullAnalysisResponse
from app.config import settings

from app.routers.gradcam import compute_gradcam, apply_heatmap_overlay
from app.routers.embeddings import extract_embedding
from app.routers.uncertainty import mc_dropout_predict, uncertainty_level

router = APIRouter()

CANCER_PATH = Path(..., description="Cancer type", pattern="^(lung|skin|breast)$")


@router.post(
    "/{cancer_type}/full-analysis",
    response_model=FullAnalysisResponse,
    summary="Full analysis: prediction + Grad-CAM + embeddings + uncertainty",
)
async def full_analysis(
    cancer_type: str = CANCER_PATH,
    file: UploadFile = File(..., description="Medical image (JPEG/PNG)"),
    mc_passes: int = Query(default=20, ge=5, le=50),
):
    try:
        img_size  = model_service.get_img_size(cancer_type)
        classes   = model_service.get_classes(cancer_type)
        model     = model_service.get_model(cancer_type)
        cam_layer = model_service.get_gradcam_layer(cancer_type)
        emb_layer = model_service.get_embed_layer(cancer_type)

        raw_bytes   = await file.read()
        image_array = load_image_from_bytes(raw_bytes, img_size)
        tensor      = model_service.preprocess(image_array, cancer_type)

        # 1. Prediction
        probs    = model.predict(tensor, verbose=0)[0]
        pred_idx = int(np.argmax(probs))

        # 2. Grad-CAM (Keras 3 tape pattern inside compute_gradcam)
        cam              = compute_gradcam(model, tensor, cam_layer, pred_idx, cancer_type)
        heatmap, overlay = apply_heatmap_overlay(image_array, cam)

        # 3. Embeddings
        embedding = extract_embedding(model, tensor, emb_layer)

        # 4. MC Uncertainty
        mean_probs, std_probs = mc_dropout_predict(model, tensor, mc_passes)
        max_std = float(std_probs.max())

        return FullAnalysisResponse(
            cancer_type=cancer_type,
            prediction=classes[pred_idx],
            confidence_scores={c: round(float(p), 6) for c, p in zip(classes, probs)},
            confidence_percent=round(float(probs[pred_idx]) * 100, 2),
            heatmap_base64=encode_image_to_base64(heatmap),
            overlay_base64=encode_image_to_base64(overlay),
            embedding_dim=int(embedding.shape[0]),
            embedding_vector=[round(float(v), 8) for v in embedding],
            mean_probabilities={c: round(float(p), 6) for c, p in zip(classes, mean_probs)},
            uncertainty_std={c: round(float(s), 6) for c, s in zip(classes, std_probs)},
            is_uncertain=bool(max_std > settings.UNCERTAINTY_THRESHOLD),
            uncertainty_level=uncertainty_level(max_std),
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Full analysis failed: {str(e)}")