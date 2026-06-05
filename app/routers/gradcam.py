"""
POST /api/v1/{cancer_type}/gradcam

Keras 3 / TF 2.20 compatible Grad-CAM.
- backbone.input/output used directly (no nested sub-model lookup)
- tape.watch(conv_out) AFTER forward pass (Keras 3 pattern)
"""
import numpy as np
import tensorflow as tf
import cv2
from fastapi import APIRouter, UploadFile, File, HTTPException, Path

from app.services.model_service import model_service
from app.utils.image_utils import load_image_from_bytes, encode_image_to_base64
from app.schemas.responses import GradCAMResponse

router = APIRouter()

CANCER_PATH = Path(..., description="Cancer type", pattern="^(lung|skin|breast)$")


def compute_gradcam(
    model: tf.keras.Model,
    tensor: np.ndarray,
    layer_name: str,
    pred_idx: int,
    cancer_type: str = None,
) -> np.ndarray:
    """
    Keras 3 compatible Grad-CAM.
    tape.watch() is called AFTER the forward pass on conv_out — not on the input.
    This is required in Keras 3 / TF 2.20 where frozen layers block input gradients.
    """
    target_layer = model.get_layer(layer_name)
    grad_model   = tf.keras.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output]
    )

    x_t = tf.constant(tensor, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x_t, training=False)
        tape.watch(conv_outputs)           # ← Keras 3: watch AFTER forward pass
        loss = predictions[:, pred_idx]

    grads        = tape.gradient(loss, conv_outputs)

    if grads is None:
        raise RuntimeError(
            f"GradientTape returned None for layer '{layer_name}'. "
            f"Ensure tape.watch(conv_outputs) is called after the forward pass."
        )

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam          = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1).numpy()

    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam /= cam.max()

    return cam.astype(np.float32)


def apply_heatmap_overlay(
    original_rgb: np.ndarray,
    cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    h, w         = original_rgb.shape[:2]
    cam_resized  = cv2.resize(cam, (w, h))
    heatmap      = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb  = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    overlay_bgr  = cv2.addWeighted(original_bgr, 0.6, heatmap, 0.4, 0)
    overlay_rgb  = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return heatmap_rgb, overlay_rgb


@router.post(
    "/{cancer_type}/gradcam",
    response_model=GradCAMResponse,
    summary="Generate Grad-CAM heatmap showing model focus regions",
)
async def gradcam(
    cancer_type: str = CANCER_PATH,
    file: UploadFile = File(..., description="Medical image (JPEG/PNG)"),
):
    try:
        img_size   = model_service.get_img_size(cancer_type)
        classes    = model_service.get_classes(cancer_type)
        model      = model_service.get_model(cancer_type)
        layer_name = model_service.get_gradcam_layer(cancer_type)

        raw_bytes   = await file.read()
        image_array = load_image_from_bytes(raw_bytes, img_size)
        tensor      = model_service.preprocess(image_array, cancer_type)

        probs    = model.predict(tensor, verbose=0)[0]
        pred_idx = int(np.argmax(probs))

        cam              = compute_gradcam(model, tensor, layer_name, pred_idx, cancer_type)
        heatmap, overlay = apply_heatmap_overlay(image_array, cam)
        cam_small        = cv2.resize(cam, (min(cam.shape[1], 50), min(cam.shape[0], 50)))

        return GradCAMResponse(
            cancer_type=cancer_type,
            prediction=classes[pred_idx],
            confidence_percent=round(float(probs[pred_idx]) * 100, 2),
            heatmap_base64=encode_image_to_base64(heatmap),
            overlay_base64=encode_image_to_base64(overlay),
            cam_scores=cam_small.tolist(),
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM failed: {str(e)}")