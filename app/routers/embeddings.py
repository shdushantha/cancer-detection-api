"""
POST /api/v1/{cancer_type}/embeddings
Returns: feature embedding vector from penultimate layer.
"""
import numpy as np
import tensorflow as tf
from fastapi import APIRouter, UploadFile, File, HTTPException, Path

from app.services.model_service import model_service
from app.utils.image_utils import load_image_from_bytes
from app.schemas.responses import EmbeddingResponse

router = APIRouter()

CANCER_PATH = Path(..., description="Cancer type", pattern="^(lung|skin|breast)$")


def extract_embedding(model: tf.keras.Model, tensor: np.ndarray, layer_name: str) -> np.ndarray:
    """
    Extract feature vector from a named layer via a forward hook sub-model.
    Falls back gracefully if layer name is not found.
    """
    try:
        embed_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=model.get_layer(layer_name).output
        )
        features = embed_model.predict(tensor, verbose=0)
        # Flatten spatial dimensions if present (e.g. GlobalAvgPool already done)
        return features.squeeze()
    except ValueError:
        # Layer not found — use the second-to-last layer as fallback
        embed_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=model.layers[-2].output
        )
        features = embed_model.predict(tensor, verbose=0)
        return features.squeeze()


@router.post(
    "/{cancer_type}/embeddings",
    response_model=EmbeddingResponse,
    summary="Extract feature embedding vector from the model's penultimate layer",
)
async def get_embeddings(
    cancer_type: str = CANCER_PATH,
    file: UploadFile = File(..., description="Medical image (JPEG/PNG)"),
):
    """
    Returns a high-dimensional feature vector (embedding) from the layer just before
    the classifier head. Useful for:
    - Clustering similar cases (t-SNE / UMAP)
    - Nearest-neighbour search across a patient database
    - Transfer learning with custom downstream classifiers
    """
    try:
        img_size   = model_service.get_img_size(cancer_type)
        classes    = model_service.get_classes(cancer_type)
        model      = model_service.get_model(cancer_type)
        layer_name = model_service.get_embed_layer(cancer_type)

        raw_bytes   = await file.read()
        image_array = load_image_from_bytes(raw_bytes, img_size)
        tensor      = model_service.preprocess(image_array, cancer_type)

        # Prediction
        probs    = model.predict(tensor, verbose=0)[0]
        pred_idx = int(np.argmax(probs))

        # Embedding
        embedding = extract_embedding(model, tensor, layer_name)

        return EmbeddingResponse(
            cancer_type=cancer_type,
            prediction=classes[pred_idx],
            embedding_dim=int(embedding.shape[0]),
            embedding_vector=[round(float(v), 8) for v in embedding],
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding extraction failed: {str(e)}")
