"""
ModelService — loads all three cancer models at startup,
exposes them by name, handles preprocessing per model.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from pathlib import Path

try:
    from keras.saving import register_keras_serializable
except ImportError:
    from tensorflow.keras.utils import register_keras_serializable

from app.config import settings


# ── Custom classes required to deserialize the skin model ─────────────────
# The skin model was saved with these classes registered under the
# "SkinCancer" package, so the registered names baked into the .keras file
# are "SkinCancer>EfficientNetPreprocess" and "SkinCancer>FocalLoss".
# The package= argument below MUST match what was used at training time.

@register_keras_serializable(package="SkinCancer")
class EfficientNetPreprocess(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trainable = False

    def call(self, inputs):
        return tf.keras.applications.efficientnet.preprocess_input(inputs)

    def get_config(self):
        return super().get_config()


@register_keras_serializable(package="SkinCancer")
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        ce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return ce * tf.pow(1 - p_t, self.gamma)

    def get_config(self):
        return {**super().get_config(), "gamma": self.gamma}


# Key by the fully-qualified registered names so load_model resolves them
# regardless of how the registry decorator ran. Bare names are kept as a
# fallback in case an older save used the unpackaged registration.
_SKIN_CUSTOM_OBJECTS = {
    "SkinCancer>EfficientNetPreprocess": EfficientNetPreprocess,
    "SkinCancer>FocalLoss": FocalLoss,
    "EfficientNetPreprocess": EfficientNetPreprocess,
    "FocalLoss": FocalLoss,
}


# ── Service ────────────────────────────────────────────────────────────────

class ModelService:
    def __init__(self):
        self._models: dict = {}

    # ── Lifecycle ──────────────────────────────────────────────
    # ── Internal single-model loader ───────────────────────────
    def _load_one(self, cancer_type: str):
        path = getattr(settings, f"{cancer_type.upper()}_MODEL_PATH")
        if not Path(path).exists():
            raise ValueError(f"{cancer_type} model not found at {path}")
        custom_objects = _SKIN_CUSTOM_OBJECTS if cancer_type == "skin" else None
        self._models[cancer_type] = tf.keras.models.load_model(
            path, custom_objects=custom_objects
        )

    def load_all_models(self):
        for cancer_type in ["lung", "skin", "breast"]:
            path = getattr(settings, f"{cancer_type.upper()}_MODEL_PATH")
            if not Path(path).exists():
                print(f"   ⚠️  {cancer_type} model not found at {path} — skipping.")
                continue

            print(f"   Loading {cancer_type} model from {path}...")
            try:
                custom_objects = _SKIN_CUSTOM_OBJECTS if cancer_type == "skin" else None
                self._models[cancer_type] = tf.keras.models.load_model(
                    path, custom_objects=custom_objects
                )
                print(f"   ✅ {cancer_type} model loaded.")
            except Exception as e:
                print(f"   ❌ Failed to load {cancer_type} model: {e}")

    def unload_all_models(self):
        self._models.clear()
        tf.keras.backend.clear_session()

    def loaded_models(self) -> list[str]:
        return list(self._models.keys())

    # ── Accessors ──────────────────────────────────────────────
    def get_model(self, cancer_type: str) -> tf.keras.Model:
        model = self._models.get(cancer_type)
        if model is None:
            # Lazy load on first use — keeps idle RAM low (one model at a time).
            print(f"   Lazy-loading {cancer_type} model...")
            self._load_one(cancer_type)
            model = self._models[cancer_type]
        return model

    def get_classes(self, cancer_type: str) -> list[str]:
        return getattr(settings, f"{cancer_type.upper()}_CLASSES")

    def get_img_size(self, cancer_type: str) -> tuple:
        return getattr(settings, f"{cancer_type.upper()}_IMG_SIZE")

    def get_gradcam_layer(self, cancer_type: str) -> str:
        return getattr(settings, f"{cancer_type.upper()}_GRADCAM_LAYER")

    def get_embed_layer(self, cancer_type: str) -> str:
        return getattr(settings, f"{cancer_type.upper()}_EMBED_LAYER")

    # ── Preprocessing ──────────────────────────────────────────
    def preprocess(self, image_array: np.ndarray, cancer_type: str) -> np.ndarray:
        """
        Apply model-specific preprocessing.
        Expects image_array in range [0, 255], shape (H, W, 3).
        Returns batched tensor (1, H, W, 3).
        """
        img_size = self.get_img_size(cancer_type)

        if image_array.shape[:2] != img_size:
            image_array = tf.image.resize(image_array, img_size).numpy()

        processed = preprocess_input(image_array.astype(np.float32))
        return np.expand_dims(processed, axis=0)


# ── Singleton ──────────────────────────────────────────────────────────────
model_service = ModelService()
