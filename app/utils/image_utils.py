"""
Shared image utilities: load from upload bytes, encode to base64, etc.
"""
import io
import base64
import numpy as np
from PIL import Image


def load_image_from_bytes(file_bytes: bytes, target_size: tuple) -> np.ndarray:
    """
    Load image from raw bytes, resize, return numpy array (H, W, 3) uint8.
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(target_size, Image.LANCZOS)
    return np.array(img)


def encode_image_to_base64(image_array: np.ndarray) -> str:
    """
    Encode a numpy image array (uint8 RGB) to a base64 PNG string.
    """
    img = Image.fromarray(image_array.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def normalize_for_display(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 255] uint8 for display."""
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    return (arr * 255).astype(np.uint8)
