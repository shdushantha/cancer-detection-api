"""
Tests for all Cancer Detection API endpoints.
Run with: pytest tests/ -v

Uses a tiny 10x10 white PNG as a dummy image — models must be loaded.
Set env var SKIP_MODEL_TESTS=1 to skip model-dependent tests.
"""
import os
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# ── Build a dummy image bytes fixture ─────────────────────
def make_dummy_image(size=(300, 300)) -> bytes:
    img = Image.new("RGB", size, color=(200, 150, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

DUMMY_PNG = make_dummy_image()
SKIP = os.getenv("SKIP_MODEL_TESTS", "0") == "1"

# ── Client fixture ─────────────────────────────────────────
@pytest.fixture(scope="session")
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


# ── Health ──────────────────────────────────────────────────
def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "status" in r.json()

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


# ── Predict ─────────────────────────────────────────────────
@pytest.mark.parametrize("cancer_type", ["lung", "skin", "breast"])
@pytest.mark.skipif(SKIP, reason="Model not loaded")
def test_predict(client, cancer_type):
    r = client.post(
        f"/api/v1/{cancer_type}/predict",
        files={"file": ("test.png", DUMMY_PNG, "image/png")},
    )
    assert r.status_code == 200
    data = r.json()
    assert "prediction" in data
    assert "confidence_percent" in data
    assert "confidence_scores" in data
    assert 0.0 <= data["confidence_percent"] <= 100.0


# ── Grad-CAM ────────────────────────────────────────────────
@pytest.mark.parametrize("cancer_type", ["lung", "skin", "breast"])
@pytest.mark.skipif(SKIP, reason="Model not loaded")
def test_gradcam(client, cancer_type):
    r = client.post(
        f"/api/v1/{cancer_type}/gradcam",
        files={"file": ("test.png", DUMMY_PNG, "image/png")},
    )
    assert r.status_code == 200
    data = r.json()
    assert "heatmap_base64" in data
    assert "overlay_base64" in data
    assert len(data["heatmap_base64"]) > 100  # non-empty base64


# ── Embeddings ──────────────────────────────────────────────
@pytest.mark.parametrize("cancer_type", ["lung", "skin", "breast"])
@pytest.mark.skipif(SKIP, reason="Model not loaded")
def test_embeddings(client, cancer_type):
    r = client.post(
        f"/api/v1/{cancer_type}/embeddings",
        files={"file": ("test.png", DUMMY_PNG, "image/png")},
    )
    assert r.status_code == 200
    data = r.json()
    assert "embedding_vector" in data
    assert data["embedding_dim"] == len(data["embedding_vector"])
    assert data["embedding_dim"] > 0


# ── Uncertainty ─────────────────────────────────────────────
@pytest.mark.parametrize("cancer_type", ["lung", "skin", "breast"])
@pytest.mark.skipif(SKIP, reason="Model not loaded")
def test_uncertainty(client, cancer_type):
    r = client.post(
        f"/api/v1/{cancer_type}/uncertainty?n_passes=5",
        files={"file": ("test.png", DUMMY_PNG, "image/png")},
    )
    assert r.status_code == 200
    data = r.json()
    assert "mean_probabilities" in data
    assert "uncertainty_std" in data
    assert "is_uncertain" in data
    assert data["uncertainty_level"] in ["low", "medium", "high"]


# ── Metrics ─────────────────────────────────────────────────
@pytest.mark.parametrize("cancer_type", ["lung", "skin", "breast"])
@pytest.mark.skipif(SKIP, reason="Model not loaded")
def test_get_metrics(client, cancer_type):
    r = client.get(f"/api/v1/{cancer_type}/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "accuracy" in data
    assert "confusion_matrix" in data
    assert "classification_report" in data


# ── Full Analysis ───────────────────────────────────────────
@pytest.mark.parametrize("cancer_type", ["lung", "skin", "breast"])
@pytest.mark.skipif(SKIP, reason="Model not loaded")
def test_full_analysis(client, cancer_type):
    r = client.post(
        f"/api/v1/{cancer_type}/full-analysis?mc_passes=5",
        files={"file": ("test.png", DUMMY_PNG, "image/png")},
    )
    assert r.status_code == 200
    data = r.json()
    # All features must be present
    for key in [
        "prediction", "confidence_scores", "heatmap_base64",
        "overlay_base64", "embedding_vector", "uncertainty_std", "is_uncertain"
    ]:
        assert key in data, f"Missing key: {key}"


# ── Invalid cancer type ─────────────────────────────────────
def test_invalid_cancer_type(client):
    r = client.post(
        "/api/v1/kidney/predict",
        files={"file": ("test.png", DUMMY_PNG, "image/png")},
    )
    assert r.status_code == 422  # validation error
