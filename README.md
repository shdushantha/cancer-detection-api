# 🫁 Cancer Detection API

Multi-cancer detection FastAPI service exposing **9 endpoints** for Lung, Skin, and Breast cancer models.

---

## 📁 Project Structure

```
cancer_api/
├── app/
│   ├── main.py              ← FastAPI app entry point
│   ├── config.py            ← Settings (model paths, class names, layer names)
│   ├── routers/
│   │   ├── predict.py       ← /predict + /predict/batch
│   │   ├── gradcam.py       ← /gradcam
│   │   ├── embeddings.py    ← /embeddings
│   │   ├── uncertainty.py   ← /uncertainty
│   │   ├── metrics.py       ← /metrics + /metrics/compute
│   │   └── full_analysis.py ← /full-analysis
│   ├── services/
│   │   └── model_service.py ← Model loader + preprocessor singleton
│   ├── schemas/
│   │   └── responses.py     ← Pydantic response models
│   └── utils/
│       └── image_utils.py   ← Image loading + base64 helpers
├── models/                  ← ⬅ Place your .h5 files here
├── metrics/                 ← Auto-created; stores computed metrics JSON
├── tests/
│   └── test_api.py
├── .env.example             ← Copy to .env and fill in paths
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt --user
```

### 2. Place your models
```
models/
  lung_cancer_model.h5
  skin_cancer_model.h5
  breast_cancer_model.h5
```

### 3. Configure
```bash
cp .env.example .env
# Edit .env — update model paths, class names, and layer names
```

**Finding your layer names:**
```python
import tensorflow as tf
model = tf.keras.models.load_model("models/lung_cancer_model.h5")
model.summary()
# Look for the last Conv2D layer name  → LUNG_GRADCAM_LAYER
# Look for the layer before Dense head → LUNG_EMBED_LAYER
```

### 4. Run
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Open API docs
```
http://localhost:8000/docs
```

---

## 🐳 Docker
```bash
docker build -t cancer-api .
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/metrics:/app/metrics \
  cancer-api
```

---

## 📡 API Endpoints

All endpoints follow the pattern:
```
/api/v1/{cancer_type}/{endpoint}
```
where `{cancer_type}` is one of: `lung`, `skin`, `breast`

| # | Method | Endpoint | Description |
|---|--------|----------|-------------|
| 1 | POST | `/{type}/predict` | Class label + confidence scores |
| 2 | POST | `/{type}/predict/batch` | Batch predictions (multiple images) |
| 3 | POST | `/{type}/gradcam` | Grad-CAM heatmap + overlay (base64 PNG) |
| 4 | POST | `/{type}/embeddings` | Feature embedding vector |
| 5 | POST | `/{type}/uncertainty` | Monte Carlo Dropout uncertainty |
| 6 | GET  | `/{type}/metrics` | Pre-computed performance metrics |
| 7 | POST | `/{type}/metrics/compute` | Compute & cache metrics from test set |
| 8 | POST | `/{type}/full-analysis` | All of the above in one request |
| 9 | GET  | `/health` | Health check + loaded models |

---

## 📖 Example Requests

### Predict (curl)
```bash
curl -X POST "http://localhost:8000/api/v1/lung/predict" \
  -F "file=@/path/to/image.jpg"
```

### Response
```json
{
  "cancer_type": "lung",
  "prediction": "lung_aca",
  "confidence_scores": {
    "lung_aca": 0.871,
    "lung_n":   0.093,
    "lung_scc": 0.036
  },
  "confidence_percent": 87.1,
  "top_prediction": "lung_aca"
}
```

### Full Analysis (Python)
```python
import requests

with open("scan.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/lung/full-analysis?mc_passes=20",
        files={"file": ("scan.jpg", f, "image/jpeg")}
    )

data = response.json()
print(data["prediction"])
print(data["confidence_percent"])
print(data["uncertainty_level"])  # "low" | "medium" | "high"

# Decode heatmap
import base64
from PIL import Image
import io
heatmap_bytes = base64.b64decode(data["overlay_base64"])
img = Image.open(io.BytesIO(heatmap_bytes))
img.show()
```

---

## 🔧 Customising Layer Names

If your models don't use EfficientNetB3, you need to update the layer names in `.env`:

| Architecture | Last Conv Layer | Pre-head Layer |
|---|---|---|
| EfficientNetB0–B7 | `top_conv` | `top_activation` |
| ResNet50 | `conv5_block3_out` | `avg_pool` |
| VGG16 | `block5_conv3` | `fc2` |
| InceptionV3 | `mixed10` | `avg_pool` |

---

## 🧪 Tests
```bash
pytest tests/ -v
```
To run without models:
```bash
SKIP_MODEL_TESTS=1 pytest tests/ -v
```
