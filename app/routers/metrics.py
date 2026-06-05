"""
GET  /api/v1/{cancer_type}/metrics          — retrieve stored metrics
POST /api/v1/{cancer_type}/metrics/compute  — compute metrics from a labelled dataset

Fix: sensitivity was incorrectly assigned macro-avg recall and
specificity was incorrectly assigned macro-avg precision.
Correct definitions:
  sensitivity (recall)   = TP / (TP + FN)  — per class, then macro-averaged
  specificity            = TN / (TN + FP)  — computed from confusion matrix
"""
import json
import numpy as np
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Path as FPath
from fastapi.responses import JSONResponse

from app.services.model_service import model_service
from app.utils.image_utils import load_image_from_bytes
from app.schemas.responses import MetricsResponse, ClassMetrics

router = APIRouter()

CANCER_PATH = FPath(..., description="Cancer type", pattern="^(lung|skin|breast)$")

METRICS_DIR = Path("metrics")
METRICS_DIR.mkdir(exist_ok=True)


def _metrics_path(cancer_type: str) -> Path:
    return METRICS_DIR / f"{cancer_type}_metrics.json"


def _default_metrics(cancer_type: str) -> dict:
    classes = model_service.get_classes(cancer_type)
    return {
        "cancer_type": cancer_type,
        "accuracy": 0.0,
        "auc_roc": 0.0,
        "sensitivity": 0.0,
        "specificity": 0.0,
        "confusion_matrix": [[0] * len(classes)] * len(classes),
        "classification_report": {
            cls: {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "support": 0}
            for cls in classes
        },
        "note": (
            "No metrics computed yet. "
            "POST to /{cancer_type}/metrics/compute with labelled test images."
        ),
    }


def _compute_specificity(cm: np.ndarray) -> float:
    """
    Macro-averaged specificity across all classes.
    For class i:  specificity_i = TN_i / (TN_i + FP_i)
    where TN_i = all correct predictions NOT in class i,
          FP_i = samples predicted as i but actually not i.
    """
    n = cm.shape[0]
    specs = []
    for i in range(n):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        denom = tn + fp
        specs.append(float(tn / denom) if denom > 0 else 0.0)
    return float(np.mean(specs))


@router.get(
    "/{cancer_type}/metrics",
    response_model=MetricsResponse,
    summary="Retrieve pre-computed performance metrics for a model",
)
async def get_metrics(cancer_type: str = CANCER_PATH):
    """
    Returns cached performance metrics (computed on the held-out test set).
    Run POST /{cancer_type}/metrics/compute first to populate real values.
    """
    try:
        model_service.get_model(cancer_type)   # validates model is loaded
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    path = _metrics_path(cancer_type)
    data = json.loads(path.read_text()) if path.exists() else _default_metrics(cancer_type)

    report = {
        cls: ClassMetrics(**v)
        for cls, v in data["classification_report"].items()
    }

    return MetricsResponse(
        cancer_type=cancer_type,
        accuracy=data["accuracy"],
        auc_roc=data["auc_roc"],
        sensitivity=data["sensitivity"],
        specificity=data["specificity"],
        confusion_matrix=data["confusion_matrix"],
        classification_report=report,
        note=data.get(
            "note",
            "Metrics are pre-computed on the held-out test set."
        ),
    )


@router.post(
    "/{cancer_type}/metrics/compute",
    summary="Compute & cache performance metrics from a labelled test set",
    response_class=JSONResponse,
)
async def compute_metrics(
    cancer_type: str = CANCER_PATH,
    files: list[UploadFile] = File(..., description="Test images"),
    labels: str = File(..., description="Comma-separated true labels matching file order"),
):
    """
    Upload labelled test images to compute and cache:
    - **Accuracy** — overall classification accuracy
    - **AUC-ROC** — one-vs-rest macro-averaged area under ROC curve
    - **Sensitivity** — macro-averaged recall (TP / (TP + FN))
    - **Specificity** — macro-averaged specificity (TN / (TN + FP))
    - Per-class Precision, Recall, F1
    - Confusion matrix

    **labels**: comma-separated class names matching files order, e.g.
    `lung_aca,lung_n,lung_scc,lung_aca`
    """
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        accuracy_score, roc_auc_score,
    )

    try:
        img_size = model_service.get_img_size(cancer_type)
        classes  = model_service.get_classes(cancer_type)
        model    = model_service.get_model(cancer_type)

        label_list = [l.strip() for l in labels.split(",")]
        if len(label_list) != len(files):
            raise HTTPException(
                status_code=400,
                detail=f"Mismatch: {len(files)} files but {len(label_list)} labels.",
            )

        y_true, y_pred, y_probs = [], [], []

        for f, lbl in zip(files, label_list):
            if lbl not in classes:
                raise HTTPException(status_code=400, detail=f"Unknown label: '{lbl}'")

            raw    = await f.read()
            arr    = load_image_from_bytes(raw, img_size)
            tensor = model_service.preprocess(arr, cancer_type)
            probs  = model.predict(tensor, verbose=0)[0]

            y_true.append(classes.index(lbl))
            y_pred.append(int(np.argmax(probs)))
            y_probs.append(probs.tolist())

        y_true  = np.array(y_true)
        y_pred  = np.array(y_pred)
        y_probs = np.array(y_probs)

        acc        = float(accuracy_score(y_true, y_pred))
        cm         = confusion_matrix(y_true, y_pred)
        report_raw = classification_report(
            y_true, y_pred, target_names=classes, output_dict=True
        )

        # ── Correct sensitivity & specificity ────────────────
        # Sensitivity = macro-averaged recall  (TP / (TP + FN))
        sensitivity = float(report_raw["macro avg"]["recall"])

        # Specificity = macro-averaged TN / (TN + FP) — derived from confusion matrix
        specificity = _compute_specificity(cm)

        # AUC (multi-class one-vs-rest)
        try:
            auc = float(roc_auc_score(y_true, y_probs, multi_class="ovr"))
        except Exception:
            auc = 0.0

        report_clean = {
            cls: {
                "precision": round(report_raw[cls]["precision"], 4),
                "recall":    round(report_raw[cls]["recall"],    4),
                "f1_score":  round(report_raw[cls]["f1-score"],  4),
                "support":   int(report_raw[cls]["support"]),
            }
            for cls in classes
        }

        payload = {
            "cancer_type":            cancer_type,
            "accuracy":               round(acc,         4),
            "auc_roc":                round(auc,         4),
            "sensitivity":            round(sensitivity, 4),
            "specificity":            round(specificity, 4),
            "confusion_matrix":       cm.tolist(),
            "classification_report":  report_clean,
            "note":                   "Metrics computed and cached ✅",
        }

        _metrics_path(cancer_type).write_text(json.dumps(payload, indent=2))

        return JSONResponse(content=payload)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics computation failed: {str(e)}")
