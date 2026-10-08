"""Research-only 2D brain MRI classification using a compact ONNX runtime."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError

from schemas import HealthReport, MRIAnalysis, RankedLabel

MODEL_NAME = "brain-mri-resnet18-brats2020"
MODEL_PATH = Path(__file__).resolve().parent / "model.onnx"
LABELS = ("glioma", "meningioma", "no_tumor", "pituitary")
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


class ImageValidationError(ValueError):
    pass


SESSION = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


def prepare_image(payload: bytes) -> np.ndarray:
    try:
        with Image.open(BytesIO(payload)) as candidate:
            candidate.verify()
        with Image.open(BytesIO(payload)) as source:
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("The uploaded file is not a readable image.") from exc
    if min(image.size) < 32:
        raise ImageValidationError(
            "Upload a readable brain MRI image at least 32 pixels wide and high."
        )
    image = image.resize((224, 224), Image.Resampling.BILINEAR)
    tensor = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return ((tensor - MEAN) / STD)[None, ...].astype(np.float32)


def analyze_brain_mri(payload: bytes, report: HealthReport | None) -> MRIAnalysis:
    logits = SESSION.run(["logits"], {"image": prepare_image(payload)})[0][0]
    shifted = logits - np.max(logits)
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    scores = {
        label: round(float(score), 4)
        for label, score in zip(LABELS, probabilities)
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_label = ranked[0][0]
    if report:
        high_count = sum(signal.risk_level == "high" for signal in report.risk_signals)
        summary = (
            f"The MRI classifier's top label is {top_label}. The supplied Vitalis "
            f"report contains {len(report.risk_signals)} risk signals, including "
            f"{high_count} high-risk signals. This context does not alter, validate, "
            "or diagnose the MRI classifier output."
        )
    else:
        summary = (
            f"The MRI classifier's top label is {top_label}. No Vitalis health "
            "report was supplied; the model output is shown independently."
        )
    return MRIAnalysis(
        model=MODEL_NAME,
        predicted_label=top_label,
        class_scores=scores,
        ranked_labels=[RankedLabel(label=label, score=score) for label, score in ranked],
        integrated_summary=summary,
    )
