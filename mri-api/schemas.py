from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class RiskSignal(BaseModel):
    model_name: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=120)
    risk_level: Literal["low", "moderate", "high", "unknown"]
    score: Optional[float] = Field(default=None, ge=0, le=1)


class HealthReport(BaseModel):
    source: str = Field(default="vitalis", min_length=1, max_length=80)
    overall_health_score: Optional[float] = Field(default=None, ge=0, le=100)
    summary: Optional[str] = Field(default=None, max_length=1200)
    risk_signals: List[RiskSignal] = Field(default_factory=list, max_length=20)


class RankedLabel(BaseModel):
    label: Literal["glioma", "meningioma", "no_tumor", "pituitary"]
    score: float = Field(ge=0, le=1)


class MRIRequest(BaseModel):
    mime_type: Literal["image/jpeg", "image/png", "image/webp"]
    image_base64: str = Field(min_length=1, max_length=7_000_000)
    health_report: Optional[HealthReport] = None


class MRIAnalysis(BaseModel):
    model: str
    device: Literal["cpu"] = "cpu"
    input_type: Literal["brain_mri_2d_image"] = "brain_mri_2d_image"
    predicted_label: Literal["glioma", "meningioma", "no_tumor", "pituitary"]
    class_scores: Dict[str, float]
    ranked_labels: List[RankedLabel]
    integrated_summary: str
    clinical_review_required: bool = True
    disclaimer: str = (
        "Research and demonstration only. This is not a diagnosis, a radiology "
        "report, or medical advice. A qualified clinician must review original "
        "MRI data and clinical context."
    )
