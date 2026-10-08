"""Vercel entrypoint for the compact brain MRI model service."""

import base64
import binascii

from fastapi import FastAPI, HTTPException

from schemas import MRIAnalysis, MRIRequest
from service import ImageValidationError, analyze_brain_mri

app = FastAPI(title="Vitalis brain MRI service", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "model": "brain-mri-resnet18-brats2020"}


@app.post("/mri", response_model=MRIAnalysis)
def mri(request: MRIRequest):
    try:
        payload = base64.b64decode(request.image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image encoding.") from exc
    if not payload:
        raise HTTPException(status_code=400, detail="Empty MRI image.")
    if len(payload) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="MRI image must be 5 MB or smaller.")
    try:
        return analyze_brain_mri(payload, request.health_report)
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Brain MRI model is unavailable. Verify its deployment and retry.",
        ) from exc
