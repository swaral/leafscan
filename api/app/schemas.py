from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Status = Literal["healthy", "diseased", "uncertain", "no_leaf"]


class DiseaseInfo(BaseModel):
    key: str
    crop: str
    name: str
    type: str
    pathogen: str | None = None
    severity: str
    summary: str
    symptoms: list[str] = []
    treatment: list[str] = []
    prevention: list[str] = []


class TopK(BaseModel):
    key: str
    crop: str
    name: str
    probability: float


class QualityReport(BaseModel):
    width: int
    height: int
    brightness: float
    sharpness: float
    warnings: list[str]


class Prediction(BaseModel):
    id: int | None = Field(None, description="History id when saved")
    filename: str | None = None
    status: Status
    confidence: float
    entropy: float
    tta: bool
    crop_filter: str | None = None
    disease: DiseaseInfo
    top_k: list[TopK]
    quality: QualityReport
    heatmap: str | None = Field(None, description="Class activation map, RGBA PNG data URL")
    preprocessing: dict | None = Field(None, description="Leaf preprocessing applied before the model "
                                       "(gate decision, timing, preview image) when the model requires it")
    inference_ms: float


class BatchItem(BaseModel):
    filename: str | None
    ok: bool
    error: str | None = None
    result: Prediction | None = None


class BatchResponse(BaseModel):
    count: int
    items: list[BatchItem]
    summary: dict[str, int]


class HistoryItem(BaseModel):
    id: int
    created_at: str
    filename: str | None
    class_key: str
    crop: str | None
    name: str | None
    status: Status
    confidence: float
    top_k: list[dict]
    thumbnail: str | None


class HistoryPage(BaseModel):
    total: int
    items: list[HistoryItem]
