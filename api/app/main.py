"""LeafScan REST API: plant disease diagnosis with explainable predictions."""
from __future__ import annotations

import json
import os
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .db import HistoryStore
from .inference import InvalidImageError, Predictor
from .preprocessing import LeafPreprocessor
from .schemas import BatchItem, BatchResponse, DiseaseInfo, HistoryPage, Prediction

log = logging.getLogger("leafscan")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

KNOWLEDGE: dict[str, dict] = json.loads(
    (Path(__file__).parent / "data" / "disease_info.json").read_text(encoding="utf-8"))

state: dict = {}


def _read_json(name: str):
    p = settings.model_dir / name
    return json.loads(p.read_text()) if p.exists() else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = time.time()
    state["predictor"] = Predictor(settings.model_dir, settings.onnx_threads)
    state["leafprep"] = LeafPreprocessor.from_model_dir(settings.model_dir)
    if state["leafprep"] is not None:
        log.info("leaf preprocessing enabled (%s)", state["leafprep"].cfg.get("segmenter"))
        if os.getenv("PRELOAD_SEGMENTER", "1") == "1":
            state["leafprep"].warm_up()
    state["history"] = HistoryStore(settings.db_path)
    state["metrics"] = _read_json("metrics.json")
    state["train"] = _read_json("train_history.json")
    state["dataset"] = _read_json("dataset_stats.json")
    missing = [c for c in state["predictor"].classes if c not in KNOWLEDGE]
    if missing:
        log.warning("no disease info for: %s", missing)
    log.info("model loaded in %.2fs (%d classes)", time.time() - t, len(state["predictor"].classes))
    yield
    state.clear()


app = FastAPI(
    title="LeafScan API",
    description="Plant leaf disease classification (38 diseases across 14 crops) with "
                "calibrated confidence, class-activation heatmaps and treatment guidance.",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                   allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1024)


def disease(key: str) -> dict:
    info = KNOWLEDGE.get(key)
    if info is None:
        crop, _, name = key.partition("___")
        info = {"crop": crop, "name": name.replace("_", " "), "type": "Unknown", "severity": "unknown",
                "summary": "", "symptoms": [], "treatment": [], "prevention": []}
    return {"key": key, **info}


async def _read_upload(file: UploadFile) -> bytes:
    limit = int(settings.max_upload_mb * 1024 * 1024)
    data = await file.read(limit + 1)
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > limit:
        raise HTTPException(413, f"Image larger than {settings.max_upload_mb:g} MB")
    return data


def _diagnose(data: bytes, filename: str | None, tta: bool, save: bool, heatmap: bool,
              crop: str | None = None) -> dict:
    p: Predictor = state["predictor"]
    t = time.perf_counter()
    img = p.load_image(data)
    q = p.quality(img)
    prep, model_img = None, img
    if state.get("leafprep") is not None:   # same preprocessing the model was trained on
        model_img, prep = state["leafprep"](img)
    if crop and crop not in p.crops():
        raise HTTPException(422, f"Unknown crop '{crop}'. Valid: {', '.join(p.crops())}")
    r = p.analyze(model_img, tta=tta, with_heatmap=heatmap, crop=crop or None)
    info = disease(r["class"])
    top_k = [{"key": k["class"], "crop": disease(k["class"])["crop"],
              "name": disease(k["class"])["name"], "probability": k["probability"]} for k in r["top_k"]]
    out = {
        "filename": filename, "status": r["status"], "confidence": r["confidence"],
        "entropy": r["entropy"], "tta": tta, "crop_filter": r["crop_filter"], "disease": info, "top_k": top_k,
        "quality": q.__dict__, "heatmap": r.get("heatmap"),
        "preprocessing": ({**prep, "image": LeafPreprocessor.preview(model_img)} if prep else None),
        "inference_ms": round((time.perf_counter() - t) * 1000, 1),
    }
    if save:
        out["id"] = state["history"].add(
            filename=filename, class_key=info["key"], crop=info["crop"], name=info["name"],
            status=r["status"], confidence=r["confidence"], top_k=top_k[:3],
            thumbnail=p.thumbnail(img))
    return out


# ---------------------------------------------------------------- routes
@app.get("/api/health", tags=["meta"])
def health():
    return {"status": "ok", "model_loaded": "predictor" in state,
            "classes": len(state["predictor"].classes) if "predictor" in state else 0}


def _model_name() -> str:
    cfg = (state.get("train") or {}).get("config", {})
    return cfg.get("backbone") or "LeafNet (MobileNetV3-Large, fine-tuned)"


def _dataset_name() -> str:
    m, ds = state.get("metrics") or {}, state.get("dataset") or {}
    n = sum(sum(v.values()) for v in ds.values()) if ds else None
    if m.get("per_source"):
        return f"{len(m['per_source'])} sources (PlantVillage + field datasets), {n:,} images" if n else "multi-source"
    return f"PlantVillage (Mendeley Data, {n:,} images)" if n else "PlantVillage"


@app.get("/api/model", tags=["meta"])
def model_info():
    """Model card: architecture, test metrics, training curves and dataset statistics."""
    return {
        "name": _model_name(),
        "framework": "PyTorch -> ONNX Runtime",
        "input": "RGB 224x224",
        "num_classes": len(state["predictor"].classes),
        "dataset": _dataset_name(),
        "metrics": state["metrics"],
        "training": state["train"],
        "dataset_stats": state["dataset"],
    }


@app.get("/api/crops", tags=["knowledge"])
def list_crops():
    """Crops the model knows: `id` is what /api/predict?crop= expects."""
    p: Predictor = state["predictor"]
    return [{"id": c, "name": disease(next(k for k in p.classes if k.startswith(c + "___")))["crop"]}
            for c in p.crops()]


@app.get("/api/diseases", response_model=list[DiseaseInfo], tags=["knowledge"])
def list_diseases(crop: str | None = None):
    items = [disease(k) for k in state["predictor"].classes]
    if crop:
        items = [d for d in items if d["crop"].lower().startswith(crop.lower())]
    return items


@app.get("/api/diseases/{key}", response_model=DiseaseInfo, tags=["knowledge"])
def get_disease(key: str):
    if key not in state["predictor"].classes:
        raise HTTPException(404, "Unknown class")
    return disease(key)


@app.post("/api/predict", response_model=Prediction, tags=["predict"])
async def predict(file: UploadFile = File(...),
                  tta: bool = Query(False, description="Average over flips/rotation for robustness"),
                  save: bool = Query(True, description="Store in history"),
                  heatmap: bool = Query(True, description="Return class activation heatmap"),
                  crop: str | None = Query(None, description="Known crop (e.g. 'Potato'): restricts the "
                                           "diagnosis to that crop's diseases. See /api/crops")):
    data = await _read_upload(file)
    try:
        return await run_in_threadpool(_diagnose, data, file.filename, tta, save, heatmap, crop)
    except InvalidImageError as e:
        raise HTTPException(415, str(e))


@app.post("/api/predict/batch", response_model=BatchResponse, tags=["predict"])
async def predict_batch(files: list[UploadFile] = File(...), tta: bool = False, save: bool = True,
                        crop: str | None = None):
    if len(files) > settings.max_batch:
        raise HTTPException(413, f"At most {settings.max_batch} images per batch")
    items, summary = [], {"healthy": 0, "diseased": 0, "uncertain": 0, "no_leaf": 0, "error": 0}
    for f in files:
        try:
            data = await _read_upload(f)
            r = await run_in_threadpool(_diagnose, data, f.filename, tta, save, False, crop)
            items.append(BatchItem(filename=f.filename, ok=True, result=r))
            summary[r["status"]] += 1
        except (HTTPException, InvalidImageError) as e:
            items.append(BatchItem(filename=f.filename, ok=False,
                                   error=getattr(e, "detail", None) or str(e)))
            summary["error"] += 1
    return BatchResponse(count=len(items), items=items, summary=summary)


@app.get("/api/history", response_model=HistoryPage, tags=["history"])
def history(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
            status: str | None = None):
    items, total = state["history"].list(limit, offset, status)
    return {"total": total, "items": items}


@app.get("/api/history/stats", tags=["history"])
def history_stats():
    return state["history"].stats()


@app.delete("/api/history/{pid}", tags=["history"])
def delete_history(pid: int):
    if not state["history"].delete(pid):
        raise HTTPException(404, "Not found")
    return {"deleted": pid}


@app.delete("/api/history", tags=["history"])
def clear_history():
    return {"deleted": state["history"].clear()}


# Serve the built React app from the same origin when present (single-container deploy).
if settings.frontend_dist.exists():
    app.mount("/", StaticFiles(directory=settings.frontend_dist, html=True), name="frontend")
