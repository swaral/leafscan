"""Leaf preprocessing at inference time, identical to what the model was trained on.

Enabled automatically when the model folder contains pp_config.json (written by the
training pipeline for models trained on leaf-preprocessed data). Steps, matching
ml/scratch/build_dataset.py + ml/scratch/preprocess_dataset.py:
  1. resize to the dataset's short side (256 px; long side centre-cropped to <= 512)
  2. BiRefNet-lite leaf mask at 512 px input
  3. safe mask gate -> colour/lighting normalisation -> soft blurred background
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import threading
import time
from pathlib import Path

from PIL import Image

from .leafprep import leaf_preprocess

log = logging.getLogger("leafscan")


def resize_like_dataset(img: Image.Image, short_side: int = 256, max_long: int = 512) -> Image.Image:
    w, h = img.size
    s = short_side / min(w, h)
    nw, nh = round(w * s), round(h * s)
    if max(nw, nh) > max_long:
        img = img.resize((nw, nh), Image.BILINEAR)
        cw, ch = min(nw, max_long), min(nh, max_long)
        left, top = (nw - cw) // 2, (nh - ch) // 2
        return img.crop((left, top, left + cw, top + ch))
    return img.resize((nw, nh), Image.BILINEAR if s < 1 else Image.BICUBIC)


class LeafPreprocessor:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._seg = None
        self._lock = threading.Lock()

    @classmethod
    def from_model_dir(cls, model_dir: Path) -> "LeafPreprocessor | None":
        p = Path(model_dir) / "pp_config.json"
        return cls(json.loads(p.read_text())) if p.exists() else None

    def _segmenter(self):
        if self._seg is None:
            import torch
            from .birefnet import BiRefNet
            torch.set_num_threads(int(os.getenv("SEG_THREADS", os.cpu_count() or 4)))
            t = time.time()
            self._seg = BiRefNet(torch.device("cpu"), size=int(self.cfg.get("seg_input", 512)), half=False)
            log.info("leaf segmenter loaded in %.1fs", time.time() - t)
        return self._seg

    def warm_up(self) -> None:
        self._segmenter()

    def __call__(self, img: Image.Image) -> tuple[Image.Image, dict]:
        t = time.perf_counter()
        img = resize_like_dataset(img, int(self.cfg.get("dataset_short_side", 256)))
        with self._lock:  # one segmentation at a time keeps CPU memory bounded
            mask = self._segmenter().masks([img])[0]
        out, info = leaf_preprocess(img, mask, crop=bool(self.cfg.get("crop", False)))
        info["preprocess_ms"] = round((time.perf_counter() - t) * 1000, 1)
        return out, info

    @staticmethod
    def preview(img: Image.Image, size: int = 384) -> str:
        t = img.copy()
        t.thumbnail((size, size))
        buf = io.BytesIO()
        t.save(buf, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
