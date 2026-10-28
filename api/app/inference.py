"""ONNX Runtime inference: preprocessing, prediction, TTA, CAM heatmaps, image quality checks."""
from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, UnidentifiedImageError

IMG_SIZE = 224
BACKGROUND = "Background_without_leaves"

# Decision thresholds. Defaults are conservative and were checked on real leaves vs flat/blank
# images; a thresholds.json next to the model (e.g. fitted on field validation data) overrides them.
DEFAULT_THRESHOLDS = {
    "min_side": 64,                  # px; smaller photos cannot be diagnosed
    "min_brightness": 20,            # mean grey level 0..255
    "max_brightness": 245,
    "min_sharpness": 3.0,            # variance of Laplacian; flat colour = 0, smoothest real leaf ~17
    "min_plant_fraction": 0.05,      # share of plant-coloured pixels; real leaves 0.25-0.7
    "uncertain_below": 0.55,         # calibrated confidence needed for a diagnosis
    "uncertain_below_with_crop": 0.55,
    "crop_mismatch_below": 0.02,     # probability mass on the chosen crop (before filtering)
}


class InvalidImageError(ValueError):
    pass


@dataclass
class Quality:
    width: int
    height: int
    brightness: float
    sharpness: float
    warnings: list[str]


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _colormap(v: np.ndarray) -> np.ndarray:
    """Map [0,1] -> RGB with a perceptual 'inferno'-like ramp (dark purple -> yellow)."""
    stops = np.array([
        [0.00, 0, 0, 4], [0.25, 87, 16, 110], [0.50, 188, 55, 84],
        [0.75, 249, 142, 9], [1.00, 252, 255, 164],
    ], dtype=np.float32)
    rgb = np.stack([np.interp(v, stops[:, 0], stops[:, i]) for i in (1, 2, 3)], axis=-1)
    return rgb.astype(np.uint8)


class Predictor:
    def __init__(self, model_dir: str | Path, threads: int = 0):
        model_dir = Path(model_dir)
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_dir / "leafnet.onnx"), opts,
                                            providers=["CPUExecutionProvider"])
        self.classes: list[str] = json.loads((model_dir / "labels.json").read_text())
        head = np.load(model_dir / "cam_head.npz")
        self.cam_weight = head["weight"]  # (C, 960)
        self.thresholds = dict(DEFAULT_THRESHOLDS)
        if (model_dir / "thresholds.json").exists():
            self.thresholds.update(json.loads((model_dir / "thresholds.json").read_text()))

    # ---------- image handling ----------
    @staticmethod
    def load_image(data: bytes) -> Image.Image:
        try:
            img = Image.open(io.BytesIO(data))
            img = ImageOps.exif_transpose(img)  # respect phone camera orientation
            return img.convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise InvalidImageError("File is not a readable image") from e

    @staticmethod
    def preprocess(img: Image.Image) -> np.ndarray:
        """Same as training: bilinear resize to 224x224, scale to [0,1], CHW."""
        arr = np.asarray(img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
        return arr.transpose(2, 0, 1)

    @staticmethod
    def quality(img: Image.Image) -> Quality:
        w, h = img.size
        g = img.convert("L")
        g.thumbnail((512, 512))
        a = np.asarray(g, dtype=np.float32)
        brightness = float(a.mean())
        # Variance of the Laplacian: low values mean few sharp edges (blur). Tiny images: 0.
        if min(a.shape) < 3:
            sharpness = 0.0
        else:
            lap = (-4 * a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:])
            sharpness = float(lap.var())
        warnings = []
        if min(w, h) < 128:
            warnings.append("Low resolution: use a photo at least 256 px on the short side.")
        if brightness < 45:
            warnings.append("Image is very dark: retake in brighter, even light.")
        elif brightness > 225:
            warnings.append("Image is overexposed: avoid direct glare.")
        if sharpness < 25:
            warnings.append("Image looks blurry: hold the camera steady and tap to focus.")
        return Quality(w, h, round(brightness, 1), round(sharpness, 1), warnings)

    # ---------- model ----------
    def _run(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits, feats = self.session.run(["logits", "features"], {"image": batch.astype(np.float32)})
        return logits, feats

    def predict_proba(self, batch: np.ndarray) -> np.ndarray:
        return _softmax(self._run(batch)[0])

    def crops(self) -> list[str]:
        return sorted({c.split("___")[0] for c in self.classes if c != BACKGROUND})

    @staticmethod
    def plant_fraction(img: Image.Image) -> float:
        """Share of plant-coloured pixels (green through yellow/brown), on a small thumbnail."""
        t = img.copy()
        t.thumbnail((256, 256))
        hsv = np.asarray(t.convert("HSV"), dtype=np.float32)
        h, s, v = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
        return float(((h >= 15) & (h <= 170) & (s >= 0.18) & (v >= 0.12)).mean())

    def photo_problem(self, q: "Quality") -> str | None:
        """Reason the photo cannot be diagnosed at all, or None."""
        th = self.thresholds
        if min(q.width, q.height) < th["min_side"]:
            return "The photo is too small to diagnose. Use a photo at least 256 px on the short side."
        if q.brightness < th["min_brightness"]:
            return "The photo is too dark to diagnose. Retake it in brighter, even light."
        if q.brightness > th["max_brightness"]:
            return "The photo is overexposed. Retake it without direct glare."
        if q.sharpness < th["min_sharpness"]:
            return "The photo shows no leaf detail (blank, flat or extremely blurred). Retake it in focus."
        return None

    def analyze(self, img: Image.Image, tta: bool = False, top_k: int = 5,
                with_heatmap: bool = True, crop: str | None = None, quality: "Quality | None" = None) -> dict:
        x = self.preprocess(img)
        views = [x]
        if tta:  # horizontal flip, vertical flip, 90-degree rotation
            views += [x[:, :, ::-1], x[:, ::-1, :], np.rot90(x, 1, axes=(1, 2))]
        logits, feats = self._run(np.ascontiguousarray(np.stack(views)))
        probs = _softmax(logits).mean(axis=0)
        th = self.thresholds
        unfiltered_best = int(probs.argmax())
        crop_mass = None
        if crop:  # user told us the crop: only its diseases (and "no leaf") are possible
            in_crop = np.array([c.split("___")[0] == crop for c in self.classes])
            crop_mass = float(probs[in_crop].sum())
            allowed = in_crop | np.array([c == BACKGROUND for c in self.classes])
            if allowed.sum() > 1:
                probs = np.where(allowed, probs, 0.0)
                probs = probs / probs.sum()

        order = np.argsort(probs)[::-1]
        if crop:
            order = order[probs[order] > 0]   # hide classes ruled out by the crop filter
        best = int(order[0])
        confidence = float(probs[best])
        entropy = float(-(probs * np.log(probs + 1e-12)).sum() / np.log(len(probs)))
        key = self.classes[best]

        # Decision, most fundamental problem first. Only "healthy"/"diseased" are diagnoses.
        q = quality or self.quality(img)
        problem = self.photo_problem(q)
        reason = None
        if problem:
            status, reason = "unusable_photo", problem
        elif (self.classes[unfiltered_best] == BACKGROUND
              or self.plant_fraction(img) < th["min_plant_fraction"]):
            status, reason = "no_leaf", "No plant leaf was recognised in the photo."
        elif crop and crop_mass is not None and (crop_mass < th["crop_mismatch_below"] or key == BACKGROUND):
            seen = self.classes[unfiltered_best].split("___")[0]
            status = "crop_mismatch"
            reason = (f"This photo does not look like {crop}: the model sees {seen} instead. "
                      f"Check the selected crop, or choose auto-detect.")
        elif confidence < (th["uncertain_below_with_crop"] if crop else th["uncertain_below"]):
            status, reason = "uncertain", "The model is not confident enough to give a diagnosis."
        elif key.endswith("healthy"):
            status = "healthy"
        else:
            status = "diseased"

        result = {
            "class": key,
            "confidence": confidence,
            "status": status,
            "entropy": round(entropy, 4),
            "tta": tta,
            "crop_filter": crop,
            "decision_reason": reason,
            "crop_match": None if crop_mass is None else round(crop_mass, 4),
            "model_sees": self.classes[unfiltered_best] if crop else None,
            "top_k": [{"class": self.classes[i], "probability": float(probs[i])}
                      for i in order[:top_k]],
        }
        if with_heatmap and key != BACKGROUND:
            result["heatmap"] = self.heatmap(feats[0], best, img.size)
        return result

    def heatmap(self, fmap: np.ndarray, cls: int, size: tuple[int, int]) -> str:
        """Class activation map for `cls` as a transparent PNG data URL sized like the input."""
        # Per-location class score. The logit is the mean of these, so the informative part is
        # where a location scores above the average location (for ViTs every patch token also
        # carries global context, so raw scores are high everywhere and the map looks flat).
        cam = np.tensordot(self.cam_weight[cls], fmap, axes=(0, 0))  # (H', W') grid
        cam = np.maximum(cam - cam.mean(), 0)
        cam = cam / cam.max() if cam.max() > 0 else cam
        w, h = size
        scale = 384 / max(w, h)
        out_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        up = Image.fromarray((cam * 255).astype(np.uint8)).resize(out_size, Image.BICUBIC)
        v = np.asarray(up, dtype=np.float32) / 255.0
        rgba = np.dstack([_colormap(v), (np.clip(v * 1.4, 0, 1) * 230).astype(np.uint8)])
        buf = io.BytesIO()
        Image.fromarray(rgba).save(buf, format="PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def thumbnail(img: Image.Image, size: int = 160) -> str:
        t = img.copy()
        t.thumbnail((size, size))
        buf = io.BytesIO()
        t.save(buf, format="JPEG", quality=75)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
