"""Evaluate the exported ONNX model end-to-end on the held-out test split.

Uses exactly the same preprocessing code path as the API, so the reported
numbers are what the deployed service actually achieves.

    python ml/evaluate.py
"""
from __future__ import annotations

import csv
import json
import sys
import time

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, top_k_accuracy_score)

from common import API_MODEL_DIR, DATA_DIR, MANIFEST, ROOT

sys.path.insert(0, str(ROOT / "api"))
from app.inference import Predictor  # noqa: E402


def ece(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    out, edges = 0.0, np.linspace(0, 1, bins + 1)
    reliability = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc, c = float((pred[m] == y[m]).mean()), float(conf[m].mean())
            out += m.mean() * abs(acc - c)
            reliability.append({"bin": [round(lo, 3), round(hi, 3)], "count": int(m.sum()),
                                "accuracy": acc, "confidence": c})
    return float(out), reliability


def main():
    with MANIFEST.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == "test"]
    pred = Predictor(API_MODEL_DIR)
    classes = pred.classes

    probs = np.zeros((len(rows), len(classes)), dtype=np.float32)
    y = np.array([int(r["label"]) for r in rows])
    batch, t0 = 32, time.time()
    for i in range(0, len(rows), batch):
        imgs = [pred.load_image((DATA_DIR / r["path"]).read_bytes()) for r in rows[i:i + batch]]
        probs[i:i + batch] = pred.predict_proba(np.stack([pred.preprocess(im) for im in imgs]))
        if (i // batch) % 25 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    secs = time.time() - t0

    # Single-image latency (what one API request pays for the model).
    x1 = pred.preprocess(pred.load_image((DATA_DIR / rows[0]["path"]).read_bytes()))[None]
    for _ in range(3):
        pred.predict_proba(x1)
    lat = []
    for _ in range(30):
        t = time.perf_counter(); pred.predict_proba(x1); lat.append((time.perf_counter() - t) * 1000)

    y_hat = probs.argmax(1)
    p, r, f, s = precision_recall_fscore_support(y, y_hat, labels=range(len(classes)), zero_division=0)
    e, reliability = ece(probs, y)
    metrics = {
        "split": "test",
        "n_images": len(rows),
        "accuracy": accuracy_score(y, y_hat),
        "top3_accuracy": top_k_accuracy_score(y, probs, k=3, labels=range(len(classes))),
        "top5_accuracy": top_k_accuracy_score(y, probs, k=5, labels=range(len(classes))),
        "macro_f1": f1_score(y, y_hat, average="macro"),
        "weighted_f1": f1_score(y, y_hat, average="weighted"),
        "ece": e,
        "reliability": reliability,
        "latency_ms_p50": float(np.percentile(lat, 50)),
        "latency_ms_p95": float(np.percentile(lat, 95)),
        "throughput_img_s": len(rows) / secs,
        "model_size_mb": (API_MODEL_DIR / "leafnet.onnx").stat().st_size / 1e6,
        "classes": classes,
        "per_class": [{"class": c, "precision": float(p[i]), "recall": float(r[i]),
                       "f1": float(f[i]), "support": int(s[i])} for i, c in enumerate(classes)],
        "confusion_matrix": confusion_matrix(y, y_hat, labels=range(len(classes))).tolist(),
    }
    (API_MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"test accuracy {metrics['accuracy']:.4f} | macro-F1 {metrics['macro_f1']:.4f} | "
          f"top-5 {metrics['top5_accuracy']:.4f} | ECE {e:.4f} | "
          f"latency p50 {metrics['latency_ms_p50']:.1f} ms")
    worst = sorted(metrics["per_class"], key=lambda d: d["f1"])[:5]
    for d in worst:
        print(f"  lowest F1  {d['f1']:.3f}  {d['class']}")


if __name__ == "__main__":
    main()
