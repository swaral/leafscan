"""Calibrate, export to ONNX (API format) and evaluate on the held-out test set.

    python export.py --data /content/leafscan_data --run <run dir> --out <run dir>/api_model

Writes everything the LeafScan API needs into --out:
  leafnet.onnx, cam_head.npz, labels.json, metrics.json, train_history.json, dataset_stats.json
Test metrics are computed with ONNX Runtime and the API's own PIL preprocessing,
overall, per data source and per domain (lab vs field).
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from model import ExportWrapper, build_model
from train import BytesDataset, GpuAugment, collate, evaluate, load_split


BACKBONES = {
    "leafresnet_s": "LeafResNet-S (trained from scratch)",
    "leafresnet_m": "LeafResNet-M (trained from scratch)",
    "mobilenet_v3_pretrained": "MobileNetV3-Large (ImageNet pretrained, fine-tuned)",
    "plantclef_vitb14": "DINOv2 ViT-B/14 + registers, PlantCLEF 2024 pretrained (Pl@ntNet), fine-tuned",
}


def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.05, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def ece(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    out, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(out)


def prf(y, p, n):
    rows = []
    for c in range(n):
        tp = int(np.sum((p == c) & (y == c))); fp = int(np.sum((p == c) & (y != c)))
        fn = int(np.sum((p != c) & (y == c)))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        rows.append((pr, rc, f1, tp + fn))
    return rows


def summary(y, probs, n):
    p = probs.argmax(1)
    top = np.argsort(-probs, 1)
    per = prf(y, p, n)
    present = [r for r in per if r[3] > 0]
    return {
        "n_images": int(len(y)),
        "accuracy": float((p == y).mean()),
        "top3_accuracy": float(np.mean([y[i] in top[i, :3] for i in range(len(y))])),
        "top5_accuracy": float(np.mean([y[i] in top[i, :5] for i in range(len(y))])),
        "macro_f1": float(np.mean([r[2] for r in present])) if present else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    root, run = Path(args.data), Path(args.run)
    out = Path(args.out) if args.out else run / "api_model"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    classes, img = ck["classes"], ck["img"]
    n = len(classes)
    net = build_model(ck["arch"], n, pretrained=False, img=img)
    net.load_state_dict(ck["ema"])
    net = net.to(device).eval()
    splits, _ = load_split(root)

    # ---- temperature scaling on validation logits
    val_dl = DataLoader(BytesDataset(root, splits["val"]), batch_size=args.batch, shuffle=False,
                        num_workers=2, collate_fn=collate, timeout=600)
    _, val_logits, val_y = evaluate(net, val_dl, GpuAugment(img, device, False), device, splits["val"], n,
                                    device.type == "cuda")
    T = fit_temperature(val_logits, val_y)
    ece_before = ece(F.softmax(val_logits, 1).numpy(), val_y.numpy())
    ece_after = ece(F.softmax(val_logits / T, 1).numpy(), val_y.numpy())
    print(f"temperature {T:.3f}  val ECE {ece_before:.4f} -> {ece_after:.4f}", flush=True)

    # ---- ONNX export (API contract)
    wrapper = ExportWrapper(net.cpu().float(), T).eval()
    onnx_path = out / "leafnet.onnx"
    torch.onnx.export(wrapper, torch.rand(1, 3, img, img), str(onnx_path), input_names=["image"],
                      output_names=["logits", "features"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}, "features": {0: "batch"}},
                      opset_version=17, dynamo=False)
    np.savez(out / "cam_head.npz", weight=net.fc.weight.detach().numpy().astype(np.float32),
             bias=net.fc.bias.detach().numpy().astype(np.float32))
    (out / "labels.json").write_text(json.dumps(classes, indent=2))

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = torch.rand(2, 3, img, img)
    with torch.no_grad():
        ref = wrapper(x)[0].numpy()
    diff = float(np.abs(ref - sess.run(["logits"], {"image": x.numpy()})[0]).max())
    print(f"exported {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB), torch vs onnx max diff {diff:.2e}")

    # ---- test evaluation with the API's preprocessing (PIL bilinear resize to img x img, [0,1]).
    # Runs the calibrated PyTorch graph on the GPU; ONNX parity is checked on real test images below.
    test = splits["test"]
    probs = np.zeros((len(test), n), dtype=np.float32)
    y = np.array([int(r["label"]) for r in test])
    gpu_wrapper = ExportWrapper(build_model(ck["arch"], n, pretrained=False, img=img), T)
    gpu_wrapper.net.load_state_dict(ck["ema"])
    gpu_wrapper = gpu_wrapper.to(device).eval()

    def load_batch(rows):
        return np.stack([np.asarray(Image.open(root / r["path"]).convert("RGB").resize((img, img), Image.BILINEAR),
                                    dtype=np.float32).transpose(2, 0, 1) / 255.0 for r in rows])

    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(test), 64):
            batch = load_batch(test[s:s + 64])
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                z = gpu_wrapper(torch.from_numpy(batch).to(device))[0].float()
            probs[s:s + 64] = torch.softmax(z, 1).cpu().numpy()
            if s % 2048 == 0:
                print(f"  test {s}/{len(test)}", flush=True)
    secs = time.time() - t0
    sample = load_batch(test[:32])
    onnx_p = sess.run(["logits"], {"image": sample})[0]
    onnx_p = np.exp(onnx_p - onnx_p.max(1, keepdims=True)); onnx_p /= onnx_p.sum(1, keepdims=True)
    parity = float(np.abs(onnx_p - probs[:32]).max())
    print(f"ONNX vs evaluated model on 32 test images: max prob diff {parity:.4f}")
    lat = []
    for _ in range(20):
        t = time.perf_counter(); sess.run(["logits"], {"image": sample[:1]}); lat.append((time.perf_counter() - t) * 1000)

    pred = probs.argmax(1)
    per = prf(y, pred, n)
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (y, pred), 1)
    metrics = {
        "split": "test", **summary(y, probs, n),
        "weighted_f1": float(sum(r[2] * r[3] for r in per) / max(1, sum(r[3] for r in per))),
        "ece": ece(probs, y),
        "latency_ms_p50": float(np.percentile(lat, 50)), "latency_ms_p95": float(np.percentile(lat, 95)),
        "throughput_img_s": len(test) / secs, "model_size_mb": onnx_path.stat().st_size / 1e6,
        "onnx_parity_max_prob_diff": parity,
        "classes": classes,
        "per_class": [{"class": c, "precision": per[i][0], "recall": per[i][1], "f1": per[i][2],
                       "support": per[i][3]} for i, c in enumerate(classes)],
        "confusion_matrix": cm.tolist(),
        "per_source": {}, "per_domain": {},
    }
    for key, field in (("per_source", "source"), ("per_domain", "domain")):
        groups = defaultdict(list)
        for i, r in enumerate(test):
            groups[r[field]].append(i)
        for g, idx in sorted(groups.items()):
            metrics[key][g] = summary(y[idx], probs[idx], n)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    history = json.loads((run / "history.json").read_text()) if (run / "history.json").exists() else []
    params = sum(p.numel() for p in net.parameters())
    (out / "train_history.json").write_text(json.dumps({
        "history": history, "best_val_macro_f1": max((h["val_macro_f1"] for h in history), default=None),
        "temperature": T, "val_ece_before": ece_before, "val_ece_after": ece_after,
        "config": {**ck["args"], "backbone": BACKBONES.get(ck["arch"], ck["arch"]),
                   "params_millions": round(params / 1e6, 2), "label_smoothing": ck["args"]["smoothing"]},
    }, indent=2))
    if (root / "pp_config.json").exists():  # trained on leaf-preprocessed data: the API must apply it too
        shutil.copy(root / "pp_config.json", out / "pp_config.json")
    stats = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for s, rows in splits.items():
        for r in rows:
            stats[r["key"]][s] += 1
    (out / "dataset_stats.json").write_text(json.dumps(stats, indent=2))

    print(f"TEST accuracy {metrics['accuracy']:.4f} | macro-F1 {metrics['macro_f1']:.4f} | "
          f"top-5 {metrics['top5_accuracy']:.4f} | ECE {metrics['ece']:.4f}")
    for g, m in metrics["per_domain"].items():
        print(f"  {g:6s} acc {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}  (n={m['n_images']})")
    for g, m in metrics["per_source"].items():
        print(f"  {g:15s} acc {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}  (n={m['n_images']})")


if __name__ == "__main__":
    main()
