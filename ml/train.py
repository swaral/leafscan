"""Train the leaf-disease classifier.

Stage 1  Cache activations of the frozen MobileNetV3 stem (ImageNet weights):
         a clean view for train/val and one augmented view for train.
Stage 2  Fine-tune the last inverted-residual blocks + linear head on the cache
         with class-weighted label-smoothed CE, AdamW, cosine LR, early stopping.
Stage 3  Temperature scaling on the validation set for calibrated confidences.

Caching the frozen stem is what makes real fine-tuning practical on a laptop
CPU (~40 min end to end instead of many hours).

    python ml/train.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

from common import (ARTIFACTS, CACHE_DIR, DATA_DIR, MANIFEST, LeafNet,
                    eval_transform, train_transform)

torch.manual_seed(0)
np.random.seed(0)


class ImageDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(DATA_DIR / r["path"]).convert("RGB")
        return self.transform(img), int(r["label"])


def load_manifest():
    with MANIFEST.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}


@torch.inference_mode()
def cache_split(net: LeafNet, rows, transform, name: str, batch: int, workers: int):
    feat_path, label_path = CACHE_DIR / f"{name}.npy", CACHE_DIR / f"{name}_y.npy"
    if feat_path.exists() and label_path.exists():
        print(f"[cache] {name}: reusing {feat_path.name}")
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(ImageDataset(rows, transform), batch_size=batch,
                        num_workers=workers, persistent_workers=workers > 0)
    tmp = CACHE_DIR / f"{name}.partial.npy"
    out = None
    labels = np.empty(len(rows), dtype=np.int64)
    i, t0 = 0, time.time()
    for x, y in loader:
        z = net.stem(x).half().numpy()
        if out is None:
            out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                            shape=(len(rows), *z.shape[1:]))
        out[i:i + len(z)] = z
        labels[i:i + len(z)] = y.numpy()
        i += len(z)
        if (i // batch) % 40 == 0:
            rate = i / (time.time() - t0)
            print(f"[cache] {name}: {i}/{len(rows)}  {rate:.0f} img/s  "
                  f"eta {(len(rows) - i) / rate / 60:.1f} min", flush=True)
    out.flush()
    del out
    tmp.replace(feat_path)
    np.save(label_path, labels)
    print(f"[cache] {name}: done in {(time.time() - t0) / 60:.1f} min")


def feature_augment(x: torch.Tensor) -> torch.Tensor:
    """Cheap extra augmentation on cached 7x7 activations: flips + 90-degree rotations."""
    if torch.rand(1) < 0.5:
        x = x.flip(3)
    if torch.rand(1) < 0.5:
        x = x.flip(2)
    k = int(torch.randint(0, 4, (1,)))
    return torch.rot90(x, k, dims=(2, 3)) if k else x


@torch.no_grad()
def predict_cached(net: LeafNet, feats: np.ndarray, batch: int = 512) -> torch.Tensor:
    net.eval()
    out = []
    for i in range(0, len(feats), batch):
        logits, _ = net.head(torch.from_numpy(feats[i:i + batch]).float())
        out.append(logits)
    return torch.cat(out)


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    ece, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)


def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.05, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    classes = json.loads((ARTIFACTS / "classes.json").read_text())
    splits = load_manifest()
    net = LeafNet(len(classes))
    net.stem.eval()
    for p in net.stem.parameters():
        p.requires_grad_(False)

    # ---- Stage 1: cache frozen stem activations ----
    cache_split(net, splits["val"], eval_transform(), "val", 64, args.workers)
    cache_split(net, splits["train"], eval_transform(), "train_clean", 64, args.workers)
    cache_split(net, splits["train"], train_transform(), "train_aug", 64, args.workers)

    print("[train] loading cache into memory")
    tr_clean = np.load(CACHE_DIR / "train_clean.npy")
    tr_aug = np.load(CACHE_DIR / "train_aug.npy")
    y_tr = np.load(CACHE_DIR / "train_clean_y.npy")
    va = np.load(CACHE_DIR / "val.npy")
    y_va = np.load(CACHE_DIR / "val_y.npy")

    # ---- Stage 2: fine-tune tail + head ----
    counts = np.bincount(y_tr, minlength=len(classes))
    weights = torch.tensor((counts.max() / counts) ** 0.5, dtype=torch.float32)
    weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    params = [p for p in list(net.tail.parameters()) + list(net.fc.parameters())]
    opt = torch.optim.AdamW([
        {"params": net.tail.parameters(), "lr": args.lr * 0.3},
        {"params": net.fc.parameters(), "lr": args.lr},
    ], weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(y_tr) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr * 0.3, args.lr], total_steps=args.epochs * steps_per_epoch,
        pct_start=0.15)

    history, best_f1, best_state, bad = [], -1.0, None, 0
    y_tr_t = torch.from_numpy(y_tr)
    for epoch in range(1, args.epochs + 1):
        net.tail.train(); net.dropout.train()
        perm = np.random.permutation(len(y_tr))
        use_aug = np.random.rand(len(y_tr)) < 0.5
        t0, total, correct, loss_sum = time.time(), 0, 0, 0.0
        for s in range(steps_per_epoch):
            idx = np.sort(perm[s * args.batch:(s + 1) * args.batch])
            a = use_aug[idx]
            xb = np.where(a[:, None, None, None], tr_aug[idx], tr_clean[idx])
            x = feature_augment(torch.from_numpy(xb).float())
            y = y_tr_t[idx]
            logits, _ = net.head(x)
            loss = criterion(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            opt.step(); sched.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)

        val_logits = predict_cached(net, va)
        val_loss = F.cross_entropy(val_logits, torch.from_numpy(y_va)).item()
        val_pred = val_logits.argmax(1).numpy()
        val_acc = float((val_pred == y_va).mean())
        val_f1 = float(f1_score(y_va, val_pred, average="macro"))
        rec = {"epoch": epoch, "train_loss": loss_sum / total, "train_acc": correct / total,
               "val_loss": val_loss, "val_acc": val_acc, "val_macro_f1": val_f1,
               "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        print(f"[train] epoch {epoch:2d}  loss {rec['train_loss']:.4f}  acc {rec['train_acc']:.4f}  "
              f"| val loss {val_loss:.4f}  acc {val_acc:.4f}  macroF1 {val_f1:.4f}  ({rec['seconds']}s)",
              flush=True)

        if val_f1 > best_f1:
            best_f1, bad = val_f1, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            # Persist immediately so an interrupted run still leaves a usable model.
            ARTIFACTS.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": best_state, "classes": classes, "temperature": 1.0,
                        "epoch": epoch}, ARTIFACTS / "leafnet.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("[train] early stopping")
                break

    net.load_state_dict(best_state)

    # ---- Stage 3: temperature scaling ----
    val_logits = predict_cached(net, va)
    y_va_t = torch.from_numpy(y_va)
    T = fit_temperature(val_logits, y_va_t)
    ece_before = expected_calibration_error(F.softmax(val_logits, 1).numpy(), y_va)
    ece_after = expected_calibration_error(F.softmax(val_logits / T, 1).numpy(), y_va)
    print(f"[calib] temperature {T:.3f}  ECE {ece_before:.4f} -> {ece_after:.4f}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "classes": classes, "temperature": T},
               ARTIFACTS / "leafnet.pt")
    (ARTIFACTS / "train_history.json").write_text(json.dumps({
        "history": history, "best_val_macro_f1": best_f1, "temperature": T,
        "val_ece_before": ece_before, "val_ece_after": ece_after,
        "config": vars(args) | {"backbone": "mobilenet_v3_large (ImageNet)",
                                "trainable_from_block": 14, "label_smoothing": 0.1},
    }, indent=2))
    print(f"[train] saved {ARTIFACTS / 'leafnet.pt'}")


if __name__ == "__main__":
    main()
