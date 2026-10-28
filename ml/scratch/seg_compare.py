"""GPU comparison of BiRefNet-lite leaf segmentation (1024 vs 512 input) with the safe
mask refinement + quality gate, run on Kaggle.

Outputs (in OUT):
  seg_compare.jpg   rows = 30 test images (6 per dataset, same picks as v1 of this test);
                    columns = original | raw BiRefNet mask (flat grey, previous behaviour)
                              | refined pipeline @1024 | refined pipeline @512
                    (gate decision written on each refined image)
  seg_compare.json  timings (GPU and CPU), gate decisions per dataset on 300 images,
                    leaf-pixel loss raw vs refined, 512-vs-1024 mask agreement
"""
from __future__ import annotations

import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from preprocess import leafy_pixels, leaf_preprocess, refine_mask
from segmenter import BiRefNet

DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SOURCES = ["plantvillage", "plantdoc", "plantwild", "ccmt", "plantpathology"]
T = 160
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "", flush=True)

rows = list(csv.DictReader(open(DATA / "manifest.csv", encoding="utf-8")))
random.seed(5)
sheet_rows = []
for src in SOURCES:  # identical sampling to the first comparison
    sheet_rows += random.sample([r for r in rows if r["source"] == src and r["split"] == "test"], 6)
random.seed(11)
stat_rows = []
for src in SOURCES:
    stat_rows += random.sample([r for r in rows if r["source"] == src and r["split"] == "test"], 60)


def load(rs):
    return [Image.open(DATA / r["path"]).convert("RGB") for r in rs]


def run_masks(model, imgs, bs=8):
    out = []
    for i in range(0, len(imgs), bs):
        out += model.masks(imgs[i:i + bs])
    return out


def leaf_loss(img, mask):
    """Share of plant-coloured pixels that end up outside the kept mask."""
    leafy = leafy_pixels(img)
    return float((leafy & ~mask).sum() / max(1, leafy.sum()))


report = {}
models = {sz: BiRefNet(dev, size=sz) for sz in (1024, 512)}
stat_imgs, sheet_imgs = load(stat_rows), load(sheet_rows)

for sz, m in models.items():
    run_masks(m, stat_imgs[:8])  # warm-up
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    masks = run_masks(m, stat_imgs)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t0) / len(stat_imgs) * 1000
    gates, losses_raw, losses_ref = defaultdict(Counter), [], []
    for r, im, mk in zip(stat_rows, stat_imgs, masks):
        refined, reason = refine_mask(mk, im)
        gates[r["source"]][reason] += 1
        losses_raw.append(leaf_loss(im, mk > 0.5))
        if refined is not None:
            losses_ref.append(leaf_loss(im, refined))
    report[f"birefnet_{sz}"] = {
        "gpu_ms_per_img": round(ms, 1),
        "gate_by_source": {s: dict(gates[s]) for s in SOURCES},
        "kept_raw_share": round(1 - sum(g["ok"] for g in gates.values()) / len(stat_rows), 3),
        "leafy_pixels_lost_raw_mask_mean": round(float(np.mean(losses_raw)), 3),
        "leafy_pixels_lost_refined_mean": round(float(np.mean(losses_ref)), 3) if losses_ref else None,
    }
    report[f"_masks_{sz}"] = masks
    print(f"birefnet@{sz}", json.dumps(report[f"birefnet_{sz}"]), flush=True)

# 512 vs 1024 agreement (IoU of hard masks)
ious = []
for a, b in zip(report.pop("_masks_1024"), report.pop("_masks_512")):
    a, b = a > 0.5, b > 0.5
    u = (a | b).sum()
    if u:
        ious.append((a & b).sum() / u)
report["mask_iou_512_vs_1024"] = round(float(np.mean(ious)), 3)

# CPU latency at 512 (Kaggle CPU, as a stand-in for the app server)
cpu = BiRefNet(torch.device("cpu"), size=512, half=False)
cpu.masks(stat_imgs[:1])
t0 = time.time()
for im in stat_imgs[:3]:
    cpu.masks([im])
report["cpu_ms_per_img_512"] = round((time.time() - t0) / 3 * 1000, 1)

# contact sheet
cols = 4
sheet = Image.new("RGB", (T * cols, T * len(sheet_imgs)), "white")
m1024 = run_masks(models[1024], sheet_imgs)
m512 = run_masks(models[512], sheet_imgs)
for i, (im, a, b) in enumerate(zip(sheet_imgs, m1024, m512)):
    sheet.paste(im.resize((T, T)), (0, i * T))
    raw = Image.composite(im, Image.new("RGB", im.size, (128, 128, 128)), Image.fromarray((a * 255).astype(np.uint8)))
    sheet.paste(raw.resize((T, T)), (T, i * T))
    for j, mk in ((2, a), (3, b)):
        out, info = leaf_preprocess(im, mk)
        tile = out.resize((T, T))
        ImageDraw.Draw(tile).rectangle([0, 0, T, 14], fill=(0, 0, 0))
        ImageDraw.Draw(tile).text((3, 1), info["gate"], fill=(255, 255, 0))
        sheet.paste(tile, (j * T, i * T))

OUT.mkdir(parents=True, exist_ok=True)
sheet.save(OUT / "seg_compare.jpg", quality=88)
(OUT / "seg_compare.json").write_text(json.dumps(report, indent=2))
print("SEGCOMPARE", json.dumps(report), flush=True)
