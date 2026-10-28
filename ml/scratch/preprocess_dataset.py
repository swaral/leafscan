"""Apply the leaf preprocessing pipeline (BiRefNet@512 mask -> safe gate -> soft background ->
colour normalisation) to every image of the combined dataset, on all available GPUs.

    python preprocess_dataset.py --data /tmp/leafscan_data --out /tmp/leafscan_pp

Output mirrors the input layout (same relative image paths, manifest with an extra
`pp_gate` column), plus pp_stats.json (gate decisions per source/split), pp_config.json
(settings the API must reproduce) and pp_samples.jpg (originals vs processed).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

SEG_SIZE = 512
BATCH = 16


def _post(job):
    """CPU part (runs in a worker process): gate + background + colour, then save."""
    from preprocess import leaf_preprocess
    src, dst, mask16 = job
    img = Image.open(src).convert("RGB")
    out, info = leaf_preprocess(img, mask16.astype(np.float32))
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    out.save(dst, quality=92)
    return info["gate"]


def worker(args):
    import torch
    from segmenter import BiRefNet
    rows = list(csv.DictReader(open(Path(args.data) / "manifest.csv", encoding="utf-8")))[args.worker::args.of]
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seg = BiRefNet(dev, size=SEG_SIZE)
    results, t0 = {}, time.time()
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) // args.of)) as pool:
        pending = []
        for i in range(0, len(rows), BATCH):
            chunk = rows[i:i + BATCH]
            imgs = [Image.open(Path(args.data) / r["path"]).convert("RGB") for r in chunk]
            masks = seg.masks(imgs)
            for r, m in zip(chunk, masks):
                pending.append((r["path"], pool.submit(_post, (str(Path(args.data) / r["path"]),
                                                               str(Path(args.out) / r["path"]),
                                                               m.astype(np.float16)))))
            if len(pending) > 256:
                for p, f in pending[:128]:
                    results[p] = f.result()
                pending = pending[128:]
            if (i // BATCH) % 100 == 0:
                done = i + len(chunk)
                print(f"[worker {args.worker}] {done}/{len(rows)}  {done / (time.time() - t0):.1f} img/s", flush=True)
        for p, f in pending:
            results[p] = f.result()
    Path(args.out, f"gates_{args.worker}.json").write_text(json.dumps(results))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", type=int, default=-1)
    ap.add_argument("--of", type=int, default=1)
    args = ap.parse_args()
    if args.worker >= 0:
        return worker(args)

    import torch
    n = max(1, torch.cuda.device_count())
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    procs = []
    for w in range(n):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(w)}
        procs.append(subprocess.Popen([sys.executable, "-u", __file__, "--data", str(data), "--out", str(out),
                                       "--worker", str(w), "--of", str(n)], env=env))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"worker failed: {codes}")

    gates = {}
    for w in range(n):
        gates.update(json.loads((out / f"gates_{w}.json").read_text()))
    rows = list(csv.DictReader(open(data / "manifest.csv", encoding="utf-8")))
    missing = [r["path"] for r in rows if r["path"] not in gates]
    if missing:
        raise SystemExit(f"{len(missing)} images were not processed")
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) + ["pp_gate"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "pp_gate": gates[r["path"]]})
    for f in ("classes.json", "stats.json"):
        shutil.copy(data / f, out / f)

    by = defaultdict(Counter)
    for r in rows:
        by[f"{r['source']}/{r['split']}"][gates[r["path"]]] += 1
    seg_share = {k: round(v["ok"] / sum(v.values()), 3) for k, v in sorted(by.items())}
    stats = {"images": len(rows), "minutes": round((time.time() - t0) / 60, 1),
             "segmented_share_overall": round(sum(g == "ok" for g in gates.values()) / len(rows), 3),
             "segmented_share": seg_share, "gates": {k: dict(v) for k, v in sorted(by.items())}}
    (out / "pp_stats.json").write_text(json.dumps(stats, indent=2))
    (out / "pp_config.json").write_text(json.dumps({
        "segmenter": "ZhengPeng7/BiRefNet_lite", "seg_input": SEG_SIZE, "fp16": True,
        "dataset_short_side": 256, "pipeline": "refine_mask gate -> normalise_colour -> soft blurred background",
        "crop": False}, indent=2))

    # visual check: 8 random test images per source, original | processed
    random.seed(3)
    T, pick = 160, []
    for src in sorted({r["source"] for r in rows}):
        pick += random.sample([r for r in rows if r["source"] == src and r["split"] == "test"], 8)
    sheet = Image.new("RGB", (T * 4, T * (len(pick) + 1) // 2), "white")
    for i, r in enumerate(pick):
        x, y = (i % 2) * 2 * T, (i // 2) * T
        sheet.paste(Image.open(data / r["path"]).convert("RGB").resize((T, T)), (x, y))
        sheet.paste(Image.open(out / r["path"]).convert("RGB").resize((T, T)), (x + T, y))
    sheet.save(out / "pp_samples.jpg", quality=88)
    print("PREPROCESS DONE", json.dumps({k: stats[k] for k in ("images", "minutes", "segmented_share_overall")}), flush=True)


if __name__ == "__main__":
    main()
