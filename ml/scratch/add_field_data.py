"""Add field photos from the external datasets to a built dataset (in place).

    python add_field_data.py --data /tmp/leafscan_data --ext /tmp/ext \
        --train-sets potato_india,maize_tanzania,coffee_ecuador

The external set prepared by external_test.py (--save-images) has a manifest.json. For each
dataset in --train-sets, images are resized like the main dataset, grouped by near-duplicate
pHash and split by group into train / val / ext_test (55/15/30, stratified by class). Train and
val rows are appended to manifest.csv as field photos (source "ext_<dataset>"); ext_test rows are
recorded but not used by train.py, so they stay a clean in-source test set. Datasets not listed
stay completely unseen (leave-dataset-out test). ext_used.json lists the external images that
were used for training or validation, so external_test.py can exclude them.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

from build_dataset import group_near_duplicates, phashes, split_groups
from external_test import resize_like_dataset

FRACS = (0.55, 0.15, 0.30)   # train, val, ext_test


def _prepare(job):
    src, dest = job
    try:
        img = resize_like_dataset(ImageOps.exif_transpose(Image.open(src)).convert("RGB"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest, quality=90)
        return phashes(img)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ext", required=True, help="extracted external set (with manifest.json)")
    ap.add_argument("--train-sets", required=True)
    args = ap.parse_args()
    root, ext = Path(args.data), Path(args.ext)
    train_sets = args.train_sets.split(",")
    classes = json.loads((root / "classes.json").read_text())
    index = {k: i for i, k in enumerate(classes)}

    items = [it for it in json.loads((ext / "manifest.json").read_text())["items"]
             if it["dataset"] in train_sets and it["key"] in index]
    jobs = [(ext / it["path"], root / "images" / f"ext_{it['dataset']}" / f"{n:06d}.jpg")
            for n, it in enumerate(items)]
    with ProcessPoolExecutor(8) as pool:
        hashes = list(pool.map(_prepare, jobs, chunksize=32))
    recs = [{"path": str(dest.relative_to(root)).replace("\\", "/"), "key": it["key"],
             "source": f"ext_{it['dataset']}", "source_label": it["label"], "phash": h, "ext_path": it["path"]}
            for it, (_, dest), h in zip(items, jobs, hashes) if h is not None]

    groups = group_near_duplicates(recs)
    splits = [{"test": "ext_test"}.get(s, s) for s in split_groups(recs, groups, fracs=FRACS)]
    with (root / "manifest.csv").open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for r, g, s in zip(recs, groups, splits):
            w.writerow([r["path"], index[r["key"]], r["key"], r["source"], r["source_label"], s, f"x{g}", "field"])

    used = sorted(r["ext_path"] for r, s in zip(recs, splits) if s in ("train", "val"))
    (root / "ext_used.json").write_text(json.dumps(used))
    summary = {d: dict(Counter(s for r, s in zip(recs, splits) if r["source"] == f"ext_{d}")) for d in train_sets}
    (root / "ext_split.json").write_text(json.dumps(summary, indent=2))
    print("field data added:", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
