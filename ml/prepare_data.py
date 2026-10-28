"""Extract PlantVillage and write a stratified train/val/test manifest.

Dataset: PlantVillage (Mendeley Data, doi:10.17632/tywbtsjrjv.1), the
"without augmentation" release: 55k leaf images, 38 crop-disease classes
plus a background class.

    python ml/prepare_data.py
"""
from __future__ import annotations

import csv
import json
import random
import zipfile
from collections import defaultdict

from common import ARTIFACTS, DATA_DIR, EXTRACT_DIR, MANIFEST, RAW_ZIP

IMAGE_EXT = {".jpg", ".jpeg", ".png"}
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42


def extract() -> None:
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.iterdir()):
        print(f"already extracted -> {EXTRACT_DIR}")
        return
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RAW_ZIP) as zf:
        members = [m for m in zf.infolist()
                   if not m.is_dir() and "__MACOSX" not in m.filename
                   and m.filename.lower().rsplit(".", 1)[-1] in {"jpg", "jpeg", "png"}]
        for i, m in enumerate(members):
            parts = m.filename.split("/")
            cls, name = parts[-2], parts[-1]
            dest = EXTRACT_DIR / cls / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(m))
            if i % 5000 == 0:
                print(f"  extracted {i}/{len(members)}")
    print(f"extracted {len(members)} images")


def build_manifest() -> None:
    by_class: dict[str, list[str]] = defaultdict(list)
    for cls_dir in sorted(p for p in EXTRACT_DIR.iterdir() if p.is_dir()):
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in IMAGE_EXT:
                by_class[cls_dir.name].append(f.relative_to(DATA_DIR).as_posix())

    classes = sorted(by_class)
    rng = random.Random(SEED)
    rows = []
    for label, cls in enumerate(classes):
        files = by_class[cls][:]
        rng.shuffle(files)
        n = len(files)
        n_val, n_test = round(n * SPLITS["val"]), round(n * SPLITS["test"])
        for i, f in enumerate(files):
            split = "test" if i < n_test else "val" if i < n_test + n_val else "train"
            rows.append({"path": f, "label": label, "class": cls, "split": split})

    with MANIFEST.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "label", "class", "split"])
        w.writeheader()
        w.writerows(rows)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    counts = {c: {s: 0 for s in SPLITS} for c in classes}
    for r in rows:
        counts[r["class"]][r["split"]] += 1
    (ARTIFACTS / "classes.json").write_text(json.dumps(classes, indent=2))
    (ARTIFACTS / "dataset_stats.json").write_text(json.dumps(counts, indent=2))
    totals = {s: sum(c[s] for c in counts.values()) for s in SPLITS}
    print(f"{len(classes)} classes, {len(rows)} images -> {totals}")


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    extract()
    build_manifest()
