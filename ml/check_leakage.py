"""Check for train/val/test leakage.

1. Exact duplicates: identical file bytes (MD5) across splits.
2. Near-duplicates: 64-bit perceptual hash (DCT pHash) of each image, compared
   across splits including all 8 flips/rotations of the query image, so
   re-encoded, resized, mirrored or rotated copies are caught.
3. Impact: test accuracy with and without the test images that have a
   near-duplicate in train.

    python ml/check_leakage.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter

import numpy as np
from PIL import Image

from common import API_MODEL_DIR, ARTIFACTS, DATA_DIR, MANIFEST, ROOT

HASH = 32  # image is reduced to 32x32 before the DCT; top-left 8x8 kept
THRESHOLDS = [0, 2, 4, 6, 8, 10]
NEAR_DUP = 4  # Hamming distance treated as "near-duplicate"


def dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    m = np.cos(np.pi * (2 * np.arange(n)[None, :] + 1) * k / (2 * n)) * np.sqrt(2 / n)
    m[0] /= np.sqrt(2)
    return m


D = dct_matrix(HASH)


def phash_bits(gray32: np.ndarray) -> np.uint64:
    coeffs = (D @ gray32 @ D.T)[:8, :8].ravel()
    bits = coeffs[1:] > np.median(coeffs[1:])  # skip DC term
    bits = np.concatenate([[False], bits])
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


def variants(g: np.ndarray):
    """The 8 dihedral transforms (identity, rotations, mirrored rotations)."""
    for k in range(4):
        r = np.rot90(g, k)
        yield r
        yield r[:, ::-1]


def hamming_min(queries: np.ndarray, refs: np.ndarray, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """For each query hash (N, V) return min Hamming distance to refs (M,) and argmin."""
    best_d = np.full(len(queries), 65, dtype=np.int64)
    best_i = np.zeros(len(queries), dtype=np.int64)
    for s in range(0, len(queries), chunk):
        q = queries[s:s + chunk]                       # (c, V)
        x = np.bitwise_count(q[:, :, None] ^ refs[None, None, :]).min(axis=1)  # (c, M)
        best_d[s:s + chunk] = x.min(axis=1)
        best_i[s:s + chunk] = x.argmin(axis=1)
    return best_d, best_i


def main():
    with MANIFEST.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"hashing {len(rows)} images ...", flush=True)

    md5, ph = [], np.zeros((len(rows), 8), dtype=np.uint64)
    for i, r in enumerate(rows):
        data = (DATA_DIR / r["path"]).read_bytes()
        md5.append(hashlib.md5(data).hexdigest())
        img = Image.open(DATA_DIR / r["path"]).convert("L").resize((HASH, HASH), Image.LANCZOS)
        g = np.asarray(img, dtype=np.float64)
        ph[i] = [phash_bits(v) for v in variants(g)]
        if i % 10000 == 0:
            print(f"  {i}", flush=True)

    split = np.array([r["split"] for r in rows])
    label = np.array([int(r["label"]) for r in rows])
    idx = {s: np.where(split == s)[0] for s in ("train", "val", "test")}
    report: dict = {"n": {s: int(len(v)) for s, v in idx.items()}}

    # ---- 1. path overlap + exact duplicates ----
    paths = {s: {rows[i]["path"] for i in v} for s, v in idx.items()}
    report["path_overlap"] = {f"{a}-{b}": len(paths[a] & paths[b])
                              for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    by_md5: dict[str, set] = {}
    for i, h in enumerate(md5):
        by_md5.setdefault(h, set()).add(split[i])
    within = sum(c - 1 for c in Counter(md5).values() if c > 1)
    report["exact_duplicate_files_total"] = within
    report["exact_duplicates_across_splits"] = {
        pair: sum(1 for s in by_md5.values() if set(pair.split("-")) <= s)
        for pair in ("train-val", "train-test", "val-test")}

    # ---- 2. near-duplicates (pHash, all 8 orientations) ----
    ref_train = ph[idx["train"], 0]
    near = {}
    nearest = {}
    for name, q_idx, ref_idx in [("val->train", idx["val"], idx["train"]),
                                 ("test->train", idx["test"], idx["train"]),
                                 ("test->val", idx["test"], idx["val"])]:
        d, j = hamming_min(ph[q_idx], ph[ref_idx, 0])
        nearest[name] = (q_idx, ref_idx[j], d)
        near[name] = {f"<={t}": int((d <= t).sum()) for t in THRESHOLDS}
        same = label[q_idx] == label[ref_idx[j]]
        near[name]["same_label_at_<=4"] = int(((d <= NEAR_DUP) & same).sum())
        print(f"{name:12s} " + "  ".join(f"d{k}: {v}" for k, v in near[name].items()), flush=True)
    report["near_duplicates_phash"] = near

    # Baseline: how close are random *different-class* pairs? (sanity check for the threshold)
    rng = np.random.default_rng(0)
    a = rng.choice(len(rows), 20000)
    b = rng.choice(len(rows), 20000)
    m = label[a] != label[b]
    rand_d = np.bitwise_count(ph[a, 0] ^ ph[b, 0])[m]
    report["random_cross_class_pairs_<=4"] = float((rand_d <= NEAR_DUP).mean())

    # ---- 3. contact sheet of the closest test->train pairs for eyeballing ----
    q_idx, r_idx, d = nearest["test->train"]
    order = np.argsort(d)[:12]
    sheet = Image.new("RGB", (2 * 128 + 10, 12 * 128), "white")
    for k, o in enumerate(order):
        for c, ii in enumerate((q_idx[o], r_idx[o])):
            im = Image.open(DATA_DIR / rows[ii]["path"]).convert("RGB").resize((128, 128))
            sheet.paste(im, (c * 138, k * 128))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    sheet.save(ARTIFACTS / "leakage_closest_pairs.jpg", quality=85)
    report["closest_pairs"] = [{"test": rows[q_idx[o]]["path"], "train": rows[r_idx[o]]["path"],
                                "hamming": int(d[o])} for o in order]

    # ---- 4. impact on test accuracy ----
    sys.path.insert(0, str(ROOT / "api"))
    from app.inference import Predictor
    pred = Predictor(API_MODEL_DIR)
    test = idx["test"]
    correct = np.zeros(len(test), dtype=bool)
    for s in range(0, len(test), 32):
        batch = test[s:s + 32]
        x = np.stack([pred.preprocess(pred.load_image((DATA_DIR / rows[i]["path"]).read_bytes())) for i in batch])
        correct[s:s + 32] = pred.predict_proba(x).argmax(1) == label[batch]
    flagged = nearest["test->train"][2] <= NEAR_DUP
    report["test_accuracy"] = {
        "all": float(correct.mean()), "n_all": int(len(test)),
        "without_near_dups": float(correct[~flagged].mean()), "n_without": int((~flagged).sum()),
        "on_near_dups": float(correct[flagged].mean()) if flagged.any() else None,
        "n_near_dups": int(flagged.sum()),
    }
    (ARTIFACTS / "leakage_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "closest_pairs"}, indent=2))


if __name__ == "__main__":
    main()
