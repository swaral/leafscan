"""Does the model look at the leaf, or at dataset "signatures" (background, lighting, session)?

Experiments on the held-out test set (background class excluded):
  A  original images                                  (reference)
  B  leaf only: background replaced by neutral grey   (should stay high if the leaf drives predictions)
  C  background only: leaf erased (inpainted)         (should fall to chance if no shortcut)
  D  background swap: leaf of image i on the background of image j from another crop
     -> does the prediction follow the leaf or the background?
  E  no deep learning: gradient boosting on the 4 image *corners* only (8x8 px each),
     trained on train, tested on test -> how much label signal sits in the background itself
  F  heatmap focus: share of CAM mass that falls on the leaf vs the leaf's share of the image

    python ml/check_shortcuts.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from sklearn.ensemble import HistGradientBoostingClassifier

from common import API_MODEL_DIR, ARTIFACTS, CACHE_DIR, DATA_DIR, MANIFEST, ROOT
from segment import leaf_mask

BG_CLASS = "Background_without_leaves"
RNG = np.random.default_rng(0)


def crop_of(cls: str) -> str:
    return cls.split("___")[0]


def _mask_job(path: str) -> np.ndarray:
    return np.packbits(leaf_mask(Image.open(DATA_DIR / path).convert("RGB")))


def _corner_job(path: str) -> np.ndarray:
    a = np.asarray(Image.open(DATA_DIR / path).convert("RGB").resize((256, 256)), dtype=np.float32)
    feats = []
    for c in (a[:8, :8], a[:8, -8:], a[-8:, :8], a[-8:, -8:]):
        px = c.reshape(-1, 3)
        feats += list(px.mean(0)) + list(px.std(0))
    return np.array(feats, dtype=np.float32)


def load_masks(rows) -> np.ndarray:
    f = CACHE_DIR / "test_leaf_masks.npy"
    if f.exists():
        packed = np.load(f)
    else:
        t = time.time()
        with Pool(4) as pool:
            packed = np.stack(pool.map(_mask_job, [r["path"] for r in rows], chunksize=32))
        np.save(f, packed)
        print(f"  masks: {len(rows)} in {(time.time() - t) / 60:.1f} min", flush=True)
    return np.unpackbits(packed, axis=1)[:, :256 * 256].reshape(-1, 256, 256).astype(bool)


def corner_features(rows, name: str) -> np.ndarray:
    f = CACHE_DIR / f"corners_{name}.npy"
    if f.exists():
        return np.load(f)
    with Pool(4) as pool:
        x = np.stack(pool.map(_corner_job, [r["path"] for r in rows], chunksize=64))
    np.save(f, x)
    return x


def erase_leaf(a: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Remove the leaf by inpainting from the surrounding background."""
    big = ndi.binary_dilation(m, iterations=6).astype(np.uint8) * 255
    return cv2.inpaint(a, big, 9, cv2.INPAINT_TELEA)


def main():
    sys.path.insert(0, str(ROOT / "api"))
    from app.inference import Predictor
    pred = Predictor(API_MODEL_DIR)
    classes = pred.classes

    with MANIFEST.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    test = [r for r in rows if r["split"] == "test" and r["class"] != BG_CLASS]
    train = [r for r in rows if r["split"] == "train" and r["class"] != BG_CLASS]
    y = np.array([int(r["label"]) for r in test])
    crops = np.array([crop_of(r["class"]) for r in test])
    print(f"test images (no background class): {len(test)}", flush=True)

    masks = load_masks(test)
    cover = masks.mean((1, 2))
    edge = np.concatenate([masks[:, :3].reshape(len(masks), -1), masks[:, -3:].reshape(len(masks), -1),
                           masks[:, :, :3].reshape(len(masks), -1), masks[:, :, -3:].reshape(len(masks), -1)], 1)
    good = (cover > 0.08) & (cover < 0.75) & (edge.mean(1) < 0.35)
    print(f"usable masks: {good.sum()} / {len(test)} (rest: leaf fills frame or segmentation failed)", flush=True)
    idx = np.where(good)[0]

    def predict(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        out_p, out_f = [], []
        for s in range(0, len(arrays), 32):
            x = np.stack([pred.preprocess(Image.fromarray(a)) for a in arrays[s:s + 32]])
            logits, feats = pred._run(x)
            out_p.append(logits.argmax(1)); out_f.append(feats)
        return np.concatenate(out_p), np.concatenate(out_f)

    imgs = {i: np.asarray(Image.open(DATA_DIR / test[i]["path"]).convert("RGB").resize((256, 256))) for i in idx}
    report: dict = {"n_test_no_background": len(test), "n_usable_masks": int(good.sum())}
    ys, cs = y[idx], crops[idx]
    chance = {"uniform_38": 1 / 38, "majority_class": float(np.bincount(ys).max() / len(ys))}
    report["chance"] = chance

    def summarise(name, p):
        pc = np.array([crop_of(classes[k]) for k in p])
        r = {"accuracy": float((p == ys).mean()), "crop_accuracy": float((pc == cs).mean()),
             "predicted_background_class": float((np.array(classes)[p] == BG_CLASS).mean())}
        report[name] = r
        print(f"{name:28s} acc {r['accuracy']:.4f}  crop-acc {r['crop_accuracy']:.4f}  "
              f"'no leaf' {r['predicted_background_class']:.3f}", flush=True)
        return r

    # A. original
    p_orig, f_orig = predict([imgs[i] for i in idx])
    summarise("A_original", p_orig)

    # B. leaf only on neutral grey
    grey = np.full((256, 256, 3), 128, np.uint8)
    leaf_only = [np.where(masks[i][..., None], imgs[i], grey) for i in idx]
    summarise("B_leaf_only_grey_bg", predict(leaf_only)[0])

    # C. background only (leaf inpainted away)
    bg_only = [erase_leaf(imgs[i], masks[i]) for i in idx]
    summarise("C_background_only", predict(bg_only)[0])

    # D. background swap with an image of a different crop
    partner = np.empty(len(idx), dtype=int)
    for k, i in enumerate(idx):
        cands = idx[crops[idx] != crops[i]]
        partner[k] = RNG.choice(cands)
    pos = {i: k for k, i in enumerate(idx)}
    swapped = [np.where(masks[i][..., None], imgs[i], bg_only[pos[j]]) for i, j in zip(idx, partner)]
    p_swap = predict(swapped)[0]
    y_bg = y[partner]
    report["D_background_swap"] = {
        "follows_leaf": float((p_swap == ys).mean()),
        "follows_background": float((p_swap == y_bg).mean()),
        "leaf_crop_kept": float((np.array([crop_of(classes[k]) for k in p_swap]) == cs).mean()),
        "background_crop": float((np.array([crop_of(classes[k]) for k in p_swap]) == crops[partner]).mean()),
    }
    print("D_background_swap           ", report["D_background_swap"], flush=True)

    # E. corners-only classifier, no deep learning
    t = time.time()
    x_tr = corner_features(train, "train")
    x_te = corner_features(test, "test")
    y_tr = np.array([int(r["label"]) for r in train])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0)
    clf.fit(x_tr, y_tr)
    acc_e = float((clf.predict(x_te) == y).mean())
    report["E_corners_only_gbm"] = {
        "accuracy": acc_e, "features": "mean+std RGB of the four 8x8 corner patches (24 numbers)",
        "majority_baseline": float(np.bincount(y).max() / len(y)),
        "crop_accuracy": float((np.array([crop_of(classes[k]) for k in clf.predict(x_te)]) == crops).mean()),
    }
    print(f"E_corners_only_gbm           acc {acc_e:.4f}  ({time.time() - t:.0f}s)", flush=True)

    # F. CAM focus
    on_leaf = []
    for k, i in enumerate(idx):
        cam = np.maximum(np.tensordot(pred.cam_weight[p_orig[k]], f_orig[k], axes=(0, 0)), 0)
        cam = np.asarray(Image.fromarray(cam.astype(np.float32)).resize((256, 256), Image.BILINEAR))
        tot = cam.sum()
        if tot > 0:
            on_leaf.append(cam[masks[i]].sum() / tot)
    report["F_cam_mass_on_leaf"] = {"mean": float(np.mean(on_leaf)),
                                    "median": float(np.median(on_leaf)),
                                    "leaf_area_share_mean": float(cover[idx].mean())}
    print("F_cam_mass_on_leaf           ", report["F_cam_mass_on_leaf"], flush=True)

    # visual examples of B, C, D
    sheet = Image.new("RGB", (4 * 128, 6 * 128), "white")
    for r_, k in enumerate(RNG.choice(len(idx), 6, replace=False)):
        for c_, a in enumerate((imgs[idx[k]], leaf_only[k], bg_only[k], swapped[k])):
            sheet.paste(Image.fromarray(a).resize((128, 128)), (c_ * 128, r_ * 128))
    sheet.save(ARTIFACTS / "shortcut_examples.jpg", quality=85)
    (ARTIFACTS / "shortcut_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
