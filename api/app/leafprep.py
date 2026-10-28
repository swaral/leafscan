# Copied verbatim from ml/scratch/preprocess.py (training). Keep identical: tests/test_leafprep.py checks it.
"""Leaf-focused preprocessing shared by training and the API.

    leaf mask (salient-object segmentation) -> neutral background -> colour/lighting
    normalisation  [-> optional crop to the leaf, off by default; to be A/B tested]

Colour normalisation is deliberately conservative because leaf colour is diagnostic
(yellowing, browning): only the *illuminant* and *exposure* are corrected.
  1. White balance: shades-of-grey illuminant estimate (Finlayson & Trezzi, p=6) on
     the whole photo, with per-channel gains clamped to [0.85, 1.18] so a green scene
     is never "corrected" towards grey.
  2. Exposure: scale lightness so the leaf's median L* (CIELAB) lands at a fixed
     target, leaving chroma (a*, b*) untouched.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as ndi

BG_GREY = (128, 128, 128)
TARGET_L = 55.0          # median leaf lightness after normalisation (L* in 0..100)
GAIN_RANGE = (0.85, 1.18)
MIN_COVER, MAX_COVER = 0.03, 0.97   # outside this, segmentation is not trusted


# ---------------------------------------------------------------- colour
def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.clip(c, 0, None), 1 / 2.4) - 0.055)


def white_balance(a: np.ndarray, p: float = 6.0) -> np.ndarray:
    """a: float RGB in [0,1]. Shades-of-grey white balance with clamped gains."""
    lin = _srgb_to_linear(a)
    illum = np.power(np.mean(np.power(lin.reshape(-1, 3) + 1e-6, p), axis=0), 1 / p)
    gains = np.clip(illum.mean() / illum, *GAIN_RANGE)
    return np.clip(_linear_to_srgb(lin * gains), 0, 1)


def normalise_exposure(a: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Scale lightness so the leaf's median L* equals TARGET_L (gamma on luminance only)."""
    lin = _srgb_to_linear(a)
    y = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    L = np.where(y > 0.008856, 116 * np.cbrt(y) - 16, 903.3 * y)
    sel = L[mask] if mask is not None and mask.any() else L.ravel()
    med = float(np.median(sel))
    if med < 1:
        return a
    target_y = ((TARGET_L + 16) / 116) ** 3
    cur_y = ((med + 16) / 116) ** 3
    k = np.clip(target_y / cur_y, 0.4, 2.5)          # avoid extreme corrections
    return np.clip(_linear_to_srgb(lin * k), 0, 1)


def normalise_colour(img: Image.Image, mask: np.ndarray | None = None) -> Image.Image:
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    a = white_balance(a)
    a = normalise_exposure(a, mask)
    return Image.fromarray((a * 255 + 0.5).astype(np.uint8))


# ---------------------------------------------------------------- mask refinement + quality gate
MIN_COVER_GATE = 0.05
LEAFY_RING_MAX = 0.55   # if the band just outside the mask is mostly leaf-coloured, the mask cuts a leaf


def leafy_pixels(img: Image.Image) -> np.ndarray:
    """Plant-coloured pixels: green through yellow/brown, not grey/white/black."""
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32)
    h, s, v = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
    return (h >= 15) & (h <= 170) & (s >= 0.18) & (v >= 0.12)


def refine_mask(soft: np.ndarray, img: Image.Image) -> tuple[np.ndarray | None, str]:
    """Turn a soft salient-object mask into a safe leaf mask, or None (= keep the photo raw).
    - keep the main region(s), fill holes (lesions/mildew inside the leaf stay),
    - extend ~2% of the image diagonal beyond the leaf with a *linear fade* (alpha 1 -> 0), so
      necrotic edges are kept but no sharp strip of original background survives (that strip
      showed up as a light halo around lab leaves, itself a dataset signature),
    - reject when too small, too large, fragmented, or when the boundary cuts through leaf colour.
    Returns a float alpha map in [0,1] (1 = leaf) or None."""
    hard = soft > 0.5
    h, w = hard.shape
    cover = float(hard.mean())
    if cover < MIN_COVER_GATE:
        return None, "too_small"
    if cover > MAX_COVER:
        return None, "fills_frame"
    lab, n = ndi.label(hard)
    sizes = ndi.sum(hard, lab, range(1, n + 1))
    big = [i + 1 for i, sz in enumerate(sizes) if sz >= 0.15 * sizes.max()]
    if len(big) > 3 or sizes.max() < 0.6 * hard.sum():
        return None, "fragmented"
    core = ndi.binary_fill_holes(np.isin(lab, big))
    r = max(2, int(0.02 * (h * w) ** 0.5))
    dist = ndi.distance_transform_edt(~core)
    alpha = np.clip(1.0 - dist / r, 0.0, 1.0).astype(np.float32)
    ring = (dist > r) & (dist <= 2 * r)
    if ring.any() and float(leafy_pixels(img)[ring].mean()) > LEAFY_RING_MAX:
        return None, "cuts_leaf"
    return alpha, "ok"


def mask_is_usable(mask: np.ndarray) -> bool:
    cover = float(mask.mean())
    return MIN_COVER <= cover <= MAX_COVER


# ---------------------------------------------------------------- background


def crop_box(mask: np.ndarray, margin: float = 0.08) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    my, mx = int((y1 - y0) * margin), int((x1 - x0) * margin)
    # square-ish crop so the later resize to 224x224 does not distort the leaf much
    y0, y1, x0, x1 = max(0, y0 - my), min(h, y1 + my), max(0, x0 - mx), min(w, x1 + mx)
    side = max(y1 - y0, x1 - x0)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0, x0 = max(0, cy - side // 2), max(0, cx - side // 2)
    return x0, y0, min(w, x0 + side), min(h, y0 + side)


def soften_background(img: Image.Image, mask: np.ndarray, mode: str = "blur") -> Image.Image:
    """Suppress the background without deleting anything: heavy blur + darken + desaturate
    ("blur"), or flat grey ("grey"). Mask edges are feathered."""
    a = np.asarray(img, dtype=np.float32)
    if mode == "grey":
        bg = np.full_like(a, BG_GREY)
    else:
        rad = max(4, int(0.04 * max(img.size)))
        b = np.asarray(img.filter(ImageFilter.GaussianBlur(rad)), dtype=np.float32)
        g = b.mean(-1, keepdims=True)
        bg = (0.5 * b + 0.5 * g) * 0.55 + 0.45 * 110   # desaturate, darken towards neutral
    alpha = mask.astype(np.float32)[..., None]   # already feathered by refine_mask
    return Image.fromarray((a * alpha + bg * (1 - alpha)).clip(0, 255).astype(np.uint8))


def leaf_preprocess(img: Image.Image, soft_mask: np.ndarray | None, colour: bool = True,
                    background: bool = True, crop: bool = False, bg_mode: str = "blur") -> tuple[Image.Image, dict]:
    """Full pipeline. When the gate rejects the mask, only colour normalisation is applied."""
    img = img.convert("RGB")
    info = {"segmented": False, "gate": "no_mask", "leaf_cover": None}
    alpha = None
    if soft_mask is not None:
        alpha, info["gate"] = refine_mask(soft_mask, img)
    leaf = alpha >= 0.999 if alpha is not None else None
    if colour:
        img = normalise_colour(img, leaf)
    if background and alpha is not None:
        info["leaf_cover"] = round(float(leaf.mean()), 3)
        img = soften_background(img, alpha, bg_mode)
        if crop:
            img = img.crop(crop_box(alpha > 0))
        info["segmented"] = True
    return img, info
