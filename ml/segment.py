"""Rough leaf/background segmentation for PlantVillage-style photos.

The background (grey paper or black cloth) is estimated from the image border.
- Light backgrounds: a pixel is leaf if its *chromaticity* (colour with brightness
  divided out, so shadows don't count) differs from the background's.
- Dark backgrounds: a pixel is leaf if it is clearly brighter than the cloth.
Then: morphology, keep the largest component, fill holes (pale lesions/mildew
inside the leaf stay in the mask).
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

try:
    import cv2
except ImportError:  # GrabCut refinement is optional
    cv2 = None


def _otsu(x: np.ndarray, lo: float, hi: float) -> float:
    hist, edges = np.histogram(x, bins=256, range=(lo, hi))
    p = hist / max(hist.sum(), 1)
    w = np.cumsum(p)
    mu = np.cumsum(p * edges[:-1])
    between = (mu[-1] * w - mu) ** 2 / (w * (1 - w) + 1e-12)
    return float(edges[np.argmax(between)])


def border_pixels(a: np.ndarray, width: int = 8) -> np.ndarray:
    return np.concatenate([a[:width].reshape(-1, 3), a[-width:].reshape(-1, 3),
                           a[:, :width].reshape(-1, 3), a[:, -width:].reshape(-1, 3)])


def rough_mask(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    inten = a.sum(-1) / 3
    bg = border_pixels(a)
    if bg.mean() < 60:  # black cloth
        m = inten > max(60.0, _otsu(inten, 0, 255))
    else:
        chroma = a / (a.sum(-1, keepdims=True) + 1.0)
        bg_c = np.median(border_pixels(chroma[..., :, :]), axis=0)
        d = np.linalg.norm(chroma - bg_c, axis=-1) * 100
        m = (d > max(4.0, _otsu(d, 0, 40))) & (inten > 25)
    m = ndi.binary_opening(m, iterations=2)
    lab, n = ndi.label(m)
    if n == 0:
        return np.zeros(m.shape, dtype=bool)
    sizes = ndi.sum(m, lab, range(1, n + 1))
    m = lab == (int(np.argmax(sizes)) + 1)
    m = ndi.binary_closing(m, iterations=4)
    return ndi.binary_fill_holes(m)


def leaf_mask(img: Image.Image, iters: int = 4) -> np.ndarray:
    """Rough colour mask refined with GrabCut (colour models + graph cut on edges)."""
    rough = rough_mask(img)
    if cv2 is None:
        return rough
    a = np.asarray(img.convert("RGB"))
    h, w = rough.shape
    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    yy, xx = np.ogrid[:h, :w]
    centre = ((yy - h / 2) / (0.32 * h)) ** 2 + ((xx - w / 2) / (0.32 * w)) ** 2 <= 1
    gc[centre | rough] = cv2.GC_PR_FGD
    edge = np.zeros_like(rough)
    edge[:4], edge[-4:], edge[:, :4], edge[:, -4:] = True, True, True, True
    gc[edge & ~ndi.binary_dilation(rough, iterations=3)] = cv2.GC_BGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(a[..., ::-1].copy(), gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return rough
    m = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    lab, n = ndi.label(m)
    if n == 0:
        return rough
    sizes = ndi.sum(m, lab, range(1, n + 1))
    m = lab == (int(np.argmax(sizes)) + 1)
    return ndi.binary_fill_holes(m)
