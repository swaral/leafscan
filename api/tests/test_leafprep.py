"""The API must preprocess leaves exactly like training did."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

API = Path(__file__).resolve().parents[1]
ML = API.parent / "ml" / "scratch"


@pytest.mark.parametrize("api_file,ml_file", [("leafprep.py", "preprocess.py"), ("birefnet.py", "segmenter.py")])
def test_copies_match_training_code(api_file, ml_file):
    if not (ML / ml_file).exists():
        pytest.skip("training code not present (e.g. inside the Docker image)")
    api_src = (API / "app" / api_file).read_text(encoding="utf-8").split("\n", 1)[1]  # drop header line
    assert api_src == (ML / ml_file).read_text(encoding="utf-8"), f"{api_file} drifted from {ml_file}"


def test_resize_like_dataset():
    from app.preprocessing import resize_like_dataset
    assert resize_like_dataset(Image.new("RGB", (4000, 3000))).size == (341, 256)
    assert resize_like_dataset(Image.new("RGB", (2000, 400))).size == (512, 256)   # long side cropped
    assert resize_like_dataset(Image.new("RGB", (100, 120))).size == (256, 307)    # small images upscaled


def test_gate_keeps_leaf_and_suppresses_background():
    from app.leafprep import leaf_preprocess
    a = np.full((256, 256, 3), 200, np.uint8)
    a[64:192, 64:192] = (60, 140, 50)                       # "leaf"
    mask = np.zeros((256, 256), np.float32)
    mask[64:192, 64:192] = 1
    out, info = leaf_preprocess(Image.fromarray(a), mask)
    o = np.asarray(out).astype(int)
    assert info["gate"] == "ok" and info["segmented"]
    assert o[128, 128, 1] > o[128, 128, 0]                  # leaf stays green
    assert abs(o[5, 5] - 200).sum() > 30                    # background changed


def test_gate_rejects_tiny_mask():
    from app.leafprep import leaf_preprocess
    mask = np.zeros((256, 256), np.float32)
    mask[:5, :5] = 1
    _, info = leaf_preprocess(Image.new("RGB", (256, 256), (90, 140, 60)), mask)
    assert info["gate"] == "too_small" and not info["segmented"]
