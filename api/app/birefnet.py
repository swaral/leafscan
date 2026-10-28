# Copied verbatim from ml/scratch/segmenter.py (training). Keep identical: tests/test_leafprep.py checks it.
"""BiRefNet-lite leaf segmentation in PyTorch (runs on GPU; ONNX Runtime's CUDA build
does not load on Kaggle). Returns soft masks in [0,1] at the original image size."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
REPO = "ZhengPeng7/BiRefNet_lite"


class BiRefNet:
    def __init__(self, device: torch.device, size: int = 1024, half: bool = True):
        from transformers import AutoModelForImageSegmentation
        self.device, self.size = device, size
        self.half = half and device.type == "cuda"
        m = AutoModelForImageSegmentation.from_pretrained(REPO, trust_remote_code=True)
        self.model = (m.half() if self.half else m.float()).to(device).eval()

    @torch.no_grad()
    def masks(self, imgs: list[Image.Image]) -> list[np.ndarray]:
        x = torch.stack([torch.from_numpy(np.asarray(im.convert("RGB").resize((self.size, self.size), Image.BILINEAR),
                                                     dtype=np.float32) / 255.0).permute(2, 0, 1) for im in imgs])
        x = ((x - MEAN) / STD).to(self.device, torch.float16 if self.half else torch.float32)
        pred = self.model(x)[-1].sigmoid().float()
        out = []
        for p, im in zip(pred, imgs):
            w, h = im.size
            out.append(F.interpolate(p[None], size=(h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy())
        return out
