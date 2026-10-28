"""Shared model definition, preprocessing and paths for the ML pipeline."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_ZIP = ROOT / "data_raw" / "plantvillage.zip"
EXTRACT_DIR = DATA_DIR / "plantvillage"
MANIFEST = DATA_DIR / "manifest.csv"
CACHE_DIR = DATA_DIR / "cache"
ARTIFACTS = ROOT / "ml" / "artifacts"
API_MODEL_DIR = ROOT / "api" / "models"

IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# MobileNetV3-Large `features` is 17 modules. Everything before SPLIT stays frozen
# (its activations are cached once); blocks SPLIT.. plus the head are fine-tuned.
# This makes real fine-tuning feasible on a laptop CPU.
SPLIT = 14
FEAT_CHANNELS = 960


class LeafNet(nn.Module):
    """MobileNetV3-Large backbone + linear head (so class activation maps are exact)."""

    def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        base = mobilenet_v3_large(weights=weights)
        self.stem = base.features[:SPLIT]
        self.tail = base.features[SPLIT:]
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(FEAT_CHANNELS, num_classes)

    def head(self, stem_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fmap = self.tail(stem_out)
        logits = self.fc(self.dropout(fmap.mean(dim=(2, 3))))
        return logits, fmap

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head(self.stem(x))


class ExportWrapper(nn.Module):
    """Inference graph for ONNX: takes RGB in [0,1], returns calibrated logits + feature map."""

    def __init__(self, net: LeafNet, temperature: float):
        super().__init__()
        self.net = net.eval()
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, fmap = self.net((x - self.mean) / self.std)
        return logits / self.temperature, fmap


def eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def train_transform() -> transforms.Compose:
    """Augmentations that mimic field photos: framing, rotation, lighting, blur."""
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(0.35, 0.35, 0.3, 0.04),
        transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.25),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
