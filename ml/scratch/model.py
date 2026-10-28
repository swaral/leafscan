"""LeafResNet: a CNN designed for this project and trained from random initialisation.

ResNet-style backbone with the "bag of tricks" refinements that matter most when
training from scratch:
  * deep 3-conv stem (ResNet-C) instead of one 7x7 conv
  * average-pool shortcut on downsampling (ResNet-D), so no information is skipped
  * squeeze-and-excitation channel attention in every block
  * stochastic depth (drop-path) for regularisation
  * zero-initialised last BN in each residual branch (blocks start as identity)
Head: global average pool -> dropout -> linear. A GAP+linear head makes class
activation maps exact, so the API's heatmaps work unchanged.

Both models return (logits, final_feature_map).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

CONFIGS = {
    # widths per stage, blocks per stage
    "s": ((64, 128, 256, 512), (2, 2, 3, 2)),
    "m": ((64, 128, 256, 512), (2, 3, 4, 2)),
}


def conv_bn(cin, cout, k=3, stride=1, act=True):
    layers = [nn.Conv2d(cin, cout, k, stride, k // 2, bias=False), nn.BatchNorm2d(cout)]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class SqueezeExcite(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        h = max(8, c // r)
        self.fc1, self.fc2 = nn.Conv2d(c, h, 1), nn.Conv2d(h, c, 1)

    def forward(self, x):
        s = x.mean((2, 3), keepdim=True)
        return x * torch.sigmoid(self.fc2(F.relu(self.fc1(s), inplace=True)))


def drop_path(x, p: float, training: bool):
    if p == 0.0 or not training:
        return x
    keep = torch.rand(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype) >= p
    return x * keep / (1 - p)


class Block(nn.Module):
    def __init__(self, cin, cout, stride, drop):
        super().__init__()
        self.branch = nn.Sequential(
            conv_bn(cin, cout, 3, stride),
            conv_bn(cout, cout, 3, 1, act=False),
            SqueezeExcite(cout),
        )
        nn.init.zeros_(self.branch[1][1].weight)  # zero-init gamma: block starts as identity
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.AvgPool2d(stride, stride, ceil_mode=True) if stride > 1 else nn.Identity(),
                conv_bn(cin, cout, 1, 1, act=False),
            )
        else:
            self.short = nn.Identity()
        self.drop = drop

    def forward(self, x):
        return F.relu(self.short(x) + drop_path(self.branch(x), self.drop, self.training), inplace=True)


class LeafResNet(nn.Module):
    def __init__(self, num_classes: int, size: str = "s", drop_path_rate: float = 0.1, dropout: float = 0.3):
        super().__init__()
        widths, depths = CONFIGS[size]
        self.stem = nn.Sequential(
            conv_bn(3, 32, 3, 2), conv_bn(32, 32), conv_bn(32, widths[0]),
            nn.MaxPool2d(3, 2, 1),
        )
        total, k = sum(depths), 0
        stages, cin = [], widths[0]
        for i, (w, d) in enumerate(zip(widths, depths)):
            blocks = []
            for j in range(d):
                blocks.append(Block(cin, w, 2 if (j == 0 and i > 0) else 1, drop_path_rate * k / max(1, total - 1)))
                cin, k = w, k + 1
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(cin, num_classes)
        self.feat_channels = cin
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, with_features: bool = False):
        fmap = self.stages(self.stem(x))
        return self.fc(self.dropout(fmap.mean((2, 3)))), (fmap if with_features else None)


class PretrainedMobileNet(nn.Module):
    """Transfer-learning baseline (ImageNet MobileNetV3-Large, fully fine-tuned) for comparison."""

    def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
        base = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
        self.features = base.features
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(960, num_classes)
        self.feat_channels = 960

    def forward(self, x, with_features: bool = False):
        fmap = self.features(x)
        return self.fc(self.dropout(fmap.mean((2, 3)))), (fmap if with_features else None)


PLANTCLEF_ARCH = "vit_base_patch14_reg4_dinov2.lvd142m"


def load_plantclef_weights(vit: nn.Module, path: str) -> None:
    """Load the PlantCLEF 2024 checkpoint (timm training format, EMA weights preferred),
    drop its 7,806-species head and resample position embeddings to our input size."""
    from timm.models.vision_transformer import checkpoint_filter_fn
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict_ema") or ck.get("state_dict") or ck
    sd = {k.removeprefix("module."): v for k, v in sd.items() if not k.startswith("head.")}
    sd = checkpoint_filter_fn(sd, vit)
    missing, unexpected = vit.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"PlantCLEF weights mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")


class PlantCLEFViT(nn.Module):
    """DINOv2 ViT-B/14 with 4 register tokens, fine-tuned by Pl@ntNet on PlantCLEF 2024
    (~1.4M plant photos, 7,806 species). We replace the species head with ours.

    Head = mean of patch tokens -> dropout -> linear. The patch tokens form a
    (768, H/14, W/14) grid, so class activation maps stay exact for the API."""

    def __init__(self, num_classes: int, img: int = 224, checkpoint: str | None = None,
                 drop_path: float = 0.1, dropout: float = 0.2):
        super().__init__()
        import timm
        self.vit = timm.create_model(PLANTCLEF_ARCH, pretrained=False, num_classes=0, global_pool="",
                                     img_size=img, drop_path_rate=drop_path)
        if checkpoint:
            load_plantclef_weights(self.vit, checkpoint)
        self.grid = img // 14
        self.prefix = self.vit.num_prefix_tokens  # 1 class token + 4 registers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.vit.embed_dim, num_classes)
        self.feat_channels = self.vit.embed_dim
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, with_features: bool = False):
        tokens = self.vit.forward_features(x)[:, self.prefix:]          # (B, N, 768), LayerNormed
        logits = self.fc(self.dropout(tokens.mean(1)))
        if not with_features:
            return logits, None
        # grid is fixed by the input size (img/14); a constant keeps ONNX tracing static
        return logits, tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], self.grid, self.grid)

    def freeze_blocks(self, n: int) -> None:
        """Freeze patch embedding, tokens and the first n transformer blocks (no grads, no backward)."""
        if n <= 0:
            return
        for mod in (self.vit.patch_embed, *self.vit.blocks[:n]):
            for p in mod.parameters():
                p.requires_grad_(False)
        for name in ("cls_token", "reg_token", "pos_embed"):
            t = getattr(self.vit, name, None)
            if isinstance(t, nn.Parameter):
                t.requires_grad_(False)

    def layer_id(self, name: str) -> int:
        """Depth index for layer-wise learning-rate decay (0 = embeddings, 13 = head)."""
        if name.startswith("vit.blocks."):
            return int(name.split(".")[2]) + 1
        if name.startswith(("vit.patch_embed", "vit.cls_token", "vit.reg_token", "vit.pos_embed")):
            return 0
        return len(self.vit.blocks) + 1


def build_model(arch: str, num_classes: int, pretrained: bool = True, img: int = 224,
                checkpoint: str | None = None) -> nn.Module:
    if arch.startswith("leafresnet"):
        size = arch.split("_")[1] if "_" in arch else "s"
        return LeafResNet(num_classes, size)
    if arch == "mobilenet_v3_pretrained":
        return PretrainedMobileNet(num_classes, pretrained)
    if arch == "plantclef_vitb14":
        return PlantCLEFViT(num_classes, img, checkpoint if pretrained else None)
    raise ValueError(arch)


class ExportWrapper(nn.Module):
    """Inference graph matching the API contract: RGB [0,1] in -> (calibrated logits, feature map)."""

    def __init__(self, net: nn.Module, temperature: float):
        super().__init__()
        self.net = net.eval()
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def forward(self, x):
        logits, fmap = self.net((x - self.mean) / self.std, with_features=True)
        return logits / self.temperature, fmap


if __name__ == "__main__":
    for arch in ("leafresnet_s", "leafresnet_m"):
        m = build_model(arch, 95)
        n = sum(p.numel() for p in m.parameters())
        logits, f = m(torch.randn(2, 3, 224, 224), with_features=True)
        print(arch, f"{n / 1e6:.1f}M params", tuple(logits.shape), tuple(f.shape))
