"""Train on the combined dataset (GPU; falls back to CPU for smoke tests).

    python train.py --data /content/leafscan_data --out /content/drive/MyDrive/leafscan/runs/scratch \
                    --arch leafresnet_s --epochs 60
    python train.py ... --arch mobilenet_v3_pretrained --epochs 15 --lr 1e-3   # transfer baseline

Pipeline: DataLoader workers only read JPEG bytes; decoding (nvJPEG) and all
augmentation run batched on the GPU, so Colab's 2 CPU cores are not the bottleneck.
Recipe: AdamW + warmup/cosine, label smoothing, mixup/cutmix, random-resized-crop,
flips/rot90, colour jitter, random erasing, class-balanced sampling (with extra
weight on real-world field photos), EMA weights, AMP, resumable checkpoints.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.io import decode_jpeg, encode_jpeg
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as TF

from model import MEAN, STD, build_model

# ---------------------------------------------------------------- data
class BytesDataset(Dataset):
    def __init__(self, root: Path, rows):
        self.root, self.rows = root, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        # Decoded here on the CPU (workers), returned as a numpy CHW uint8 array:
        #  - GPU nvJPEG decoding leaked GPU memory and crashed long runs at epoch 9,
        #  - numpy (not torch tensors) avoids one shared file descriptor per image (EBADF on Kaggle).
        r = self.rows[i]
        img = np.asarray(Image.open(self.root / r["path"]).convert("RGB"), dtype=np.uint8)
        return np.ascontiguousarray(img.transpose(2, 0, 1)), int(r["label"]), i


def collate(batch):
    data, labels, idx = zip(*batch)
    return list(data), torch.tensor(labels), torch.tensor(idx)


def load_split(root: Path):
    with (root / "manifest.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    classes = json.loads((root / "classes.json").read_text())
    return {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}, classes


_GPU_DECODE_FAILURES = 0


def decode(batch, device) -> list[torch.Tensor]:
    """Images arrive either already decoded (CHW uint8 numpy from BytesDataset) or as JPEG bytes
    (bench.py). Decoded images are just moved to the device."""
    global _GPU_DECODE_FAILURES
    if batch and isinstance(batch[0], np.ndarray) and batch[0].ndim == 3:
        return [torch.from_numpy(b).to(device, non_blocking=True) for b in batch]
    batch = [torch.from_numpy(b) if isinstance(b, np.ndarray) else b for b in batch]
    if device.type == "cuda":
        try:
            return decode_jpeg(batch, device=device)
        except RuntimeError as e:  # nvJPEG fails sporadically after thousands of batches
            _GPU_DECODE_FAILURES += 1
            if _GPU_DECODE_FAILURES <= 3:
                print(f"  GPU JPEG decode failed ({e}); decoding this batch on CPU", flush=True)
            torch.cuda.empty_cache()
            return [decode_jpeg(b).to(device, non_blocking=True) for b in batch]
    return [decode_jpeg(b) for b in batch]


class GpuAugment:
    """Batched augmentation on uint8 CHW images of varying size -> normalised (B,3,S,S) float."""

    def __init__(self, size: int, device, train: bool, degrade: float = 0.0):
        self.size, self.device, self.train, self.degrade = size, device, train, degrade
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
        self.rrc = RandomResizedCrop(size, scale=(0.25, 1.0), ratio=(3 / 4, 4 / 3))

    def __call__(self, imgs: list[torch.Tensor]) -> torch.Tensor:
        out = []
        for im in imgs:
            if self.train:
                i, j, h, w = self.rrc.get_params(im, self.rrc.scale, self.rrc.ratio)
                im = im[:, i:i + h, j:j + w]
                k = int(torch.randint(0, 4, (1,)))
                if k:
                    im = torch.rot90(im, k, (1, 2))
            # eval: plain resize to SxS (same as the API's preprocessing)
            out.append(F.interpolate(im[None].float(), size=(self.size, self.size), mode="bilinear",
                                     antialias=True, align_corners=False)[0])
        x = torch.stack(out) / 255.0
        if self.train and self.degrade > 0:
            x = degrade(x, self.degrade)
        if self.train:
            b = x.shape[0]
            flip = torch.rand(b, device=x.device) < 0.5
            x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
            # colour jitter: brightness, contrast, saturation
            def rnd(lo, hi):
                return torch.empty(b, 1, 1, 1, device=x.device).uniform_(lo, hi)
            x = x * rnd(0.7, 1.3)
            gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
            x = (x - gray.mean((2, 3), keepdim=True)) * rnd(0.7, 1.3) + gray.mean((2, 3), keepdim=True)
            gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
            x = gray + (x - gray) * rnd(0.6, 1.4)
            x = x.clamp(0, 1)
        x = (x - self.mean) / self.std
        if self.train:  # random erasing (p=0.25), filled with noise
            b = x.shape[0]
            for n in torch.nonzero(torch.rand(b) < 0.25).flatten().tolist():
                eh, ew = (int(self.size * float(torch.empty(1).uniform_(0.1, 0.4))) for _ in range(2))
                ey, ex = int(torch.randint(0, self.size - eh + 1, (1,))), int(torch.randint(0, self.size - ew + 1, (1,)))
                x[n, :, ey:ey + eh, ex:ex + ew] = torch.randn(3, eh, ew, device=x.device)
        return x.contiguous(memory_format=torch.channels_last)


def _jpeg(x: torch.Tensor, quality: int) -> torch.Tensor:
    """Re-compress a batch (B,3,H,W in [0,1]) as JPEG; GPU encoder when available."""
    u8 = (x * 255).round().clamp(0, 255).to(torch.uint8)
    try:
        enc = [e.cpu() for e in encode_jpeg(list(u8), quality=quality)]  # decoder wants CPU bytes
        dec = decode_jpeg(enc, device=x.device) if x.is_cuda else [decode_jpeg(e) for e in enc]
    except (RuntimeError, TypeError, ValueError):  # older torchvision: CPU round-trip
        enc = [encode_jpeg(t.cpu(), quality=quality) for t in u8]
        dec = [decode_jpeg(e).to(x.device) for e in enc]
    return torch.stack(dec).float() / 255.0


def degrade(x: torch.Tensor, p: float) -> torch.Tensor:
    """Randomise acquisition fingerprints (sharpness, resampling, sensor noise, compression)
    so the model cannot identify a dataset/camera from them. Real-ESRGAN-style pipeline."""
    b, _, h, w = x.shape
    sel = torch.nonzero(torch.rand(b) < p).flatten()
    if len(sel) == 0:
        return x
    y = x[sel]
    if np.random.rand() < 0.5:  # blur
        sigma = float(np.random.uniform(0.2, 1.5))
        y = TF.gaussian_blur(y, kernel_size=2 * math.ceil(2 * sigma) + 1, sigma=sigma)
    if np.random.rand() < 0.5:  # down- then up-sample
        s = float(np.random.uniform(0.35, 0.9))
        mode = ["bilinear", "bicubic", "nearest"][np.random.randint(3)]
        y = F.interpolate(y, scale_factor=s, mode="bilinear", antialias=True, align_corners=False)
        y = F.interpolate(y, size=(h, w), mode=mode, **({} if mode == "nearest" else {"align_corners": False}))
    if np.random.rand() < 0.5:  # sensor noise (gaussian, optionally colour)
        std = float(np.random.uniform(0.005, 0.04))
        n = torch.randn_like(y[:, :1]) if np.random.rand() < 0.5 else torch.randn_like(y)
        y = y + n * std
    if np.random.rand() < 0.7:  # JPEG re-compression at a random quality
        y = _jpeg(y.clamp(0, 1), int(np.random.randint(30, 96)))
    x = x.clone()
    x[sel] = y.clamp(0, 1)
    return x


def mix(x, y_soft, mixup_alpha, cutmix_alpha, prob):
    """Mixup or CutMix on a batch with soft targets."""
    if np.random.rand() > prob:
        return x, y_soft
    perm = torch.randperm(x.shape[0], device=x.device)
    if np.random.rand() < 0.5 and cutmix_alpha > 0:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        H, W = x.shape[2:]
        ch, cw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
        cy, cx = np.random.randint(H), np.random.randint(W)
        y0, y1, x0, x1 = max(cy - ch // 2, 0), min(cy + ch // 2, H), max(cx - cw // 2, 0), min(cx + cw // 2, W)
        x = x.clone()
        x[:, :, y0:y1, x0:x1] = x[perm, :, y0:y1, x0:x1]
        lam = 1 - (y1 - y0) * (x1 - x0) / (H * W)
    else:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        x = lam * x + (1 - lam) * x[perm]
    return x, lam * y_soft + (1 - lam) * y_soft[perm]


# ---------------------------------------------------------------- EMA
class EMA:
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1 + step) / (10 + step))  # warm up the average early in training
        for e, m in zip(self.model.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                e.copy_(m)


# ---------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, loader, aug, device, rows, n_classes, amp):
    model.eval()
    logits_all, y_all, idx_all = [], [], []
    for data, y, idx in loader:
        x = aug(decode(data, device))
        with torch.autocast(device.type, enabled=amp):
            logits, _ = model(x)
        logits_all.append(logits.float().cpu()); y_all.append(y); idx_all.append(idx)
    logits = torch.cat(logits_all)
    y = torch.cat(y_all)
    idx = torch.cat(idx_all)
    pred = logits.argmax(1)
    res = {"loss": float(F.cross_entropy(logits, y)), "acc": float((pred == y).float().mean()),
           "macro_f1": macro_f1(y.numpy(), pred.numpy(), n_classes)}
    for dom in ("lab", "field"):
        m = torch.tensor([rows[i]["domain"] == dom for i in idx.tolist()])
        if m.any():
            res[f"acc_{dom}"] = float((pred[m] == y[m]).float().mean())
    return res, logits, y


def macro_f1(y, p, n):
    f1s = []
    for c in range(n):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        if tp + fp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else 0.0


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="leafresnet_s")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=float, default=3, help="warmup epochs")
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--cutmix", type=float, default=1.0)
    ap.add_argument("--mix-prob", type=float, default=0.5)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--field-boost", type=float, default=1.5, help="sampling weight multiplier for field photos")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    ap.add_argument("--max-steps-per-epoch", type=int, default=0, help="smoke tests")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--checkpoint", default=None, help="PlantCLEF weights for --arch plantclef_vitb14")
    ap.add_argument("--layer-decay", type=float, default=1.0, help="layer-wise LR decay (ViT fine-tuning: 0.75)")
    ap.add_argument("--degrade", type=float, default=0.5, help="probability of acquisition-degradation aug")
    ap.add_argument("--freeze-blocks", type=int, default=0, help="ViT: freeze patch embed + first N blocks")
    ap.add_argument("--init", default=None, help="start from a trained best.pt (its EMA weights)")
    ap.add_argument("--ext-boost", type=float, default=1.0,
                    help="extra sampling weight for added external field photos (source ext_*)")
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    torch.backends.cudnn.benchmark = True
    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    splits, classes = load_split(root)
    n_classes = len(classes)
    print(f"device {device} | classes {n_classes} | " +
          " ".join(f"{k} {len(v)}" for k, v in splits.items()), flush=True)

    # class-balanced sampling (1/sqrt(n)), with real-world photos boosted
    train_rows = splits["train"]
    counts = Counter(int(r["label"]) for r in train_rows)
    w = [(1 / math.sqrt(counts[int(r["label"])])) * (args.field_boost if r["domain"] == "field" else 1.0)
         * (args.ext_boost if r["source"].startswith("ext_") else 1.0) for r in train_rows]
    sampler = WeightedRandomSampler(w, num_samples=len(train_rows), replacement=True)
    kw = dict(batch_size=args.batch, num_workers=args.workers, collate_fn=collate,
              pin_memory=False, persistent_workers=args.workers > 0,
              timeout=600 if args.workers > 0 else 0)  # a stalled worker raises instead of hanging
    train_dl = DataLoader(BytesDataset(root, train_rows), sampler=sampler, drop_last=True, **kw)
    val_dl = DataLoader(BytesDataset(root, splits["val"]), shuffle=False, **kw)
    aug_train = GpuAugment(args.img, device, True, args.degrade)
    aug_eval = GpuAugment(args.img, device, False)

    core = build_model(args.arch, n_classes, pretrained=not args.no_pretrained, img=args.img,
                       checkpoint=args.checkpoint).to(device)
    if args.init:
        init = torch.load(args.init, map_location="cpu", weights_only=False)
        assert init["classes"] == classes, "--init checkpoint has a different class list"
        core.load_state_dict(init["ema"])
        print(f"initialised from {args.init} (epoch {init.get('epoch')})", flush=True)
    core = core.to(memory_format=torch.channels_last)
    if args.freeze_blocks and hasattr(core, "freeze_blocks"):
        core.freeze_blocks(args.freeze_blocks)
    n_train = sum(p.numel() for p in core.parameters() if p.requires_grad)
    print(f"trainable params {n_train / 1e6:.1f}M of {sum(p.numel() for p in core.parameters()) / 1e6:.1f}M", flush=True)
    ema = EMA(core, args.ema)
    model = nn.DataParallel(core) if torch.cuda.device_count() > 1 else core
    if torch.cuda.device_count() > 1:
        print(f"using {torch.cuda.device_count()} GPUs (DataParallel)", flush=True)

    # parameter groups: no weight decay on norms/biases/tokens; layer-wise LR decay for ViTs
    groups = {}
    n_layers = len(core.vit.blocks) + 1 if hasattr(core, "vit") else 0
    for name, p in core.named_parameters():
        if not p.requires_grad:
            continue
        lid = core.layer_id(name) if hasattr(core, "layer_id") else 0
        no_wd = p.ndim <= 1 or name.endswith(("cls_token", "reg_token", "pos_embed"))
        key = (lid, no_wd)
        if key not in groups:
            scale = args.layer_decay ** (n_layers - lid) if n_layers else 1.0
            groups[key] = {"params": [], "weight_decay": 0.0 if no_wd else args.wd, "lr_scale": scale}
        groups[key]["params"].append(p)
    opt = torch.optim.AdamW(list(groups.values()), lr=args.lr, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler(enabled=amp)
    steps_per_epoch = args.max_steps_per_epoch or len(train_dl)
    total, warm = args.epochs * steps_per_epoch, int(args.warmup * steps_per_epoch)

    def lr_at(step):
        if step < warm:
            return args.lr * (step + 1) / warm
        t = (step - warm) / max(1, total - warm)
        return 1e-6 + 0.5 * (args.lr - 1e-6) * (1 + math.cos(math.pi * t))

    start_epoch, step, history, best = 1, 0, [], -1.0
    ckpt = out / "last.pt"
    if ckpt.exists():
        s = torch.load(ckpt, map_location=device, weights_only=False)
        core.load_state_dict(s["model"]); ema.model.load_state_dict(s["ema"])
        opt.load_state_dict(s["opt"]); scaler.load_state_dict(s["scaler"])
        start_epoch, step, history, best = s["epoch"] + 1, s["step"], s["history"], s["best"]
        print(f"resumed from epoch {s['epoch']} (best val macro-F1 {best:.4f})", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0, seen, correct, loss_sum = time.time(), 0, 0.0, 0.0
        for it, (data, y, _) in enumerate(train_dl):
            if args.max_steps_per_epoch and it >= args.max_steps_per_epoch:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step) * g["lr_scale"]
            y = y.to(device, non_blocking=True)
            x = aug_train(decode(data, device))
            y_soft = F.one_hot(y, n_classes).float() * (1 - args.smoothing) + args.smoothing / n_classes
            x, y_mix = mix(x, y_soft, args.mixup, args.cutmix, args.mix_prob)
            with torch.autocast(device.type, enabled=amp):
                logits, _ = model(x)
                loss = torch.sum(-y_mix * F.log_softmax(logits.float(), 1), 1).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_([p for p in core.parameters() if p.requires_grad], 5.0)
            scaler.step(opt); scaler.update()
            ema.update(core, step)
            step += 1
            seen += len(y); loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y_mix.argmax(1)).sum().item()
            if it % 100 == 0:
                rate = seen / (time.time() - t0)
                print(f"  ep {epoch} it {it}/{steps_per_epoch} loss {loss.item():.3f} "
                      f"lr {lr_at(step):.2e} {rate:.0f} img/s", flush=True)

        val, _, _ = evaluate(ema.model, val_dl, aug_eval, device, splits["val"], n_classes, amp)
        rec = {"epoch": epoch, "train_loss": loss_sum / max(seen, 1), "train_acc": correct / max(seen, 1),
               "val_loss": val["loss"], "val_acc": val["acc"], "val_macro_f1": val["macro_f1"],
               "val_acc_lab": val.get("acc_lab"), "val_acc_field": val.get("acc_field"),
               "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        print(f"[epoch {epoch}/{args.epochs}] train loss {rec['train_loss']:.3f} | val acc {val['acc']:.4f} "
              f"macroF1 {val['macro_f1']:.4f} lab {val.get('acc_lab', 0):.4f} field {val.get('acc_field', 0):.4f} "
              f"| {rec['seconds']:.0f}s", flush=True)

        meta = {"arch": args.arch, "classes": classes, "img": args.img, "args": vars(args)}
        if val["macro_f1"] > best:
            best = val["macro_f1"]
            torch.save({"ema": ema.model.state_dict(), "epoch": epoch, **meta}, out / "best.pt")
        torch.save({"model": core.state_dict(), "ema": ema.model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "step": step, "history": history,
                    "best": best, **meta}, out / "last.pt.tmp")
        os.replace(out / "last.pt.tmp", ckpt)  # atomic, so a disconnect never corrupts the checkpoint
        (out / "history.json").write_text(json.dumps(history, indent=2))

    print(f"done. best val macro-F1 {best:.4f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
