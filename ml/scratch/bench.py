"""Measure real training/eval throughput of the fine-tuning pipeline on this GPU.

Uses synthetic JPEGs but the exact same steps as train.py (GPU decode, augmentation
incl. degradation, mixup, AMP forward/backward, AdamW, EMA), so the img/s figure
converts directly into time per epoch.

    python bench.py --arch plantclef_vitb14 --batch 128 --img 224
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import encode_jpeg

from model import build_model
from train import EMA, GpuAugment, decode, mix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="plantclef_vitb14")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--classes", type=int, default=95)
    ap.add_argument("--degrade", type=float, default=0.5)
    ap.add_argument("--single-gpu", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--freeze-blocks", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda"
    torch.backends.cudnn.benchmark = True

    # synthetic 256x320 JPEGs with natural-ish structure (smoothed noise)
    rng = np.random.default_rng(0)
    pool = []
    for _ in range(64):
        a = rng.integers(0, 255, (3, 32, 40), dtype=np.uint8)
        t = F.interpolate(torch.from_numpy(a)[None].float(), size=(256, 320), mode="bilinear")[0]
        pool.append(encode_jpeg(t.clamp(0, 255).to(torch.uint8), quality=90))

    core = build_model(args.arch, args.classes, pretrained=False, img=args.img).to(dev)
    core = core.to(memory_format=torch.channels_last)
    if args.freeze_blocks:
        core.freeze_blocks(args.freeze_blocks)
    multi = torch.cuda.device_count() > 1 and not args.single_gpu
    model = nn.DataParallel(core) if multi else core
    ema = EMA(core, 0.9995)
    opt = torch.optim.AdamW([p for p in core.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.amp.GradScaler(enabled=amp)
    aug, aug_eval = GpuAugment(args.img, dev, True, args.degrade), GpuAugment(args.img, dev, False)
    timing = {"input": 0.0, "model": 0.0}

    def sync():
        if amp:
            torch.cuda.synchronize()

    def batch():
        return [pool[i] for i in rng.integers(0, len(pool), args.batch)], torch.randint(0, args.classes, (args.batch,))

    def train_step(step):
        data, y = batch()
        sync(); t0 = time.time()
        y = y.to(dev)
        x = aug(decode(data, dev))
        sync(); t1 = time.time()
        ys = F.one_hot(y, args.classes).float()
        x, ym = mix(x, ys, 0.8, 1.0, 0.5)
        with torch.autocast(dev.type, enabled=amp):
            logits, _ = model(x)
            loss = torch.sum(-ym * F.log_softmax(logits.float(), 1), 1).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        ema.update(core, step)
        sync(); timing["input"] += t1 - t0; timing["model"] += time.time() - t1

    for i in range(8):  # warm-up (cudnn autotune, allocator)
        train_step(i)
    timing.update(input=0.0, model=0.0)
    if amp:
        torch.cuda.synchronize()
    t0 = time.time()
    for i in range(args.steps):
        train_step(i)
    if amp:
        torch.cuda.synchronize()
    train_ips = args.steps * args.batch / (time.time() - t0)

    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(20):
            data, _ = batch()
            with torch.autocast(dev.type, enabled=amp):
                model(aug_eval(decode(data, dev)))
    if amp:
        torch.cuda.synchronize()
    eval_ips = 20 * args.batch / (time.time() - t0)
    mem = torch.cuda.max_memory_allocated() / 1e9 if amp else 0
    tot = timing["input"] + timing["model"]
    res = {"tag": args.tag, "freeze_blocks": args.freeze_blocks, "multi_gpu": multi, "degrade": args.degrade,
           "input_share": round(timing["input"] / tot, 3), "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
           "arch": args.arch, "img": args.img, "batch": args.batch,
           "train_img_per_s": round(train_ips, 1), "eval_img_per_s": round(eval_ips, 1),
           "peak_mem_gb_gpu0": round(mem, 2)}
    print("BENCH " + json.dumps(res), flush=True)
    return res


if __name__ == "__main__":
    main()
