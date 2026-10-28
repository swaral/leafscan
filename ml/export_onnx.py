"""Export the trained model to ONNX for a lightweight, torch-free API.

The graph takes RGB float32 in [0,1] (N,3,224,224), normalises internally and
returns temperature-calibrated logits plus the final 960x7x7 feature map. The
linear head weights are saved alongside so the API can compute class
activation maps (exactly equal to Grad-CAM for a GAP + linear head).

    python ml/export_onnx.py
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import onnx
import torch

from common import API_MODEL_DIR, ARTIFACTS, IMG_SIZE, ExportWrapper, LeafNet


def main():
    ckpt = torch.load(ARTIFACTS / "leafnet.pt", map_location="cpu")
    classes = ckpt["classes"]
    net = LeafNet(len(classes), pretrained=False)
    net.load_state_dict(ckpt["state_dict"])
    model = ExportWrapper(net, ckpt["temperature"]).eval()

    API_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = API_MODEL_DIR / "leafnet.onnx"
    dummy = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(model, dummy, str(out), input_names=["image"],
                      output_names=["logits", "features"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"},
                                    "features": {0: "batch"}},
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(str(out)))

    np.savez(API_MODEL_DIR / "cam_head.npz",
             weight=net.fc.weight.detach().numpy().astype(np.float32),
             bias=net.fc.bias.detach().numpy().astype(np.float32))
    (API_MODEL_DIR / "labels.json").write_text(json.dumps(classes, indent=2))

    # Sanity check: ONNX output must match PyTorch.
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = torch.rand(2, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        ref, _ = model(x)
    got = sess.run(["logits"], {"image": x.numpy()})[0]
    diff = float(np.abs(ref.numpy() - got).max())
    print(f"exported {out} ({out.stat().st_size / 1e6:.1f} MB), max |torch-onnx| = {diff:.2e}")
    assert diff < 1e-3

    for f in ("train_history.json", "dataset_stats.json"):
        if (ARTIFACTS / f).exists():
            shutil.copy(ARTIFACTS / f, API_MODEL_DIR / f)


if __name__ == "__main__":
    main()
