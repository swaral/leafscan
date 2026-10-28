"""Generate self-contained Kaggle script kernels (code embedded) under ml/scratch/kaggle/.

    python ml/scratch/make_kaggle.py --user thestonedape
    python -m kaggle.cli kernels push -p ml/scratch/kaggle/build      # CPU: datasets + PlantCLEF weights
    python -m kaggle.cli kernels push -p ml/scratch/kaggle/bench      # GPU: throughput benchmark
    python -m kaggle.cli kernels push -p ml/scratch/kaggle/finetune   # GPU: fine-tune + export
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent

PRELUDE = '''import os, sys, subprocess, json, shutil, time, glob, tarfile
from pathlib import Path

FILES = __FILES__
CODE = Path("/tmp/leafscan_code")
CODE.mkdir(parents=True, exist_ok=True)
for name, src in FILES.items():
    (CODE / name).write_text(src, encoding="utf-8")
WORK = Path("/kaggle/working")


def run(*args, **kw):
    print("$", " ".join(map(str, args)), flush=True)
    t = time.time()
    subprocess.run([str(a) for a in args], cwd=CODE, check=True, **kw)
    print(f"[done in {(time.time() - t) / 60:.1f} min]", flush=True)


def pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)

'''

FETCH_PLANTCLEF = '''
import requests
PLANTCLEF = Path("/tmp/plantclef_vitb14_ema.pth")
# PlantCLEF 2024 checkpoint (Zenodo record 10848263, CC BY 4.0): fetch only the fully
# fine-tuned model from the 2.3 GB tar with HTTP range requests, keep the EMA weights.
URL = ("https://zenodo.org/api/records/10848263/files/"
       "PlantNet_PlantCLEF2024_pretrained_models_on_the_flora_of_south-western_europe.tar/content")
url = requests.head(URL, allow_redirects=True, timeout=60).url
off = 0
while True:
    h = requests.get(url, headers={"Range": f"bytes={off}-{off + 511}"}, timeout=60).content
    if len(h) < 512 or h == b"\\0" * 512:
        raise SystemExit("PlantCLEF checkpoint not found in tar")
    ti = tarfile.TarInfo.frombuf(h, "utf-8", "surrogateescape")
    if ti.name.endswith("onlyclassifier_then_all/model_best.pth.tar") and "/._" not in ti.name:
        break
    off += 512 + ((ti.size + 511) // 512) * 512
print(f"downloading {ti.name} ({ti.size / 1e9:.2f} GB)", flush=True)
rng = {"Range": f"bytes={off + 512}-{off + 512 + ti.size - 1}"}
with requests.get(url, headers=rng, stream=True, timeout=600) as r:
    with open("/tmp/plantclef_full.pth.tar", "wb") as fh:
        for b in r.iter_content(1 << 24):
            fh.write(b)
import torch
ck = torch.load("/tmp/plantclef_full.pth.tar", map_location="cpu", weights_only=False)
sd = ck.get("state_dict_ema") or ck["state_dict"]
torch.save({"state_dict_ema": sd}, PLANTCLEF)
print(f"saved EMA weights: {PLANTCLEF.stat().st_size / 1e6:.0f} MB", flush=True)
'''

BUILD = '''
import requests
t_start = time.time()
DATA = Path("/tmp/leafscan_data")
run(sys.executable, "-u", "build_dataset.py", "--out", DATA, "--workers", "32")

print("archiving dataset ...", flush=True)
subprocess.run(["tar", "--exclude=raw", "--exclude=records_*", "--exclude=.done_*",
                "-cf", str(WORK / "leafscan_data.tar"), "-C", "/tmp", "leafscan_data"], check=True)
for f in ("stats.json", "classes.json"):
    shutil.copy(DATA / f, WORK / f)
print(f"archive {(WORK / 'leafscan_data.tar').stat().st_size / 1e9:.2f} GB", flush=True)

print(f"BUILD DONE in {(time.time() - t_start) / 60:.1f} min", flush=True)
'''

BENCH = '''
pip("timm>=1.0")
for tag, extra in (("freeze4", ["--freeze-blocks", "4"]), ("freeze6", ["--freeze-blocks", "6"])):
    run(sys.executable, "-u", "bench.py", "--arch", "plantclef_vitb14", "--img", 224, "--batch", 128,
        "--steps", 40, "--tag", tag, *extra)
'''

EXPORT = '''
t_start = time.time()
pip("timm>=1.0", "onnx", "onnxruntime", "onnxscript")
tars = glob.glob("/kaggle/input/**/leafscan_data.tar", recursive=True)
bests = glob.glob("/kaggle/input/**/best.pt", recursive=True)
hists = glob.glob("/kaggle/input/**/history.json", recursive=True)
if not (tars and bests):
    subprocess.run(["find", "/kaggle/input", "-maxdepth", "4"], check=False)
    raise SystemExit("need leafscan_data.tar and best.pt inputs")
subprocess.run(["tar", "-xf", tars[0], "-C", "/tmp"], check=True)
RUN = Path("/tmp/run"); RUN.mkdir(exist_ok=True)
shutil.copy(bests[0], RUN / "best.pt")
if hists:
    shutil.copy(hists[0], RUN / "history.json")
run(sys.executable, "-u", "export.py", "--data", "/tmp/leafscan_data", "--run", RUN, "--out", WORK / "api_model",
    timeout=3600)
shutil.make_archive(str(WORK / "leafscan_api_model"), "zip", WORK / "api_model")
print(f"EXPORT DONE in {(time.time() - t_start) / 60:.1f} min", flush=True)
'''

RESUME_INPUTS = '''RUN.mkdir(parents=True, exist_ok=True)
for name in ("last.pt", "best.pt", "history.json"):
    found = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not found:
        raise SystemExit(f"resume input {name} not found")
    shutil.copy(found[0], RUN / name)
print("resuming from", RUN / "last.pt", flush=True)
'''

SEGCOMPARE = '''
pip("timm>=1.0", "kornia", "einops", "transformers")
tars = glob.glob("/kaggle/input/**/leafscan_data.tar", recursive=True)
if not tars:
    raise SystemExit("attach leafscan-build-dataset")
subprocess.run(["tar", "-xf", tars[0], "-C", "/tmp"], check=True)
run(sys.executable, "-u", "seg_compare.py", "/tmp/leafscan_data", WORK, timeout=3000)
'''

PREPROCESS = '''
t_start = time.time()
pip("timm>=1.0", "kornia", "einops", "transformers")
tars = glob.glob("/kaggle/input/**/leafscan_data.tar", recursive=True)
if not tars:
    raise SystemExit("attach leafscan-build-dataset")
subprocess.run(["tar", "-xf", tars[0], "-C", "/tmp"], check=True)
run(sys.executable, "-u", "preprocess_dataset.py", "--data", "/tmp/leafscan_data", "--out", "/tmp/leafscan_pp",
    timeout=5 * 3600)
for f in ("pp_stats.json", "pp_config.json", "pp_samples.jpg"):
    shutil.copy(f"/tmp/leafscan_pp/{f}", WORK / f)
subprocess.run(["tar", "--exclude=gates_*", "-cf", str(WORK / "leafscan_data_pp.tar"), "-C", "/tmp", "leafscan_pp"], check=True)
print(f"PREPROCESS KERNEL DONE in {(time.time() - t_start) / 60:.1f} min", flush=True)
'''

EXTERNAL = '''
t_start = time.time()
pip("timm>=1.0", "kornia", "einops", "transformers", "pyarrow")
def find(pattern):
    hits = glob.glob(pattern, recursive=True)
    if not hits:
        subprocess.run(["find", "/kaggle/input", "-maxdepth", "6", "-name", "*.json"], check=False)
        raise SystemExit(f"missing input: {pattern}")
    return hits[0]
subprocess.run(["tar", "-xf", find("/kaggle/input/**/leafscan_data.tar"), "-C", "/tmp"], check=True)
raw, pp = Path("/tmp/models/raw"), Path("/tmp/models/pp")
raw.mkdir(parents=True, exist_ok=True); pp.mkdir(parents=True, exist_ok=True)
shutil.copy(find("/kaggle/input/**/leafscan-plantclef-vit-epoch8/**/best.pt"), raw / "best.pt")
shutil.copy(find("/kaggle/input/**/leafscan-export-plantclef/**/api_model/train_history.json"), raw / "train_history.json")
shutil.copy(find("/kaggle/input/**/leafscan-resume-pp/**/run/best.pt"), pp / "best.pt")
shutil.copy(find("/kaggle/input/**/leafscan-resume-pp/**/api_model/train_history.json"), pp / "train_history.json")
shutil.copy(find("/kaggle/input/**/leafscan-resume-pp/**/api_model/pp_config.json"), pp / "pp_config.json")
run(sys.executable, "-u", "external_test.py", "--train-data", "/tmp/leafscan_data",
    "--models", f"current={raw}", f"preprocessed={pp}", "--out", WORK,
    "--save-images", WORK / "external_data.tar", timeout=4 * 3600)
print(f"EXTERNAL KERNEL DONE in {(time.time() - t_start) / 60:.1f} min", flush=True)
'''

FINETUNE = '''
t_start = time.time()
pip("timm>=1.0", "onnx", "onnxruntime", "onnxscript")
tars = glob.glob("/kaggle/input/**/__TAR__", recursive=True)
if not tars:
    subprocess.run(["find", "/kaggle/input", "-maxdepth", "4"], check=False)
    raise SystemExit("leafscan_data.tar not found: attach the output of leafscan-build-dataset")
print("extracting", tars[0], flush=True)
subprocess.run(["tar", "-xf", tars[0], "-C", "/tmp"], check=True)
DATA, RUN = "/tmp/__DATADIR__", WORK / "run"
subprocess.run(["nvidia-smi"], check=False)

run(sys.executable, "-u", "train.py", "--data", DATA, "--out", RUN,
    "--arch", "plantclef_vitb14", "--checkpoint", PLANTCLEF, "--img", __IMG__,
    "--epochs", __EPOCHS__, "--batch", __BATCH__, "--lr", __LR__, "--wd", 0.05, "--warmup", 1,
    "--layer-decay", 0.75, "--freeze-blocks", 4, "--ema", 0.9995, "--mixup", 0.8, "--cutmix", 1.0, "--mix-prob", 0.5,
    "--smoothing", 0.1, "--degrade", 0.5, "--field-boost", 1.5, "--workers", 4, *__EXTRA__,
    timeout=__TRAIN_TIMEOUT__)   # hard stop: a hang can never eat the GPU quota again
run(sys.executable, "-u", "export.py", "--data", DATA, "--run", RUN, "--out", WORK / "api_model",
    timeout=3600)
shutil.make_archive(str(WORK / "leafscan_api_model"), "zip", WORK / "api_model")
(RUN / "last.pt").unlink(missing_ok=True)   # optimizer state is not needed in the outputs
print(f"FINETUNE DONE in {(time.time() - t_start) / 60:.1f} min", flush=True)
'''


def kernel(folder: Path, slug: str, title: str, body: str, files: list[str], gpu: bool, sources=(), datasets=(),
           competitions=()):
    folder.mkdir(parents=True, exist_ok=True)
    embedded = {f: (HERE / f).read_text(encoding="utf-8") for f in files}
    script = PRELUDE.replace("__FILES__", repr(embedded)) + body
    (folder / f"{slug}.py").write_text(script, encoding="utf-8")
    meta = {
        "id": f"{USER}/{slug}", "title": title, "code_file": f"{slug}.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
        "enable_internet": True, "dataset_sources": list(datasets), "competition_sources": list(competitions),
        "kernel_sources": list(sources), "model_sources": [],
    }
    if gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print("wrote", folder)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    a = ap.parse_args()
    USER = a.user
    out = HERE / "kaggle"
    kernel(out / "build", "leafscan-build-dataset", "LeafScan Build Dataset", BUILD,
           ["taxonomy.py", "sources.py", "build_dataset.py"], gpu=False)
    kernel(out / "bench", "leafscan-gpu-benchmark", "LeafScan GPU Benchmark", BENCH,
           ["model.py", "train.py", "bench.py"], gpu=True)
    def finetune(epochs, extra, train_timeout, tar="leafscan_data.tar", datadir="leafscan_data"):
        return FETCH_PLANTCLEF + (FINETUNE.replace("__TAR__", tar).replace("__DATADIR__", datadir)
                                  .replace("__IMG__", str(a.img)).replace("__EPOCHS__", str(epochs))
                                  .replace("__BATCH__", str(a.batch)).replace("__LR__", str(a.lr))
                                  .replace("__EXTRA__", repr(extra)).replace("__TRAIN_TIMEOUT__", str(train_timeout)))

    sources = [f"{USER}/leafscan-build-dataset"]
    files = ["model.py", "train.py", "export.py"]
    kernel(out / "finetune", "leafscan-finetune-plantclef", "LeafScan Finetune PlantCLEF",
           finetune(a.epochs, [], 4 * 3600), files, gpu=True, sources=sources)
    export_pp = (EXPORT.replace('"/kaggle/input/**/leafscan_data.tar"', '"/kaggle/input/**/leafscan_data_pp.tar"')
                 .replace('"/tmp/leafscan_data"', '"/tmp/leafscan_pp"'))
    assert export_pp != EXPORT
    kernel(out / "export_pp", "leafscan-export-pp", "LeafScan Export PP", export_pp, ["model.py", "train.py", "export.py"],
           gpu=True, sources=[f"{USER}/leafscan-preprocess-dataset"], datasets=[f"{USER}/leafscan-pp-vit-best"])
    kernel(out / "export", "leafscan-export-plantclef", "LeafScan Export PlantCLEF", EXPORT, ["model.py", "train.py", "export.py"],
           gpu=True, sources=sources, datasets=[f"{USER}/leafscan-plantclef-vit-epoch8"])
    kernel(out / "segcompare", "leafscan-seg-compare", "LeafScan Seg Compare", SEGCOMPARE,
           ["preprocess.py", "segmenter.py", "seg_compare.py"], gpu=True, sources=sources)
    # Resume an interrupted run from its last.pt (uploaded as a private dataset) and finish the epochs.
    anchor = 'DATA, RUN = "/tmp/leafscan_data", WORK / "run"\n'
    resume_body = finetune(a.epochs, [], 4 * 3600).split(FETCH_PLANTCLEF, 1)[1].replace(anchor, anchor + RESUME_INPUTS)
    kernel(out / "resume", "leafscan-resume-plantclef", "LeafScan Resume PlantCLEF", 'PLANTCLEF = ""\n' + resume_body,
           files, gpu=True, sources=sources, datasets=[f"{USER}/leafscan-plantclef-vit-resume"])
    # Leaf preprocessing (background suppression + colour normalisation) and training on it
    pp_sources = [f"{USER}/leafscan-preprocess-dataset"]
    kernel(out / "preprocess", "leafscan-preprocess-dataset", "LeafScan Preprocess Dataset", PREPROCESS,
           ["preprocess.py", "segmenter.py", "preprocess_dataset.py"], gpu=True, sources=sources)
    kernel(out / "smoke_pp", "leafscan-smoke-pp", "LeafScan Smoke PP",
           finetune(1, ["--max-steps-per-epoch", 40], 1200, "leafscan_data_pp.tar", "leafscan_pp"),
           files, gpu=True, sources=pp_sources)
    kernel(out / "finetune_pp", "leafscan-finetune-pp", "LeafScan Finetune PP",
           finetune(a.epochs, [], 4 * 3600, "leafscan_data_pp.tar", "leafscan_pp"),
           files, gpu=True, sources=pp_sources)
    anchor_pp = 'DATA, RUN = "/tmp/leafscan_pp", WORK / "run"\n'
    resume_pp = (finetune(a.epochs, [], 4 * 3600, "leafscan_data_pp.tar", "leafscan_pp")
                 .split(FETCH_PLANTCLEF, 1)[1].replace(anchor_pp, anchor_pp + RESUME_INPUTS))
    assert "resuming from" in resume_pp
    kernel(out / "resume_pp", "leafscan-resume-pp", "LeafScan Resume PP", 'PLANTCLEF = ""\n' + resume_pp,
           files, gpu=True, sources=pp_sources, datasets=[f"{USER}/leafscan-pp-vit-resume"])
    kernel(out / "external", "leafscan-external-test", "LeafScan External Test", EXTERNAL,
           ["taxonomy.py", "sources.py", "build_dataset.py", "model.py", "preprocess.py", "segmenter.py",
            "external_test.py"], gpu=True,
           sources=[f"{USER}/leafscan-build-dataset", f"{USER}/leafscan-export-plantclef", f"{USER}/leafscan-resume-pp"],
           datasets=[f"{USER}/leafscan-plantclef-vit-epoch8"], competitions=["cassava-leaf-disease-classification"])
    # 1 epoch x 40 steps on the real data, real DataLoader and real export: run this first.
    kernel(out / "smoke", "leafscan-finetune-smoke", "LeafScan Finetune Smoke", finetune(1, ["--max-steps-per-epoch", 40], 1200),
           files, gpu=True, sources=sources)
