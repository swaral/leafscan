"""Generate LeafScan_train_from_scratch.ipynb with the current code embedded.

    python ml/scratch/make_notebook.py
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
FILES = ["taxonomy.py", "sources.py", "build_dataset.py", "model.py", "train.py", "export.py"]

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md("""# LeafScan: train a plant-disease CNN **from scratch**

This notebook:
1. builds a **combined dataset** from 5 sources (PlantVillage, PlantDoc, PlantWild, CCMT, Plant Pathology 2021), with ~130k images, ~90 classes, cross-dataset de-duplication and a group-aware split,
2. trains **LeafResNet** (our own CNN, random initialisation) on the GPU,
3. trains a **pretrained MobileNetV3** baseline on the same data for comparison,
4. calibrates and exports both to ONNX in the format the LeafScan API loads, with test metrics per data source.

**Runtime → Change runtime type → GPU (T4).** On Kaggle, enable *Internet* and a GPU accelerator.

Everything is saved to Google Drive (`MyDrive/leafscan`) on Colab, or `/kaggle/working/leafscan` on Kaggle. If the session disconnects, **just run all cells again**: the dataset is restored from the saved archive and training resumes from the last epoch.

| Step | Time (T4) |
|---|---|
| Build dataset (first run only) | ~45–75 min |
| LeafResNet from scratch, 50 epochs | ~3–4 h |
| MobileNetV3 baseline, 12 epochs | ~40 min |
| Export + test evaluation | ~10 min |
"""),
    md("## 1. Setup"),
    code("""import os, sys, json, subprocess, shutil, pathlib
IN_COLAB = 'google.colab' in sys.modules
IN_KAGGLE = os.path.exists('/kaggle/working')
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    PERSIST, WORK = '/content/drive/MyDrive/leafscan', '/content'
elif IN_KAGGLE:
    PERSIST, WORK = '/kaggle/working/leafscan', '/tmp'
else:
    PERSIST, WORK = os.path.abspath('leafscan_persist'), os.path.abspath('.')
DATA = f'{WORK}/leafscan_data'
CODE = f'{WORK}/leafscan_code'
os.makedirs(PERSIST, exist_ok=True); os.makedirs(CODE, exist_ok=True)
os.chdir(WORK)
print('persist ->', PERSIST, '| data ->', DATA)
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip -q install onnx onnxruntime onnxscript pyarrow"""),
    md("## 2. Code\n(Generated from `ml/scratch/` in the repo. Run these cells as-is.)"),
]
for f in FILES:
    cells.append(code(f"%%writefile leafscan_code/{f}\n" + (HERE / f).read_text(encoding="utf-8")))

cells += [
    md("""## 3. Build (or restore) the combined dataset
First run downloads ~20 GB in total (Plant Pathology parquet shards are streamed and deleted one at a time),
resizes everything to a short side of 256 px and writes a ~3–4 GB archive to Drive. Later runs restore that archive in ~2 minutes."""),
    code("""ARCHIVE = f'{PERSIST}/leafscan_data.tar'
if os.path.exists(f'{DATA}/manifest.csv'):
    print('dataset already present')
elif os.path.exists(ARCHIVE):
    print('restoring dataset from', ARCHIVE)
    !tar -xf "$ARCHIVE" -C "$WORK"
else:
    !cd leafscan_code && python -u build_dataset.py --out "$DATA" --workers 32
    print('archiving to', ARCHIVE)
    !tar --exclude='raw' --exclude='records_*' -cf "$ARCHIVE" -C "$WORK" leafscan_data
stats = json.load(open(f'{DATA}/stats.json'))
print(json.dumps({k: v for k, v in stats.items() if k not in ('per_class', 'label_conflict_examples')}, indent=1))"""),
    md("## 4. Train LeafResNet from scratch\nRe-run this cell after a disconnect; it resumes from `last.pt`."),
    code("""RUN_SCRATCH = f'{PERSIST}/runs/leafresnet_s'
!cd leafscan_code && python -u train.py --data "$DATA" --out "$RUN_SCRATCH" --arch leafresnet_s --epochs 50 --batch 128 --lr 2e-3 --workers 4"""),
    md("## 5. Transfer-learning baseline (pretrained MobileNetV3, fine-tuned)"),
    code("""RUN_PRE = f'{PERSIST}/runs/mobilenet_v3_pretrained'
!cd leafscan_code && python -u train.py --data "$DATA" --out "$RUN_PRE" --arch mobilenet_v3_pretrained --epochs 12 --batch 128 --lr 1e-3 --wd 0.02 --warmup 1 --workers 4"""),
    md("## 6. Calibrate, export to ONNX and evaluate on the test set"),
    code("""for run in (RUN_SCRATCH, RUN_PRE):
    if os.path.exists(f'{run}/best.pt'):
        !cd leafscan_code && python -u export.py --data "$DATA" --run "$run" """),
    code("""import pandas as pd
rows = []
for name, run in (('LeafResNet (from scratch)', RUN_SCRATCH), ('MobileNetV3 (pretrained)', RUN_PRE)):
    p = f'{run}/api_model/metrics.json'
    if not os.path.exists(p):
        continue
    m = json.load(open(p))
    h = json.load(open(f'{run}/api_model/train_history.json'))
    row = {'model': name, 'params (M)': h['config']['params_millions'], 'test acc': m['accuracy'],
           'macro-F1': m['macro_f1'], 'top-5': m['top5_accuracy'], 'ECE': m['ece']}
    for d, dm in m['per_domain'].items():
        row[f'{d} acc'] = dm['accuracy']
    for s, sm in m['per_source'].items():
        row[f'{s} acc'] = sm['accuracy']
    rows.append(row)
pd.DataFrame(rows).set_index('model').T.round(4)"""),
    md("""## 7. Download the model for the LeafScan app
Unzip into `plant-disease-ai/api/models/` (replacing the current files) and restart the API."""),
    code("""for run in (RUN_SCRATCH, RUN_PRE):
    if os.path.exists(f'{run}/api_model/leafnet.onnx'):
        z = shutil.make_archive(f'{run}/leafscan_api_model', 'zip', f'{run}/api_model')
        print('saved', z)
if IN_COLAB and os.path.exists(f'{RUN_SCRATCH}/leafscan_api_model.zip'):
    from google.colab import files
    files.download(f'{RUN_SCRATCH}/leafscan_api_model.zip')"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
})
out = HERE / "LeafScan_train_from_scratch.ipynb"
nbf.write(nb, out)
print("wrote", out)
