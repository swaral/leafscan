# 🌿 LeafScan: Explainable Plant Disease Classification

End-to-end deep learning system that diagnoses **94 plant diseases and healthy states across 33 crops** from a single leaf photo, trained on **106k lab and real-world field images from 5 datasets**. It shows **where the model looked** (class activation heatmaps), reports **calibrated confidence**, flags poor photos, and returns **treatment and prevention guidance**.

Three independent modules:

| Module | Stack | What it does |
|---|---|---|
| `ml/` | PyTorch, timm, scikit-learn, ONNX, Kaggle GPUs | Multi-source dataset build, fine-tuning (PlantCLEF DINOv2 ViT-B/14; v1: MobileNetV3), calibration, ONNX export, leakage and shortcut audits |
| `api/` | FastAPI, ONNX Runtime, SQLite | REST inference service (no PyTorch at runtime), heatmaps, history, OpenAPI docs |
| `frontend/` | React 18 + Vite | Diagnose / batch scan / history / model dashboard / disease library |

> Upgraded from a 3-class potato classifier (custom CNN, TensorFlow, single endpoint) to a 95-class, explainable, calibrated, tested and containerised system.

---

## Results: current model (v3)

**DINOv2 ViT-B/14 (with registers), pretrained by Pl@ntNet on PlantCLEF 2024 (about 1.4M plant photos), fine-tuned on 5 combined datasets, then further on real field photos from India, Tanzania and Ecuador.** Evaluated on **10,598 held-out test images** (never used for training, epoch selection or calibration), using the same preprocessing as the API.

| Test set | Images | Accuracy | Macro-F1 |
|---|---|---|---|
| **All** | 10,598 | **95.67 %** | 0.910 |
| Lab photos (PlantVillage) | 5,542 | 99.75 % | 0.997 |
| **Field / real-world photos** | 5,056 | **91.20 %** | 0.853 |
| └ Plant Pathology 2021 (apple orchards) | 1,567 | 98.47 % | 0.985 |
| └ CCMT (Ghana farms) | 1,600 | 95.37 % | 0.954 |
| └ PlantWild (in-the-wild web photos) | 1,645 | 82.37 % | 0.811 |
| └ PlantDoc (web photos) | 244 | 76.64 % | 0.789 |

**Outside the training sources** (3,804 field photos from 5 independent datasets, none used for training or validation): **94.6 %** exact disease (previous model 73.4 %). Two of these datasets (potato from Ethiopia, beans from Uganda) were never seen at all: 98.1 % and 94.8 %. The others are held-out photos from datasets partly added to training, e.g. Indian potato went from 44 % to 90 %. Details in [`results/README.md`](results/README.md).

Top-5 accuracy 99.41 %, calibration error (ECE) 1.24 % after temperature scaling (T = 0.75). The "uncertain" cut-off is fitted on field validation photos (`api/models/thresholds.json`). The model is 343 MB (ONNX) and takes about 240–335 ms per image on CPU.

The weakest classes are small, single-source PlantWild classes (for example apple mosaic virus, F1 0.57, and bean halo blight, F1 0.58), each with only about 100–250 web images.

## Results: version 1 (PlantVillage only, MobileNetV3)

Evaluated **end to end through the exported ONNX model and the API's own preprocessing code** on 5,544 test images that were never used for training, early stopping or calibration. The model is kept in `api/models_v1_mobilenet/` (point `MODEL_DIR` at it to switch).

<!-- METRICS:START -->
| Metric | Value |
|---|---|
| **Accuracy** | **99.60 %** (22 errors out of 5,544) |
| Macro F1 (class-balanced) | 0.994 |
| Weighted F1 | 0.996 |
| Top-3 / Top-5 accuracy | 100 % / 100 % |
| Expected calibration error | 0.20 % (temperature scaling, T = 0.48; validation ECE 14.5 % → 0.17 %) |
| CPU latency, 1 image | 4.3 ms p50 / 7.6 ms p95 (ONNX Runtime, 4-thread laptop CPU) |
| Model size | 12 MB (ONNX) |

Hardest classes: the two corn leaf blights, Gray Leaf Spot (F1 0.906) and Northern Leaf Blight (0.948), whose lesions look alike. They account for 10 of the 22 test errors. Next is Spider Mites vs Target Spot on tomato.

| | Original project | LeafScan |
|---|---|---|
| Classes | 3 (potato only) | **39** (14 crops + background) |
| Training images | ~2,150 | **44,360** |
| Model | custom CNN, TensorFlow | MobileNetV3 transfer learning → ONNX |
| Explainability | none | class activation heatmaps |
| Confidence | raw softmax | temperature-calibrated + uncertainty status |
| API | 1 endpoint | 9 endpoints, batch, history, knowledge base, tests |
| Deployment | manual | Docker, single container |
<!-- METRICS:END -->

The **Model** page in the app shows the full breakdown for whichever model is installed: training curves, per-class F1, the confusion matrix, the most-confused pairs and accuracy per data source.

---

## Does it learn the disease or the dataset? (shortcut audit)

PlantVillage photos were taken in lab sessions, so each class tends to have its own background, lighting and camera setup. A model can reach high accuracy by recognising the *session* instead of the *disease*. `ml/check_shortcuts.py` tests this on 5,259 test leaves, segmented with a colour mask refined by GrabCut. `ml/check_leakage.py` checks for duplicate images across splits.

| Test | Result | Chance / majority class | Reading |
|---|---|---|---|
| **Corners only, no deep learning** (4 × 8×8 px patches → gradient boosting) | **62.4 %** | 10.1 % | The **dataset** has strong background signatures |
| CNN on **background only** (leaf erased) | 8.0 % (69 % "no leaf") | 10.5 % | The **model** barely uses them |
| CNN on **swapped backgrounds** (leaf from crop A on background from crop B) | follows leaf 86.4 %, follows background 1.0 % | – | Predictions follow the leaf |
| CNN on **leaf only** (neutral grey background) | 92.0 % (crop 96.8 %) | – | Most signal comes from the leaf; ~8 pts depend on familiar context |
| Heatmap mass on leaf | 75 % | leaf is 44 % of image | Attention concentrates on the leaf |
| Duplicate images across splits | 4 test, 2 val (exact copies) | – | Removing all near-matches changes test accuracy by 0.02 % |

**Takeaway:** the dataset itself is biased, since 62 % of labels can be guessed from four corner patches. The trained CNN mostly ignores this and relies on the leaf. Its accuracy still drops from 99.6 % to roughly 86 to 92 % when the photo context changes, which is a more realistic expectation for photos taken outside the lab.

---

## Version 2: how it was built

1. **Dataset** (`ml/scratch/build_dataset.py`, Kaggle CPU notebook, 32 min):
   - PlantVillage, PlantDoc, PlantWild, CCMT and Plant Pathology 2021, mapped to one label list (`taxonomy.py`).
   - Exact duplicates removed (651).
   - Near-duplicates grouped across datasets (130 cross-dataset groups) and kept within one split, so copies can't leak from training into test.
   - Final set: **106,036 images in 95 classes**, split train 84,837 / val 10,601 / test 10,598.
2. **Fine-tuning** (`train.py`, Kaggle 2× T4, about 13 min per epoch, 10 epochs):
   - PlantCLEF 2024 checkpoint (Zenodo 10848263, CC BY 4.0), at 224 px.
   - Patch embedding and first 4 blocks frozen; layer-wise learning-rate decay 0.75; AdamW with warmup and cosine schedule.
   - Mixup/CutMix, label smoothing, EMA weights.
   - **Acquisition-degradation augmentation** (blur, rescaling, noise, JPEG re-compression), so the model can't identify a dataset from its camera fingerprints.
   - Class-balanced sampling with field photos weighted 1.5×.
   - Best epoch chosen by validation macro-F1 (epoch 8; epochs 9–10 were within noise).
3. **Export** (`export.py`): temperature scaling on validation, then ONNX export (matches PyTorch to within 2e-6), then test metrics per data source.
4. **Explainability:** the classifier reads the average of the patch tokens, so the per-patch class scores form a 16×16 heatmap. It is centred on the mean so it shows which regions pushed the decision above average.

The Kaggle notebooks are generated by `ml/scratch/make_kaggle.py`: build, smoke, finetune, resume and export. Always run the smoke notebook first. `ml/scratch/` also contains **LeafResNet**, a CNN of our own design for training from scratch, as an optional baseline.

---

## Features

**Model / ML**
- Transfer learning from ImageNet **MobileNetV3-Large**: fine-tunes the last inverted-residual blocks plus a new linear head
- **Activation caching**: the frozen stem is run once per image (clean + augmented view), which makes real fine-tuning practical on a laptop CPU (~40 min instead of many hours)
- Field-photo augmentation: random resized crop, flips, rotation, colour jitter, blur, plus feature-level flips and 90° rotations
- Class-imbalance handling (152 to 5,507 images per class): sqrt-inverse-frequency **class-weighted loss** + **label smoothing**
- AdamW + one-cycle LR, **early stopping on validation macro-F1**
- **Temperature scaling** so "90% confident" really means ~90% correct (ECE reported)
- Stratified 80/10/10 split with a fixed seed; metrics: accuracy, top-k, macro/weighted F1, per-class P/R/F1, confusion matrix, ECE, latency

**API**
- `POST /api/predict`: top-5 predictions, calibrated confidence, status (`healthy` / `diseased` / `uncertain` / `no_leaf`), normalised entropy, **CAM heatmap**, image-quality report (blur / exposure / resolution), disease info
- Optional **test-time augmentation** (`?tta=true`, averages 4 views)
- `POST /api/predict/batch`: up to 20 images per request with per-file error handling
- Prediction **history** in SQLite with thumbnails, filters and aggregate stats
- Disease knowledge base: pathogen, type, severity, symptoms, treatment, prevention
- Upload size limit, content validation, EXIF orientation fix, CORS allow-list, gzip
- Runs on **ONNX Runtime** (12 MB model, small Docker image, no PyTorch needed)
- Serves the built React app from the same origin, so one container covers everything
- Pytest suite

**Frontend**
- Drag & drop, **camera capture** (full resolution, front/back switch) and sample images
- Heatmap viewer: original / overlay / heatmap modes with an opacity slider
- Top-5 probability chart, quality warnings, treatment cards, **printable report**
- **Batch scan** with CSV export
- **History** dashboard with stats and filters
- **Model dashboard**: training curves, per-class F1, confusion matrix, dataset distribution
- **Disease library** with search and crop filter
- Light / dark mode, responsive down to phone width

---

## How the heatmap works

The classifier head is global-average-pooling → linear: over the CNN feature map in v1, over the ViT's patch tokens in v2. So the logit is exactly the *average* of the per-location class scores `s_c(x, y) = Σ_k w_{c,k} · F_k(x, y)`. The API shows

`map_c(x, y) = ReLU( s_c(x, y) − mean(s_c) )`

which marks the regions that pushed the decision above the average location. No gradients are needed. The ONNX graph outputs the logits and the feature grid (960×7×7 for v1, 768×16×16 for v2), and the API combines it with the head weights (`cam_head.npz`) in NumPy, so the serving stack stays torch-free. Mean-centring matters for ViTs: every patch token also carries global context, so raw scores are high everywhere.

---

## Quick start

### Option A: Docker (one container)

```bash
docker compose up --build
# open http://localhost:8000   (API docs at http://localhost:8000/docs)
```

### Option B: local dev

```bash
# API
cd api
pip install -r requirements-dev.txt
python run.py                    # http://127.0.0.1:8000/docs

# Frontend (new terminal)
cd frontend
npm install
npm run dev                      # http://localhost:5173 (proxies /api to :8000)
```

The trained model is already in `api/models/`, so you only need to retrain if you want to.

### Retrain the model

```bash
pip install -r ml/requirements.txt
# download PlantVillage (Mendeley Data, doi:10.17632/tywbtsjrjv.1, ~870 MB) to data_raw/plantvillage.zip
curl -L -o data_raw/plantvillage.zip "https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded"
python ml/prepare_data.py        # extract + stratified split manifest
python ml/train.py               # cache, fine-tune, calibrate -> ml/artifacts/leafnet.pt
python ml/export_onnx.py         # -> api/models/leafnet.onnx (+ parity check vs PyTorch)
python ml/evaluate.py            # -> api/models/metrics.json
```

### Tests

```bash
cd api && pytest -q
```

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Liveness + model loaded |
| GET | `/api/model` | Model card, test metrics, training history, dataset stats |
| POST | `/api/predict?tta=&save=&heatmap=` | Diagnose one image (multipart `file`) |
| POST | `/api/predict/batch` | Diagnose up to 20 images (multipart `files`) |
| GET | `/api/diseases?crop=` | Knowledge base |
| GET | `/api/diseases/{key}` | One disease |
| GET | `/api/history?limit=&offset=&status=` | Saved predictions |
| GET | `/api/history/stats` | Aggregates |
| DELETE | `/api/history/{id}` · `/api/history` | Delete one / all |

Example:

```bash
curl -F "file=@leaf.jpg" "http://localhost:8000/api/predict?tta=true"
```

```json
{
  "status": "diseased",
  "confidence": 0.987,
  "disease": { "key": "Tomato___Early_blight", "crop": "Tomato", "name": "Early Blight", "type": "Fungal", "...": "..." },
  "top_k": [{ "key": "Tomato___Early_blight", "probability": 0.987 }, "..."],
  "quality": { "brightness": 112.4, "sharpness": 341.2, "warnings": [] },
  "heatmap": "data:image/png;base64,...",
  "inference_ms": 48.3
}
```

Configuration (environment variables): `MODEL_DIR`, `DB_PATH`, `CORS_ORIGINS`, `MAX_UPLOAD_MB`, `MAX_BATCH`, `ONNX_THREADS`, `FRONTEND_DIST`, `PORT`, `WORKERS`.

---

## Project structure

```
plant-disease-ai/
├── ml/
│   ├── common.py          # model (LeafNet), transforms, paths
│   ├── prepare_data.py    # extract + stratified split
│   ├── train.py           # cache → fine-tune → temperature scaling
│   ├── export_onnx.py     # ONNX export + parity check
│   └── evaluate.py        # end-to-end test metrics
├── api/
│   ├── app/
│   │   ├── main.py        # FastAPI routes
│   │   ├── inference.py   # ONNX Runtime, preprocessing, TTA, CAM, quality checks
│   │   ├── db.py          # SQLite history
│   │   ├── schemas.py     # Pydantic models
│   │   ├── config.py
│   │   └── data/disease_info.json
│   ├── models/            # leafnet.onnx, cam_head.npz, labels.json, metrics.json, ...
│   └── tests/
├── frontend/
│   └── src/ (pages/, components/)
├── Dockerfile
└── docker-compose.yml
```

---

## Limitations (honest notes)

- Real-world accuracy (91 % on field photos) is lower than lab accuracy (99.7 %). The hardest sources are web photos (PlantWild 82 %, PlantDoc 76 %).
- 74 of the 108 possible classes come from a single dataset, so for those classes dataset and class can't be fully separated. Per-source metrics are reported for this reason.
- CCMT is capped at 1,000 images per class by Mendeley's API. Ambiguous labels (e.g. CCMT tomato "leaf blight") were dropped, and 13 classes with fewer than 100 images were removed.
- v2 is a ViT-B: 343 MB and about 0.3 s per image on CPU, versus 12 MB and 4 ms for v1. Int8 quantisation or distillation into a small CNN would close that gap.
- Treatment guidance is general. Always follow local extension advice and product labels.

## Dataset & credits

- PlantVillage: Hughes & Salathé (2015), *arXiv:1511.08060*; Mendeley release by Geetharamani & Arun Pandian (2019), doi:10.17632/tywbtsjrjv.1 (CC BY 4.0)
- PlantDoc: Singh et al. (2020), [github.com/pratikkayal/PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset) (CC BY 4.0)
- PlantWild v1: Wei et al. (ACM MM 2024), [huggingface.co/datasets/uqtwei2/PlantWild](https://huggingface.co/datasets/uqtwei2/PlantWild) (**CC BY-NC-ND 4.0, non-commercial use only**)
- CCMT: Mensah et al. (2023), *Data in Brief*, doi:10.17632/bwh3zbpkpv.1 (CC BY 4.0)
- Plant Pathology 2021 (FGVC8): Thapa et al., via [timm/plant-pathology-2021](https://huggingface.co/datasets/timm/plant-pathology-2021)
- Pretrained model: PlantCLEF 2024 DINOv2 ViT-B/14 by Pl@ntNet, Zenodo record [10848263](https://zenodo.org/records/10848263) (CC BY 4.0)
- Inspired by [modakverma-dev/plant-disease-classification](https://github.com/modakverma-dev/plant-disease-classification)
