# Training from scratch on 5 datasets (GPU)

This folder trains **LeafResNet**, a CNN designed for this project and trained from **random initialisation** (no pretrained weights), on a combined lab + real-world dataset. It also trains a pretrained MobileNetV3 on the same data as a comparison.

**Run it:** upload [`LeafScan_train_from_scratch.ipynb`](LeafScan_train_from_scratch.ipynb) to Google Colab (Runtime → GPU) or Kaggle (GPU + Internet on), then *Run all*. Re-running after a disconnect resumes where it stopped.

## Data

| Source | Kind | Raw images | Licence | Notes |
|---|---|---|---|---|
| [PlantVillage](https://data.mendeley.com/datasets/tywbtsjrjv/1) | lab | 55,448 | CC BY 4.0 | 38 classes + background |
| [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) | web / field | ~2,600 | CC BY 4.0 | 27 classes, same crops as PlantVillage |
| [PlantWild v1](https://huggingface.co/datasets/uqtwei2/PlantWild) | in-the-wild | 18,542 | CC BY-NC-ND 4.0 | 89 classes; non-commercial use only |
| [CCMT](https://data.mendeley.com/datasets/bwh3zbpkpv/1) | field (Ghana) | 24,881 | CC BY 4.0 | **raw** images only; Mendeley's API caps each folder at 1,000 files |
| [Plant Pathology 2021](https://huggingface.co/datasets/timm/plant-pathology-2021) | field (apple) | 18,632 | Apache-2.0 | single-label images only |

All labels map to one PlantVillage-style label set in [`taxonomy.py`](taxonomy.py). Labels are merged only when they name the same disease, and ambiguous labels are dropped. Examples: CCMT tomato "leaf blight" could be early or late blight, so it's dropped. PlantWild "carrot cavity spot" is a root disease, so it's dropped. Plant Pathology's "frog-eye leaf spot" is caused by the same fungus as apple black rot, so they merge. Merges that are reasonable but not certain are listed in `UNCERTAIN`.

`build_dataset.py` then:
- resizes everything to a short side of 256 px,
- removes exact duplicates (MD5),
- groups **near-duplicates across all datasets** (perceptual hash, any flip or rotation, distance ≤ 1); PlantDoc and PlantWild were both scraped from web image search,
- drops duplicate groups whose labels disagree,
- drops classes with fewer than 100 images,
- splits 80/10/10 **by duplicate group**, stratified by class × source, so a copy of a test image can never be in training.

## Model and training

- **LeafResNet-S** (12.8M params): ResNet-C deep stem, ResNet-D average-pool shortcuts, squeeze-excite attention, stochastic depth, zero-init residual BN, GAP + linear head (keeps heatmaps exact).
- **Pipeline:** JPEG decoding (nvJPEG) and augmentation run batched on the GPU, so Colab's 2 CPU cores aren't a bottleneck.
- **Augmentation:** random-resized-crop (25–100%), flips, 90° rotations, brightness/contrast/saturation jitter, random erasing, mixup/cutmix.
- **Optimisation:** AdamW with warmup + cosine LR, label smoothing 0.1, weight EMA, AMP, gradient clipping.
- **Sampling:** class-balanced (1/√n), with real-world photos sampled 1.5× more, so the 55k lab images don't dominate.
- **Model selection:** best epoch by validation macro-F1. Temperature scaling on validation, then test metrics overall, **per source and lab vs field**.

## Using the result in the app

The notebook produces `leafscan_api_model.zip`. Unzip it into `api/models/` (replacing the current files) and restart the API. The knowledge base already covers all 108 possible classes.

## Local smoke test (CPU, a few minutes)

```bash
cd ml/scratch
python build_dataset.py --out ./smoke --smoke --max-per-class 12 --min-images 20 --plantvillage-zip ../../data_raw/plantvillage.zip
python train.py --data ./smoke --out ./smoke_run --epochs 2 --batch 16 --max-steps-per-epoch 4 --workers 0
python export.py --data ./smoke --run ./smoke_run
```

After editing any `.py` file here, run `python make_notebook.py` to regenerate the notebook.
