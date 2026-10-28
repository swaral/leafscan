# LeafScan: results, problems found and changes made

This folder collects every result of the project, from the first model (v1) to the
latest experiments.

| Folder | Contents |
|---|---|
| `v1_mobilenet/` | v1 model metrics and training history (PlantVillage only) |
| `v3_field/` | **Current model in the app** (v2 + field photos): metrics, thresholds, external comparison, log |
| `v2_vit/` | Previous app model: metrics, training history, class list |
| `v2_vit_preprocessed/` | Preprocessed-model results, transcribed from the Kaggle logs (its files are on Kaggle) |
| `audits/` | Leakage check and shortcut (background) audit, with images |
| `segmentation/` | Background-removal model comparison, safety-gate test, heatmap fix |
| `external_test/` | Test on 5 unseen field datasets: JSON results and sample predictions; `v2_tta_ensemble/` adds TTA and ensemble |
| `logs/` | Raw logs of the Kaggle runs |

---

## 1. Where we started

The original project ([modakverma-dev/plant-disease-classification](https://github.com/modakverma-dev/plant-disease-classification)):
- **Data and model:** 3 classes (potato only), about 2,150 images, a small custom CNN in TensorFlow.
- **Backend:** one API endpoint.
- **Missing:** heatmaps, calibration, tests and deployment setup.

## 2. What was wrong, found step by step, and what we changed

### A. Problems in the data and the model
| # | Problem found | Evidence | Change made |
|---|---|---|---|
| 1 | Only 3 classes, potato only | original repo | v1: all of PlantVillage, **39 classes, 14 crops**; v2: 5 datasets, **95 classes, 33 crops** |
| 2 | **Lab-only photos (PlantVillage)**: the background reveals the class | A classifier using only the **4 image corners** reached **62.4%** (chance level: 10.1%) | Added 4 real-world field datasets (v2); acquisition-degradation augmentation |
| 3 | v1 looked 99.6% accurate but relied partly on lab context | Leaf on grey background: 92.0%; leaf on another crop's background: 86.4% | Measured it honestly; moved to field data (v2) |
| 4 | Duplicate images across train and test | v1: 4 test and 2 validation images were exact copies; v2 sources: **651 exact duplicates**, **130 duplicate groups spanning datasets** | Exact-duplicate removal plus **duplicate-group-aware split** (a copy can never sit on both sides) |
| 5 | Datasets used different, conflicting labels | e.g. "leaf blight" could mean early or late blight | One unified label list (`ml/scratch/taxonomy.py`); ambiguous labels **dropped, not guessed**; 264 conflicting images and 13 tiny classes removed |
| 6 | Every dataset has a camera/compression "fingerprint" | 15 simple image statistics identify the source dataset **74.8%** of the time (chance: 32.5%) | Training-time degradation augmentation (blur, rescale, noise, JPEG re-compression) |
| 7 | 74 of 108 classes come from a single dataset | taxonomy analysis | Accuracy reported **per dataset**; external test added |
| 8 | Model confidence was not trustworthy | v1 raw calibration error 14.5% | **Temperature scaling** → 0.2% (v1), 1.4% (v2) |

### B. Engineering bugs found and fixed
| # | Bug | Effect | Fix |
|---|---|---|---|
| 1 | First training run lost when the session ended | No checkpoint saved | Runs detached; checkpoint saved every epoch |
| 2 | Kaggle data loader passed tensors through file descriptors | Workers crashed silently; run hung for **12 h** | Workers return plain arrays; 10-min loader timeout; hard time limits; **smoke-test notebook before every long run** |
| 3 | GPU JPEG decoder (nvJPEG) **leaks GPU memory** | Training crashed at **epoch 9 twice** | JPEG decoding moved to CPU workers; no crash since |
| 4 | Mendeley blocks Python downloads | CCMT/PlantVillage downloads failed | Downloads via `curl` |
| 5 | Kaggle can't use output from a failed run | Fine-tune couldn't find the dataset | Rebuild until success; resume via private datasets |
| 6 | ViT heatmaps were bright everywhere | Heatmap useless with the new model | Heatmap = per-patch score **minus the mean** → shows lesions (`segmentation/heatmap_fix_vit_vs_cnn.jpg`) |
| 7 | Background removal **cut into leaves** | Up to 56% of leaf pixels lost | Hole filling, edge expansion and a **safety gate** that keeps the photo whole when unsure → 4.9% loss |
| 8 | Light **halo** around segmented leaves | Itself a "this is a lab photo" signature | Edge fades into the blurred background |
| 9 | App text hardcoded for the old model | Wrong model name and counts in the UI | Read from the installed model's metadata |
| 10 | Crop filter listed 0% classes | Confusing top-5 list | Ruled-out classes hidden |

## 3. Results

### 3.1 Internal test (held-out split of the training sources)
| Model | Test images | Accuracy | Macro-F1 | Lab photos | **Field photos** | Calibration error |
|---|---|---|---|---|---|---|
| v1 MobileNetV3 (PlantVillage only, 39 classes) | 5,544 | 99.60% | 0.994 | 99.60% | n/a | 0.20% |
| v2 ViT (95 classes) | 10,598 | 95.63% | 0.910 | 99.73% | 91.14% | 1.39% |
| **v3 ViT + field photos (current app model)** | 10,598 | **95.67%** | **0.910** | **99.75%** | **91.20%** | **1.24%** |
| v2 ViT + background removal and colour normalisation | 10,598 | 95.39% | 0.905 | 99.49% | 90.88% | 1.51% |

v2 per dataset:

| Dataset | v2 ViT | v2 preprocessed |
|---|---|---|
| PlantVillage (lab) | **99.73%** | 99.49% |
| Plant Pathology 2021 | 98.60% | 98.60% |
| CCMT (Ghana) | 95.25% | **95.56%** |
| PlantWild (web) | **82.31%** | 81.03% |
| PlantDoc (web) | 75.82% | **77.05%** |

v1's 99.6% is on lab photos and a 39-class task. v2 solves a much harder 95-class task that includes real-world photos.

### 3.2 External test: 5 datasets **neither model has ever seen** (9,657 field photos)
Only 1 image duplicated a training image and was removed.

| | **v2 ViT (current)** | v2 preprocessed |
|---|---|---|
| **Exact disease accuracy** | **66.5%** | **68.1%** |
| Correct crop | 83.7% | 87.1% |
| Accuracy if the crop is known (new crop picker in the app) | **77.5%** | 76.5% |
| Top-5 | 92.8% | 93.1% |
| Average confidence | 82% (overconfident) | 82% |

| External dataset | v2 ViT | v2 preprocessed |
|---|---|---|
| Potato, Ethiopia (430) | 97.0% | 97.7% |
| Bean, Uganda (862) | **93.5%** | 89.0% |
| Maize, Tanzania (3,000) | **85.7%** | 80.0% |
| Coffee, Ecuador (865) | 77.9% | **87.6%** |
| Potato, India (4,500) | 43.4% | **49.6%** |

**For v2, the honest real-world accuracy was about 66–68%, or 77.5% when the user selected the crop (see 3.6 for v3).** The weakest case is late blight on dried leaves in the Indian potato photos.

### 3.3 Background removal and colour normalisation experiment
- **Segmenter:** BiRefNet-lite was the best of 4 models. It takes 45 ms per image on GPU and 5.4 s on CPU.
- **Safety gate:** cut leaf-pixel loss from 56% to 4.9%, but kept most field photos whole. Share actually segmented: PlantVillage 89%, CCMT 47%, PlantDoc and PlantWild 25%, Plant Pathology 1%.
- **Verdict:** internally slightly worse (−0.24%), externally +1.6% overall but mixed (coffee +9.7%, maize −5.7%). It costs about 6 s per photo in the app. **The current app model remains the plain v2 ViT.**

### 3.4 Training speed (Kaggle 2× T4, ViT-B at 224 px)
| Setup | Images/s |
|---|---|
| Full fine-tune | 101.6 |
| **First 4 blocks frozen (used)** | **123.9** |
| First 6 blocks frozen | 131.9 |
| 336 px, full | 42.1 |

### 3.5 TTA and ensemble on the external test (v2 models, 9,657 photos)
| Setup | Exact disease | Crop known | Calibration error |
|---|---|---|---|
| v2 ViT | 66.5% | 77.5% | 0.159 |
| v2 ViT + TTA | 67.1% | 77.9% | 0.132 |
| v2 preprocessed + TTA | 69.2% | 77.2% | 0.110 |
| Ensemble of both + TTA | 69.3% | 78.4% | 0.086 |

A small gain for 8x the compute, so it was not adopted; adding field data (3.6) helped far more.

### 3.6 v3: fine-tuned further on real field photos (current app model)
**What changed:** the v2 model was trained for 4 more epochs (learning rate 5e-5, about 1 hour on 2× T4).
Field photos from 3 of the external datasets were added to training and sampled 3x as often:
- Potato India: 2,471 train / 677 val / 1,352 held out
- Maize Tanzania: 1,650 / 450 / 900
- Coffee Ecuador: 476 / 129 / 260

Near-duplicates were kept on one side of the split.
**Potato Ethiopia and Beans Uganda were left out completely** to measure true new-farm performance.

**Field thresholds:** the "uncertain" cut-off is now fitted on the 6,315 field validation photos so
that accepted diagnoses are ≥95% correct: 0.77 without a crop (93% of photos accepted), 0.53 with the
crop selected (99% accepted). Written to `api/models/thresholds.json`, read by the app.

| Test photos (none used for training or validation) | v2 (previous) | **v3 (current)** |
|---|---|---|
| Potato, Ethiopia (430), **never seen** | 97.0% | **98.1%** |
| Bean, Uganda (862), **never seen** | 93.5% | **94.8%** |
| Potato, India: held-out 30% (1,352) | 44.3% | **89.8%** |
| Maize, Tanzania: held-out 30% (900) | 85.2% | **99.1%** |
| Coffee, Ecuador: held-out 30% (260) | 78.1% | **96.9%** |
| **All 3,804 photos** | 73.4% | **94.6%** |
| Calibration error (ECE) | 0.113 | **0.012** |
| Internal test (10,598) | 95.63% | **95.67%** (no regression) |

**How to read this:** the held-out India, Tanzania and Ecuador photos come from the same regions and
cameras as the photos added to training, so they show what happens **after local field data is
collected** (India potato: 44% → 90%). The two never-seen datasets show the gain on a brand-new
farm: small (+1.1 and +1.3 points), and with no regression. The fix for a weak region is to collect
labelled field photos from it. Scope is still 9 classes in 4 crops; PlantDoc (77%) and PlantWild
(82%) remain the weakest internal sources.

## 4. Still pending
- **Cassava (Uganda) external dataset:** needs the Kaggle competition rules accepted.
- **Next improvement:** the same field-data recipe for the weakest internal sources (PlantDoc, PlantWild), and a new never-seen dataset per crop to keep testing honest.
