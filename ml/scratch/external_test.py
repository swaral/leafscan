"""External test: evaluate the trained models on independent field datasets they never saw.

Runs on Kaggle. Downloads each dataset there, maps its labels to our classes (unknown
diseases are skipped), removes any image that is a (near-)duplicate of a training image,
then evaluates every model exactly as the API runs it.

    python external_test.py --train-data /tmp/leafscan_data --models raw=<dir> pp=<dir> --out /kaggle/working

Each model dir holds best.pt + train_history.json (temperature); a model whose dir also
contains pp_config.json gets the leaf preprocessing (BiRefNet mask -> gate -> colour ->
soft background) before classification, like the API.
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import random
import shutil
import subprocess
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageOps

EXT = Path("/tmp/ext")
CAP = 1500          # max images per (dataset, class)
UA = "Mozilla/5.0"
MENDELEY = "https://data.mendeley.com/public-api/datasets"
random.seed(0)


# ---------------------------------------------------------------- download helpers
def curl(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-sSfL", "--retry", "5", "-A", UA, "-o", str(dest), url], check=True)


def curl_json(url: str):
    r = subprocess.run(["curl", "-sSfL", "-A", UA, url], capture_output=True, check=True)
    return json.loads(r.stdout)


def save_img(data: bytes, dest: Path) -> bool:
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        im.thumbnail((1600, 1600), Image.LANCZOS)   # models see <=512 px; keeps decoding fast
        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest, quality=95)
        return True
    except Exception:
        return False


def capped(items, key):
    """Keep at most CAP random items per label."""
    by = defaultdict(list)
    for it in items:
        by[key(it)].append(it)
    out = []
    for k, v in by.items():
        random.shuffle(v)
        out += v[:CAP]
    return out


# ---------------------------------------------------------------- datasets -> [(path, source_label)]
def potato_india():
    """PLDD-UP (Uttar Pradesh, India; Mendeley 3j4nfkvp2n v1): EB.zip, Healthy.zip, LB.zip."""
    files = curl_json(f"{MENDELEY}/3j4nfkvp2n/files?folder_id=root&version=1")
    out = []
    for f in files:
        label = f["filename"].rsplit(".", 1)[0]
        z = EXT / "raw" / f["filename"]
        curl(f["content_details"]["download_url"], z)
        with zipfile.ZipFile(z) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]
            random.shuffle(names)
            for n in names[:CAP]:
                dest = EXT / "potato_india" / label / Path(n).name
                if save_img(zf.read(n), dest):
                    out.append((dest, label))
        z.unlink()
    return out


def potato_ethiopia():
    """Holeta, Ethiopia (Mendeley v4w72bsts5 v2): Local.rar with class folders."""
    f = curl_json(f"{MENDELEY}/v4w72bsts5/files?folder_id=root&version=2")[0]
    rar = EXT / "raw" / f["filename"]
    curl(f["content_details"]["download_url"], rar)
    dest = EXT / "potato_ethiopia_raw"
    dest.mkdir(parents=True, exist_ok=True)
    for cmd in (["bsdtar", "-xf", str(rar), "-C", str(dest)], ["7z", "x", "-y", f"-o{dest}", str(rar)],
                ["unrar", "x", "-o+", str(rar), str(dest)]):
        try:
            if subprocess.run(cmd, capture_output=True).returncode == 0 and any(dest.rglob("*.*")):
                break
        except FileNotFoundError:
            continue
    else:
        subprocess.run(["pip", "install", "-q", "rarfile", "unrar-cffi"], check=False)
        try:
            import rarfile
            rarfile.RarFile(rar).extractall(dest)
        except Exception as e:
            print("  potato_ethiopia: cannot extract rar:", e)
            return []
    out = []
    for p in dest.rglob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            out.append((p, p.parent.name))
    return out


def hf_parquet(repo: str, name: str, keep, max_shards: int | None = None):
    """Read image/label rows from a Hugging Face parquet dataset (label names from schema)."""
    import pyarrow.parquet as pq
    tree = requests.get(f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true", timeout=60).json()
    shards = sorted(x["path"] for x in tree if x["path"].endswith(".parquet"))
    if max_shards:
        shards = shards[:max_shards]
    out, counts = [], Counter()
    for sh in shards:
        local = EXT / "raw" / Path(sh).name
        curl(f"https://huggingface.co/datasets/{repo}/resolve/main/{sh}", local)
        pf = pq.ParquetFile(local)
        meta = json.loads(pf.schema_arrow.metadata[b"huggingface"])
        feats = meta["info"]["features"]
        label_col = next(k for k, v in feats.items() if v.get("_type") == "ClassLabel")
        img_col = next(k for k, v in feats.items() if v.get("_type") == "Image")
        names = feats[label_col]["names"]
        for batch in pf.iter_batches(columns=[img_col, label_col], batch_size=64):
            for row in batch.to_pylist():
                lab = names[row[label_col]]
                if not keep(lab) or counts[lab] >= CAP:
                    continue
                dest = EXT / name / lab / f"{name}_{sum(counts.values())}.jpg"
                if save_img(row[img_col]["bytes"], dest):
                    out.append((dest, lab))
                    counts[lab] += 1
        local.unlink()
    print(f"  {name}: {dict(counts)}")
    return out


def maize_tanzania():
    return hf_parquet("Project-AgML/maize_disease_classification", "maize_tanzania",
                      keep=lambda l: "mln" not in l.lower() and "necrosis" not in l.lower(), max_shards=8)


def bean_uganda():
    return hf_parquet("AI-Lab-Makerere/beans", "bean_uganda", keep=lambda l: "angular" not in l.lower())


def coffee_ecuador():
    """RoCoLe (Mendeley c5yvn32dzg v2). The API lists at most 1000 of the 1560 photos."""
    folders = {f["name"]: f["id"] for f in curl_json(f"{MENDELEY}/c5yvn32dzg/folders/2")}
    ann = curl_json(f"{MENDELEY}/c5yvn32dzg/files?folder_id={folders['Annotations']}&version=2")
    csv_url = next(f["content_details"]["download_url"] for f in ann if f["filename"].endswith(".csv"))
    curl(csv_url, EXT / "raw" / "rocole.csv")
    labels = {}
    for r in csv.DictReader(open(EXT / "raw" / "rocole.csv", encoding="utf-8", errors="replace")):
        fname = r["Labeled Data"].rsplit("-", 1)[-1]
        try:
            labels[fname] = json.loads(r["Label"]).get("classification", "")
        except json.JSONDecodeError:
            pass
    photos = curl_json(f"{MENDELEY}/c5yvn32dzg/files?folder_id={folders['Photos']}&version=2")
    out = []

    def fetch(f):
        lab = labels.get(f["filename"], "")
        if not lab or "mite" in lab:
            return None
        dest = EXT / "coffee_ecuador" / f["filename"]
        try:
            curl(f["content_details"]["download_url"], dest)
            return dest, lab
        except subprocess.CalledProcessError:
            return None

    with ThreadPoolExecutor(16) as pool:
        out = [x for x in pool.map(fetch, photos) if x]
    return out


def cassava_uganda():
    """Kaggle 'Cassava Leaf Disease Classification' (2020), attached as competition data."""
    csvs = glob.glob("/kaggle/input/**/cassava-leaf-disease-classification/train.csv", recursive=True) or \
        glob.glob("/kaggle/input/**/train.csv", recursive=True)
    csvs = [c for c in csvs if Path(c).parent.joinpath("train_images").exists()]
    if not csvs:
        print("  cassava: competition data not attached")
        return []
    root = Path(csvs[0]).parent
    names = {0: "CBB", 1: "CBSD", 2: "CGM", 3: "CMD", 4: "Healthy"}
    rows = [(root / "train_images" / r["image_id"], names[int(r["label"])])
            for r in csv.DictReader(open(root / "train.csv")) if int(r["label"]) != 1]   # CBSD unknown to us
    return capped(rows, key=lambda x: x[1])


DATASETS = {
    "potato_india": potato_india, "potato_ethiopia": potato_ethiopia, "maize_tanzania": maize_tanzania,
    "bean_uganda": bean_uganda, "coffee_ecuador": coffee_ecuador, "cassava_uganda": cassava_uganda,
}


def map_label(dataset: str, label: str) -> str | None:
    l = label.lower().replace("_", " ")
    if dataset.startswith("potato"):
        if l in ("eb",) or "early" in l:
            return "Potato___Early_blight"
        if l in ("lb",) or "late" in l:
            return "Potato___Late_blight"
        if "health" in l:
            return "Potato___healthy"
    if dataset == "maize_tanzania":
        if "health" in l:
            return "Corn___healthy"
        if "msv" in l or "streak" in l:
            return "Corn___Maize_streak_virus"
    if dataset == "bean_uganda":
        if "rust" in l:
            return "Bean___Rust"
        if "health" in l:
            return "Bean___healthy"
    if dataset == "coffee_ecuador":
        if "health" in l:
            return "Coffee___healthy"
        if "rust" in l:
            return "Coffee___Leaf_rust"
    if dataset == "cassava_uganda":
        return {"cbb": "Cassava___Bacterial_blight", "cgm": "Cassava___Green_mite",
                "cmd": "Cassava___Mosaic_disease", "healthy": "Cassava___healthy"}.get(l)
    return None


# ---------------------------------------------------------------- de-duplication against training
def _resize256(img):
    w, h = img.size
    s = 256 / min(w, h)
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)


def _hash_job(path):
    from build_dataset import phashes
    try:
        return phashes(_resize256(Image.open(path).convert("RGB")))
    except Exception:
        return None


def near_duplicates(train_paths, ext_paths, max_dist=1):
    with ProcessPoolExecutor(4) as pool:
        th = list(pool.map(_hash_job, train_paths, chunksize=256))
        eh = list(pool.map(_hash_job, ext_paths, chunksize=64))
    buckets = [defaultdict(list), defaultdict(list)]
    canon = []
    for i, h in enumerate(th):
        if h is None:
            continue
        canon.append(h[0])
        buckets[0][h[0] >> 32].append(len(canon) - 1)
        buckets[1][h[0] & 0xFFFFFFFF].append(len(canon) - 1)
    dup = []
    for hs in eh:
        found = False
        if hs is not None:
            for v in hs:
                for half, key in ((0, v >> 32), (1, v & 0xFFFFFFFF)):
                    for j in buckets[half].get(key, ())[:500]:
                        if bin(v ^ canon[j]).count("1") <= max_dist:
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        dup.append(found)
    return dup


# ---------------------------------------------------------------- models
def resize_like_dataset(img, short_side=256, max_long=512):   # identical to api/app/preprocessing.py
    w, h = img.size
    s = short_side / min(w, h)
    nw, nh = round(w * s), round(h * s)
    if max(nw, nh) > max_long:
        img = img.resize((nw, nh), Image.BILINEAR)
        cw, ch = min(nw, max_long), min(nh, max_long)
        left, top = (nw - cw) // 2, (nh - ch) // 2
        return img.crop((left, top, left + cw, top + ch))
    return img.resize((nw, nh), Image.BILINEAR if s < 1 else Image.BICUBIC)


def load_model(d: Path, dev):
    from model import ExportWrapper, build_model
    ck = torch.load(d / "best.pt", map_location="cpu", weights_only=False)
    T = json.loads((d / "train_history.json").read_text())["temperature"]
    net = build_model(ck["arch"], len(ck["classes"]), pretrained=False, img=ck["img"])
    net.load_state_dict(ck["ema"])
    return ExportWrapper(net, T).to(dev).eval(), ck["classes"], ck["img"], (d / "pp_config.json").exists()


def _pp_job(args):
    from preprocess import leaf_preprocess
    path, mask16, img_size = args
    img = resize_like_dataset(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))
    out, info = leaf_preprocess(img, mask16.astype(np.float32))
    return np.asarray(out.resize((img_size, img_size), Image.BILINEAR), dtype=np.uint8), info["gate"]


def model_inputs(paths, img_size, leafprep: bool, dev):
    """Yield (batch uint8 HxWx3 arrays, gates) exactly as the API would feed the model."""
    seg = None
    if leafprep:
        from segmenter import BiRefNet
        seg = BiRefNet(dev, size=512)
    pool = ProcessPoolExecutor(4) if leafprep else None
    for i in range(0, len(paths), 32):
        chunk = paths[i:i + 32]
        if not leafprep:   # API: exif fix -> RGB -> direct resize to model size
            arrs = [np.asarray(ImageOps.exif_transpose(Image.open(p)).convert("RGB")
                               .resize((img_size, img_size), Image.BILINEAR), dtype=np.uint8) for p in chunk]
            yield arrs, ["raw"] * len(chunk)
        else:
            imgs = [resize_like_dataset(ImageOps.exif_transpose(Image.open(p)).convert("RGB")) for p in chunk]
            masks = seg.masks(imgs)
            res = list(pool.map(_pp_job, [(p, m.astype(np.float16), img_size) for p, m in zip(chunk, masks)]))
            yield [r[0] for r in res], [r[1] for r in res]
    if pool:
        pool.shutdown()


@torch.no_grad()
def predict(model, paths, img_size, leafprep, dev):
    """Returns (plain probs, TTA probs, gates). TTA = mean over identity, h-flip, v-flip and
    90-degree rotation (the same views as the API's TTA option)."""
    plain, tta, gates = [], [], []
    for arrs, g in model_inputs(paths, img_size, leafprep, dev):
        x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).float().div(255).to(dev)
        views = torch.cat([x, x.flip(3), x.flip(2), torch.rot90(x, 1, (2, 3))])
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            z = model(views)[0].float()
        pr = torch.softmax(z, 1).view(4, len(arrs), -1)
        plain.append(pr[0].cpu().numpy())
        tta.append(pr.mean(0).cpu().numpy())
        gates += g
    return np.concatenate(plain), np.concatenate(tta), gates


def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def crop(k):
    return k.split("___")[0]


def summarise(p, y, classes):
    pred = p.argmax(1)
    crops = np.array([crop(c) for c in classes])
    within = []
    for i in range(len(y)):
        m = crops == crops[y[i]]
        within.append(np.flatnonzero(m)[p[i, m].argmax()] == y[i])
    top5 = np.argsort(-p, 1)[:, :5]
    conf = Counter((classes[t], classes[q]) for t, q in zip(y, pred) if t != q)
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "top5": float(np.mean([y[i] in top5[i] for i in range(len(y))])),
        "crop_accuracy": float(np.mean(crops[pred] == crops[y])),
        "accuracy_given_crop": float(np.mean(within)),
        "mean_confidence": float(p.max(1).mean()),
        "ece": ece(p, y),
        "top_confusions": [[t, q, n] for (t, q), n in conf.most_common(5)],
        "selective": {f"conf>={th}": {"coverage": float((p.max(1) >= th).mean()),
                                      "accuracy": (float((pred == y)[p.max(1) >= th].mean())
                                                   if (p.max(1) >= th).any() else None)}
                      for th in (0.5, 0.7, 0.9)},
    }


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="name=dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-images", default=None, help="write the prepared external set to this tar for reuse")
    ap.add_argument("--reuse", default=None, help="tar written by --save-images: skip downloading")
    args = ap.parse_args()
    out = Path(args.out)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    # 1. download + map (or reuse a previously prepared set)
    items, skipped, reused = [], Counter(), None
    if args.reuse:
        subprocess.run(["tar", "-xf", args.reuse, "-C", str(EXT.parent)], check=True)
        reused = json.loads((EXT / "manifest.json").read_text())
        items = [{**it, "path": str(EXT / it["path"])} for it in reused["items"]]
        skipped = Counter(reused["skipped"])
        print(f"reusing {len(items)} prepared external images", flush=True)
    for name, fn in ({} if reused else DATASETS).items():
        t = time.time()
        try:
            raw = fn()
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {e}", flush=True)
            raw = []
        n_ok = 0
        for path, lab in raw:
            key = map_label(name, lab)
            if key is None:
                skipped[f"{name}:{lab}"] += 1
            else:
                items.append({"dataset": name, "path": str(path), "label": lab, "key": key})
                n_ok += 1
        print(f"[{name}] {n_ok} usable images ({(time.time() - t) / 60:.1f} min)", flush=True)
    shutil.rmtree(EXT / "raw", ignore_errors=True)

    models = {}
    for spec in args.models:
        nm, d = spec.split("=", 1)
        models[nm] = load_model(Path(d), dev)
    classes = next(iter(models.values()))[1]
    assert all(m[1] == classes for m in models.values()), "models disagree on class list"
    unknown = Counter(it["key"] for it in items if it["key"] not in classes)
    items = [it for it in items if it["key"] in classes]

    # 2. remove near-duplicates of any training image
    if reused:
        n_dup = Counter(reused["near_duplicates_removed"])
    else:
        tr = list(csv.DictReader(open(Path(args.train_data) / "manifest.csv", encoding="utf-8")))
        dup = near_duplicates([str(Path(args.train_data) / r["path"]) for r in tr], [it["path"] for it in items])
        n_dup = Counter(it["dataset"] for it, d in zip(items, dup) if d)
        items = [it for it, d in zip(items, dup) if not d]
    print(f"removed near-duplicates of training images: {dict(n_dup)}", flush=True)
    if args.save_images and not reused:
        for it in items:   # images living outside EXT (e.g. competition data) are copied in
            if not it["path"].startswith(str(EXT)):
                dst = EXT / it["dataset"] / Path(it["path"]).name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(it["path"], dst)
                it["path"] = str(dst)
        rel = [{**it, "path": str(Path(it["path"]).relative_to(EXT))} for it in items]
        (EXT / "manifest.json").write_text(json.dumps({"items": rel, "skipped": dict(skipped),
                                                       "near_duplicates_removed": dict(n_dup)}))
        subprocess.run(["tar", "-cf", args.save_images, "-C", str(EXT.parent), EXT.name], check=True)
        print(f"saved prepared external set -> {args.save_images}", flush=True)

    # 3. evaluate
    y = np.array([classes.index(it["key"]) for it in items])
    paths = [it["path"] for it in items]
    report = {"images": len(items), "skipped_labels": dict(skipped), "classes_not_in_model": dict(unknown),
              "near_duplicates_removed": dict(n_dup), "models": {}}
    datasets_ = sorted({it["dataset"] for it in items})
    keys_ = sorted(set(it["key"] for it in items))

    def full_report(p, extra=None):
        r = {"overall": summarise(p, y, classes), "per_dataset": {}, "per_class": {}, **(extra or {})}
        for d in datasets_:
            m = np.array([it["dataset"] == d for it in items])
            r["per_dataset"][d] = summarise(p[m], y[m], classes)
        for k in keys_:
            m = np.array([it["key"] == k for it in items])
            r["per_class"][k] = summarise(p[m], y[m], classes)
        return r

    preds, probs_all = {}, {}
    for nm, (model, _, img_size, leafprep) in models.items():
        t = time.time()
        p, p_tta, gates = predict(model, paths, img_size, leafprep, dev)
        preds[nm] = p
        probs_all[nm], probs_all[nm + "+tta"] = p, p_tta
        extra = {"seconds": round(time.time() - t), "leafprep": leafprep}
        if leafprep:
            extra["gate_share"] = {d: dict(Counter(g for it, g in zip(items, gates) if it["dataset"] == d))
                                   for d in datasets_}
        report["models"][nm] = full_report(p, extra)
        report["models"][nm + "+tta"] = full_report(p_tta)
    if len(models) > 1:   # ensemble = mean of the models' probabilities
        names = list(models)
        probs_all["ensemble"] = np.mean([probs_all[n] for n in names], 0)
        probs_all["ensemble+tta"] = np.mean([probs_all[n + "+tta"] for n in names], 0)
        report["models"]["ensemble"] = full_report(probs_all["ensemble"])
        report["models"]["ensemble+tta"] = full_report(probs_all["ensemble+tta"])
    for cfg, r in report["models"].items():
        o = r["overall"]
        print(f"[{cfg:22s}] acc {o['accuracy']:.4f}  crop {o['crop_accuracy']:.4f}  "
              f"given-crop {o['accuracy_given_crop']:.4f}  top5 {o['top5']:.4f}  ece {o['ece']:.3f}", flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(args.out) / "external_probs.npz", y=y,
                        **{k.replace("+", "_"): v for k, v in probs_all.items()})

    out.mkdir(parents=True, exist_ok=True)
    (out / "external_results.json").write_text(json.dumps(report, indent=2))

    # 4. sample sheet: 6 per dataset with each model's prediction
    T = 200
    by_ds = defaultdict(list)
    for i, it in enumerate(items):
        by_ds[it["dataset"]].append(i)
    pick = [i for d in sorted(by_ds) for i in random.sample(by_ds[d], min(6, len(by_ds[d])))]
    sheet = Image.new("RGB", (T * 3, T * ((len(pick) + 2) // 3)), "white")
    for n, i in enumerate(pick):
        tile = ImageOps.exif_transpose(Image.open(items[i]["path"])).convert("RGB").resize((T, T))
        dr = ImageDraw.Draw(tile)
        dr.rectangle([0, 0, T, 14 + 12 * len(preds)], fill=(0, 0, 0))
        dr.text((3, 1), f"{items[i]['dataset']}: {items[i]['key'].split('___')[1][:20]}", fill=(255, 255, 255))
        for j, (nm, p) in enumerate(preds.items()):
            k = classes[p[i].argmax()]
            ok = k == items[i]["key"]
            dr.text((3, 13 + 12 * j), f"{nm}: {k.split('___')[0][:8]} {k.split('___')[1][:14]} {p[i].max():.2f}",
                    fill=(120, 255, 120) if ok else (255, 120, 120))
        sheet.paste(tile, ((n % 3) * T, (n // 3) * T))
    sheet.save(out / "external_samples.jpg", quality=88)
    print("EXTERNAL DONE", json.dumps({nm: round(r["overall"]["accuracy"], 4) for nm, r in report["models"].items()}),
          f"in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
