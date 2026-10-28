"""Build the combined multi-source dataset.

    python build_dataset.py --out /content/leafscan_data              # full (Colab/Kaggle)
    python build_dataset.py --out ./smoke --smoke --max-per-class 12  # tiny local test

Steps
  1. Download each source, map labels (taxonomy.py), resize to short side 256,
     re-encode as RGB JPEG, compute MD5 + perceptual hashes (8 orientations).
  2. Drop exact duplicates; group near-duplicates (pHash distance <= 1 under any
     flip/rotation) across ALL sources; drop groups whose labels disagree.
  3. Drop classes with fewer than --min-images images.
  4. Group-aware stratified split (train/val/test = 80/10/10 per class x source),
     so copies of an image can never sit on both sides of a split.
Outputs: images/, manifest.csv, classes.json, stats.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import sources
from taxonomy import FIELD_SOURCES, SOURCES, UNCERTAIN, map_label

SHORT_SIDE = 256
MAX_LONG = 512


# ---------------------------------------------------------------- hashing
def _dct(n):
    k = np.arange(n)[:, None]
    m = np.cos(np.pi * (2 * np.arange(n)[None, :] + 1) * k / (2 * n)) * np.sqrt(2 / n)
    m[0] /= np.sqrt(2)
    return m


_D = _dct(32)


def phashes(img: Image.Image) -> list[int]:
    """64-bit DCT perceptual hash for each of the 8 flips/rotations."""
    g = np.asarray(img.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    out = []
    for k in range(4):
        for flip in (False, True):
            v = np.rot90(g, k)
            v = v[:, ::-1] if flip else v
            c = (_D @ v @ _D.T)[:8, :8].ravel()
            bits = np.concatenate([[False], c[1:] > np.median(c[1:])])
            out.append(int("".join("1" if b else "0" for b in bits), 2))
    return out


# ---------------------------------------------------------------- per-image processing
def process(item: sources.Item, out_dir: Path) -> dict | None:
    try:
        data = item.fetch()
        img = Image.open(io.BytesIO(data))
        if img.format == "JPEG":
            img.draft("RGB", (SHORT_SIDE * 2, SHORT_SIDE * 2))  # fast DCT-domain downscale
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size
        if min(w, h) < 64:
            return None
        s = SHORT_SIDE / min(w, h)
        nw, nh = round(w * s), round(h * s)
        if max(nw, nh) > MAX_LONG:  # extreme aspect ratios: centre-crop the long side
            img = img.resize((nw, nh), Image.BILINEAR)
            cw, ch = min(nw, MAX_LONG), min(nh, MAX_LONG)
            l, t = (nw - cw) // 2, (nh - ch) // 2
            img = img.crop((l, t, l + cw, t + ch))
        else:
            img = img.resize((nw, nh), Image.BILINEAR if s < 1 else Image.BICUBIC)
        uid = hashlib.sha1(f"{item.source}/{item.uid}".encode()).hexdigest()[:16]
        rel = f"images/{item.source}/{uid}.jpg"
        img.save(out_dir / rel, quality=90)
        return {"path": rel, "source": item.source, "source_label": item.label,
                "key": map_label(item.source, item.label), "md5": hashlib.md5(data).hexdigest(),
                "phash": phashes(img), "orig_w": w, "orig_h": h, **item.meta}
    except Exception as e:  # corrupt image / network failure: skip, but count it
        return {"error": f"{type(e).__name__}: {e}", "source": item.source, "uid": item.uid}


def collect(source: str, out: Path, args) -> list[dict]:
    done_file = out / f"records_{source}.jsonl"
    if done_file.exists() and (out / f".done_{source}").exists():
        print(f"[{source}] already built, reusing")
        return [json.loads(l) for l in done_file.open(encoding="utf-8")]
    (out / "images" / source).mkdir(parents=True, exist_ok=True)
    kw = {"local_zip": args.plantvillage_zip} if source == "plantvillage" else {}
    items = sources.REGISTRY[source](out / "raw", args.smoke, **kw)

    per_label: Counter = Counter()
    records, errors, t0 = [], [], time.time()
    with ThreadPoolExecutor(args.workers) as pool, done_file.open("w", encoding="utf-8") as fh:
        futures = []
        for it in items:
            if it.label not in SOURCES[source]:
                raise KeyError(f"[{source}] label {it.label!r} missing from taxonomy.py")
            if SOURCES[source][it.label] is None:
                continue
            if args.max_per_class and per_label[it.label] >= args.max_per_class:
                continue
            per_label[it.label] += 1
            futures.append(pool.submit(process, it, out))
            if len(futures) >= args.workers * 8:  # bound memory: drain in order
                for f in futures[: args.workers * 4]:
                    _take(f.result(), records, errors, fh)
                futures = futures[args.workers * 4:]
                if len(records) % 2000 < args.workers * 4:
                    print(f"  [{source}] {len(records)} images ({len(records) / (time.time() - t0):.0f}/s)",
                          flush=True)
        for f in futures:
            _take(f.result(), records, errors, fh)
    (out / f".done_{source}").touch()
    print(f"[{source}] {len(records)} images, {len(errors)} errors, {(time.time() - t0) / 60:.1f} min",
          flush=True)
    if errors:
        (out / f"errors_{source}.json").write_text(json.dumps(errors[:200], indent=2))
    return records


def _take(r, records, errors, fh):
    if r is None:
        return
    if "error" in r:
        errors.append(r)
    else:
        records.append(r)
        fh.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------- dedup + grouping
class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a


def group_near_duplicates(recs: list[dict], max_dist: int = 1) -> list[int]:
    """Union images whose pHash (any orientation) is within max_dist of another's canonical hash.
    Pigeonhole: with distance <= 1, at least one 32-bit half matches exactly."""
    n = len(recs)
    dsu = DSU(n)
    canon = np.array([r["phash"][0] for r in recs], dtype=np.uint64)
    buckets = [defaultdict(list), defaultdict(list)]
    for i, h in enumerate(canon):
        buckets[0][int(h) >> 32].append(i)
        buckets[1][int(h) & 0xFFFFFFFF].append(i)
    for i, r in enumerate(recs):
        for v in r["phash"]:
            for half, key in ((0, v >> 32), (1, v & 0xFFFFFFFF)):
                cand = buckets[half].get(key, ())
                if len(cand) > 500:  # degenerate hash (blank image); skip to avoid giant groups
                    continue
                for j in cand:
                    if j != i and bin(v ^ int(canon[j])).count("1") <= max_dist:
                        dsu.union(i, j)
    return [dsu.find(i) for i in range(n)]


# ---------------------------------------------------------------- split
def split_groups(recs, groups, seed=42, fracs=(0.8, 0.1, 0.1)):
    """Assign whole groups to splits, stratified by (key, majority source)."""
    rng = random.Random(seed)
    members = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    strata = defaultdict(list)
    for g, idx in members.items():
        src = Counter(recs[i]["source"] for i in idx).most_common(1)[0][0]
        strata[(recs[idx[0]]["key"], src)].append(g)
    assign = {}
    for _, gs in sorted(strata.items()):
        rng.shuffle(gs)
        total = sum(len(members[g]) for g in gs)
        n_test, n_val = round(total * fracs[2]), round(total * fracs[1])
        if total >= 10:  # make sure tiny strata still get >= 1 val/test image
            n_test, n_val = max(n_test, 1), max(n_val, 1)
        c_test = c_val = 0
        for g in gs:
            if c_test < n_test:
                assign[g], c_test = "test", c_test + len(members[g])
            elif c_val < n_val:
                assign[g], c_val = "val", c_val + len(members[g])
            else:
                assign[g] = "train"
    return [assign[g] for g in groups]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", default=",".join(sources.REGISTRY))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--min-images", type=int, default=100, help="drop classes smaller than this")
    ap.add_argument("--max-per-class", type=int, default=0, help="cap per source label (smoke tests)")
    ap.add_argument("--smoke", action="store_true", help="stream from remote zips, no big downloads")
    ap.add_argument("--plantvillage-zip", default=None, help="use a local PlantVillage zip")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    recs: list[dict] = []
    for s in args.sources.split(","):
        recs += collect(s, out, args)
    n_raw = len(recs)

    # 1. exact duplicates (same file bytes): keep the first occurrence
    seen, kept, exact_dups = set(), [], 0
    for r in recs:
        if r["md5"] in seen:
            exact_dups += 1
            (out / r["path"]).unlink(missing_ok=True)
        else:
            seen.add(r["md5"])
            kept.append(r)
    recs = kept

    # 2. near-duplicate groups; drop groups with conflicting labels
    groups = group_near_duplicates(recs)
    keys_in = defaultdict(set)
    for r, g in zip(recs, groups):
        keys_in[g].add(r["key"])
    conflict = {g for g, ks in keys_in.items() if len(ks) > 1}
    conflict_examples = [[r["source"] + ":" + r["source_label"] for r, g2 in zip(recs, groups) if g2 == g]
                         for g in list(conflict)[:15]]
    recs_groups = [(r, g) for r, g in zip(recs, groups) if g not in conflict]
    n_conflict = len(recs) - len(recs_groups)
    group_sizes = Counter(g for _, g in recs_groups)
    group_sources = defaultdict(set)
    for r, g in recs_groups:
        group_sources[g].add(r["source"])
    cross_source = sum(1 for s in group_sources.values() if len(s) > 1)

    # 3. minimum class size
    counts = Counter(r["key"] for r, _ in recs_groups)
    small = {k: n for k, n in counts.items() if n < args.min_images}
    recs_groups = [(r, g) for r, g in recs_groups if r["key"] not in small]
    recs = [r for r, _ in recs_groups]
    groups = [g for _, g in recs_groups]

    # 4. split
    splits = split_groups(recs, groups)
    classes = sorted({r["key"] for r in recs})
    index = {k: i for i, k in enumerate(classes)}
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "label", "key", "source", "source_label", "split", "group", "domain"])
        for r, g, s in zip(recs, groups, splits):
            w.writerow([r["path"], index[r["key"]], r["key"], r["source"], r["source_label"], s, g,
                        "field" if r["source"] in FIELD_SOURCES else "lab"])

    per_class = defaultdict(lambda: defaultdict(Counter))
    for r, s in zip(recs, splits):
        per_class[r["key"]][r["source"]][s] += 1
    stats = {
        "images_downloaded": n_raw,
        "exact_duplicates_removed": exact_dups,
        "near_duplicate_groups_multi": sum(1 for n in group_sizes.values() if n > 1),
        "cross_source_duplicate_groups": cross_source,
        "label_conflict_images_dropped": n_conflict,
        "label_conflict_examples": conflict_examples,
        "classes_dropped_too_small": small,
        "final_images": len(recs),
        "final_classes": len(classes),
        "split_sizes": Counter(splits),
        "per_source": Counter(r["source"] for r in recs),
        "uncertain_mappings_used": {f"{s}:{l}": note for (s, l), note in UNCERTAIN.items()
                                    if any(r["source"] == s and r["source_label"] == l for r in recs)},
        "per_class": {k: {s: dict(c) for s, c in v.items()} for k, v in sorted(per_class.items())},
    }
    (out / "classes.json").write_text(json.dumps(classes, indent=2))
    (out / "stats.json").write_text(json.dumps(stats, indent=2, default=dict))
    print(json.dumps({k: v for k, v in stats.items() if k not in ("per_class", "label_conflict_examples")},
                     indent=2, default=dict))


if __name__ == "__main__":
    main()
