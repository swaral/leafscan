"""Downloaders for each dataset. Each source yields Items; bytes are fetched lazily
(in worker threads) so network downloads and image processing overlap.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import threading
import urllib.parse
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import requests

UA = {"User-Agent": "Mozilla/5.0"}
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

PLANTVILLAGE_URL = ("https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/"
                    "d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded")
PLANTDOC_TREE = "https://api.github.com/repos/pratikkayal/PlantDoc-Dataset/git/trees/master?recursive=1"
PLANTDOC_RAW = "https://raw.githubusercontent.com/pratikkayal/PlantDoc-Dataset/master/"
PLANTWILD_URL = "https://huggingface.co/datasets/uqtwei2/PlantWild/resolve/main/plantwild.zip"
CCMT_API = "https://data.mendeley.com/public-api/datasets/bwh3zbpkpv"
PP2021_TREE = "https://huggingface.co/api/datasets/timm/plant-pathology-2021/tree/main/data"
PP2021_FILE = "https://huggingface.co/datasets/timm/plant-pathology-2021/resolve/main/"


@dataclass
class Item:
    source: str
    label: str
    uid: str
    fetch: Callable[[], bytes]
    meta: dict = field(default_factory=dict)


def _needs_curl(url: str) -> bool:
    # Mendeley rejects Python HTTP clients (TLS fingerprinting) but accepts curl.
    return "mendeley.com" in url


def fetch(url: str, retries: int = 4) -> bytes:
    last = None
    for _ in range(retries):
        try:
            if _needs_curl(url):
                r = subprocess.run(["curl", "-sSfL", "--max-time", "120", "-A", UA["User-Agent"], url],
                                   capture_output=True)
                if r.returncode == 0:
                    return r.stdout
                last = RuntimeError(f"curl exit {r.returncode} for {url}: {r.stderr[:200]!r}")
            else:
                r = requests.get(url, headers=UA, timeout=60)
                if r.status_code == 200:
                    return r.content
                last = RuntimeError(f"HTTP {r.status_code} for {url}")
        except (requests.RequestException, OSError) as e:
            last = e
    raise last


def fetch_json(url: str):
    return json.loads(fetch(url))


def download(url: str, dest: Path, chunk: int = 1 << 22) -> Path:
    """Download a large file once (skipped if already complete)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if dest.exists():
        return dest
    print(f"    downloading {dest.name} ...", flush=True)
    if _needs_curl(url):
        subprocess.run(["curl", "-sSfL", "--retry", "5", "-C", "-", "-A", UA["User-Agent"], "-o", str(tmp), url],
                       check=True)
    else:
        with requests.get(url, headers=UA, stream=True, timeout=120) as r:
            r.raise_for_status()
            size, done = int(r.headers.get("Content-Length", 0)), 0
            with open(tmp, "wb") as fh:
                for block in r.iter_content(chunk):
                    fh.write(block)
                    done += len(block)
                    if size and done % (chunk * 64) < chunk:
                        print(f"    {dest.name}: {done / 1e9:.2f}/{size / 1e9:.2f} GB", flush=True)
    tmp.replace(dest)
    return dest


class RangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file via HTTP range requests (for smoke tests)."""

    def __init__(self, url: str):
        r = requests.head(url, headers=UA, allow_redirects=True, timeout=60)
        self.url, self.size, self.pos = r.url, int(r.headers["Content-Length"]), 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.size, self.pos + len(b)) - 1
        data = requests.get(self.url, headers={**UA, "Range": f"bytes={self.pos}-{end}"}, timeout=120).content
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def _zip_items(source: str, zip_src, label_of: Callable[[str], str | None]) -> Iterator[Item]:
    """Items from a (local or remote) zip; each worker thread gets its own handle."""
    local = threading.local()

    def handle():
        if not hasattr(local, "zf"):
            src = zip_src if isinstance(zip_src, (str, Path)) else zip_src()
            local.zf = zipfile.ZipFile(src if isinstance(src, (str, Path)) else io.BufferedReader(src, 1 << 20))
        return local.zf

    for info in handle().infolist():
        n = info.filename
        if info.is_dir() or "__MACOSX" in n or not n.lower().endswith(IMG_EXT):
            continue
        label = label_of(n)
        if label is None:
            continue
        yield Item(source, label, n, (lambda name=n: handle().read(name)))


# ---------------------------------------------------------------- sources
def plantvillage(raw: Path, smoke: bool, local_zip: str | None = None) -> Iterator[Item]:
    src = local_zip or download(PLANTVILLAGE_URL, raw / "plantvillage.zip")
    yield from _zip_items("plantvillage", src, lambda n: n.split("/")[-2])


def plantdoc(raw: Path, smoke: bool) -> Iterator[Item]:
    tree = fetch_json(PLANTDOC_TREE)["tree"]
    for t in tree:
        p = t["path"]
        parts = p.split("/")
        if t["type"] != "blob" or len(parts) != 3 or parts[0] not in ("train", "test"):
            continue
        if not p.lower().endswith(IMG_EXT):
            continue
        url = PLANTDOC_RAW + urllib.parse.quote(p)
        yield Item("plantdoc", parts[1], p, (lambda u=url: fetch(u)), {"official_split": parts[0]})


def plantwild(raw: Path, smoke: bool) -> Iterator[Item]:
    src = (lambda: RangeFile(PLANTWILD_URL)) if smoke else download(PLANTWILD_URL, raw / "plantwild.zip")

    def label_of(n: str):
        parts = n.split("/")
        return parts[-2] if len(parts) >= 3 and parts[-3] == "images" else None

    yield from _zip_items("plantwild", src, label_of)


def ccmt(raw: Path, smoke: bool) -> Iterator[Item]:
    """CCMT *raw* images only (the augmented release has transformed copies of the same
    photos in train and test). Mendeley's API lists at most 1000 files per folder."""
    folders = fetch_json(f"{CCMT_API}/folders/1")
    by_id = {f["id"]: f for f in folders}

    def path(f):
        out = []
        while f:
            out.append(f["name"])
            f = by_id.get(f.get("parent_id"))
        return "/".join(reversed(out))

    for f in sorted(folders, key=path):
        p = path(f)
        if not p.startswith("Raw Data/CCMT Dataset/") or p.count("/") != 3:
            continue
        label = "/".join(p.split("/")[2:])  # e.g. "Cassava/mosaic"
        files = fetch_json(f"{CCMT_API}/files?folder_id={f['id']}&version=1")
        for x in files:
            if not x["filename"].lower().endswith(IMG_EXT):
                continue
            url = x["content_details"]["download_url"]
            yield Item("ccmt", label, f"{label}/{x['filename']}", (lambda u=url: fetch(u)))


def plantpathology(raw: Path, smoke: bool) -> Iterator[Item]:
    """Plant Pathology 2021 (FGVC8) via the timm parquet mirror; single-label images only.
    Shards are downloaded one at a time and deleted after use (16 GB total)."""
    import pyarrow.parquet as pq

    shards = sorted(x["path"] for x in fetch_json(PP2021_TREE) if x["path"].endswith(".parquet"))
    if smoke:
        shards = shards[-1:]
    for shard in shards:
        url = PP2021_FILE + shard
        if smoke:
            pf = pq.ParquetFile(io.BufferedReader(RangeFile(url), 1 << 22))
            groups = [0]
        else:
            local = download(url, raw / "pp2021" / Path(shard).name)
            pf = pq.ParquetFile(local)
            groups = range(pf.num_row_groups)
        for g in groups:
            table = pf.read_row_group(g, columns=["image", "label_names", "image_id"]).to_pylist()
            for row in table:
                names = row["label_names"]
                if len(names) != 1:
                    continue  # multi-label images are ambiguous for single-label training
                data = row["image"]["bytes"]
                yield Item("plantpathology", names[0], row["image_id"], (lambda d=data: d))
        if not smoke:
            os.remove(local)


REGISTRY = {
    "plantvillage": plantvillage,
    "plantdoc": plantdoc,
    "plantwild": plantwild,
    "ccmt": ccmt,
    "plantpathology": plantpathology,
}
