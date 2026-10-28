"""API tests. Run from api/:  pytest -q"""
import io
import os
import sys
from pathlib import Path

import pytest
from PIL import Image

API_DIR = Path(__file__).resolve().parents[1]
SAMPLES = API_DIR.parent / "frontend" / "public" / "samples"
sys.path.insert(0, str(API_DIR))

if not (API_DIR / "models" / "leafnet.onnx").exists():
    pytest.skip("trained model not exported yet (run ml/export_onnx.py)", allow_module_level=True)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["DB_PATH"] = str(tmp_path_factory.mktemp("db") / "test.db")
    os.environ["FRONTEND_DIST"] = str(tmp_path_factory.mktemp("nofrontend") / "missing")
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


LABELS = __import__("json").loads((API_DIR / "models" / "labels.json").read_text())


def sample(name: str) -> tuple[str, bytes, str]:
    return (f"{name}.jpg", (SAMPLES / f"{name}.jpg").read_bytes(), "image/jpeg")


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True
    assert r.json()["classes"] == len(LABELS)


def test_every_class_has_disease_info(client):
    import json
    from app.main import KNOWLEDGE
    for key in json.loads((API_DIR / "models" / "labels.json").read_text()):
        assert key in KNOWLEDGE, key


@pytest.mark.parametrize("name,expected", [
    ("tomato_early_blight", "Tomato___Early_blight"),
    ("potato_late_blight", "Potato___Late_blight"),
    ("corn_common_rust", "Corn___Common_rust"),
    ("pepper_healthy", "Pepper,_bell___healthy"),
])
def test_predict_known_samples(client, name, expected):
    r = client.post("/api/predict", files={"file": sample(name)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["disease"]["key"] == expected
    assert 0 <= body["confidence"] <= 1
    assert abs(sum(k["probability"] for k in body["top_k"]) - 1) < 0.2
    assert body["heatmap"].startswith("data:image/png;base64,")
    assert body["status"] in {"healthy", "diseased"}
    assert body["id"] is not None


def test_predict_tta_and_no_save(client):
    r = client.post("/api/predict?tta=true&save=false&heatmap=false", files={"file": sample("apple_scab")})
    body = r.json()
    assert r.status_code == 200
    assert body["tta"] is True and body["id"] is None and body["heatmap"] is None


def test_rejects_non_image(client):
    r = client.post("/api/predict", files={"file": ("notes.txt", b"hello world", "text/plain")})
    assert r.status_code == 415


def test_rejects_empty(client):
    r = client.post("/api/predict", files={"file": ("x.jpg", b"", "image/jpeg")})
    assert r.status_code == 400


def test_quality_warnings_for_dark_image(client):
    buf = io.BytesIO()
    Image.new("RGB", (300, 300), (5, 5, 5)).save(buf, "JPEG")
    r = client.post("/api/predict?save=false", files={"file": ("dark.jpg", buf.getvalue(), "image/jpeg")})
    assert r.status_code == 200
    assert any("dark" in w for w in r.json()["quality"]["warnings"])


def test_batch(client):
    files = [("files", sample("grape_black_rot")), ("files", sample("apple_scab")),
             ("files", ("bad.txt", b"nope", "text/plain"))]
    r = client.post("/api/predict/batch", files=files)
    body = r.json()
    assert r.status_code == 200
    assert body["count"] == 3 and body["summary"]["error"] == 1
    assert body["items"][0]["ok"] and not body["items"][2]["ok"]


def test_history_roundtrip(client):
    page = client.get("/api/history?limit=5").json()
    assert page["total"] >= 4
    first = page["items"][0]
    assert first["thumbnail"].startswith("data:image/jpeg")
    assert client.get("/api/history/stats").json()["total"] == page["total"]
    assert client.delete(f"/api/history/{first['id']}").status_code == 200
    assert client.delete(f"/api/history/{first['id']}").status_code == 404
    assert client.get("/api/history").json()["total"] == page["total"] - 1


def test_diseases_endpoint(client):
    all_ = client.get("/api/diseases").json()
    assert len(all_) == len(LABELS)
    tomato = client.get("/api/diseases?crop=tomato").json()
    assert len(tomato) == sum(k.startswith("Tomato___") for k in LABELS)
    assert client.get("/api/diseases/Nope").status_code == 404


def test_crop_filter_restricts_diagnosis(client):
    crops = client.get("/api/crops").json()
    assert {"id": "Potato", "name": "Potato"} in crops
    r = client.post("/api/predict?save=false&heatmap=false&crop=Tomato", files={"file": sample("potato_late_blight")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["crop_filter"] == "Tomato"
    assert all(k["key"].startswith("Tomato___") or k["key"] == "Background_without_leaves" for k in body["top_k"])
    assert client.post("/api/predict?crop=Banana2", files={"file": sample("apple_scab")}).status_code == 422
