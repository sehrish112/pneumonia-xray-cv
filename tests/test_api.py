import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch
from PIL import Image

from model import build_model


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    ckpt = tmp_path_factory.mktemp("m") / "dummy.pt"
    torch.save({"model": build_model(False).state_dict(), "size": 64}, ckpt)
    import inference
    inference.MODEL_PATH = str(ckpt)
    inference.get_model.cache_clear()
    from fastapi.testclient import TestClient
    from api import app
    return TestClient(app)


def _png():
    buf = io.BytesIO()
    Image.new("L", (80, 80), 128).save(buf, "PNG")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["input_size"] == 64


def test_predict(client):
    r = client.post("/predict", files={"file": ("x.png", _png(), "image/png")})
    j = r.json()
    assert r.status_code == 200
    assert 0 <= j["probability_pneumonia"] <= 1
    assert j["label"] in ("normal", "pneumonia")
    assert "disclaimer" in j


def test_predict_with_gradcam(client):
    r = client.post("/predict?include_gradcam=true", files={"file": ("x.png", _png(), "image/png")})
    assert r.status_code == 200 and "gradcam_png_base64" in r.json()


def test_gradcam_png(client):
    r = client.post("/predict/gradcam", files={"file": ("x.png", _png(), "image/png")})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"


def test_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400
