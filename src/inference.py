import io
import os
from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from matplotlib import colormaps

from model import build_model
from evaluate import gradcam
from ood import feature_extractor, mahalanobis

MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/best.pt")
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))
DISCLAIMER = ("Research prototype trained on a public benchmark (PneumoniaMNIST). "
              "Not a medical device; do not use for diagnosis.")


@lru_cache(maxsize=1)
def get_model():
    ck = torch.load(MODEL_PATH, map_location="cpu")
    m = build_model(False)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, int(ck["size"])


@lru_cache(maxsize=1)
def get_ood():
    # statistics of the training images, made by `python src/ood.py`
    path = os.path.join(os.path.dirname(MODEL_PATH), "ood_stats.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path)
    return z["mean"], z["precision"], float(z["threshold"])


def preprocess(img: Image.Image, size: int):
    g = img.convert("L").resize((size, size))
    arr = np.asarray(g, dtype=np.float32) / 255.0
    x = torch.from_numpy((arr - 0.5) / 0.5).expand(3, -1, -1).unsqueeze(0).contiguous()
    return x, np.asarray(g)


def predict(img: Image.Image, with_cam=False):
    model, size = get_model()
    x, gray = preprocess(img, size)
    with torch.no_grad():
        p = torch.sigmoid(model(x)).item()
    out = {
        "probability_pneumonia": round(p, 6),
        "label": "pneumonia" if p >= THRESHOLD else "normal",
        "threshold": THRESHOLD,
        "disclaimer": DISCLAIMER,
    }
    ood = get_ood()
    if ood is not None:
        mean, precision, threshold = ood
        with torch.no_grad():
            feats = feature_extractor(model)(x).numpy()
        ratio = float(mahalanobis(feats, mean, precision)[0] / threshold)
        out["unfamiliar_image"] = ratio > 1.0
        out["distance_ratio"] = round(ratio, 2)
    overlay = None
    if with_cam:
        cam = gradcam(model, x)[0]
        heat = (colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8)
        base = np.stack([gray] * 3, -1)
        overlay = Image.fromarray((0.6 * base + 0.4 * heat).astype(np.uint8))
    return out, overlay


def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
