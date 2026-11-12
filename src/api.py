# run with: uvicorn api:app --app-dir src --port 8000   (docs at /docs)
import base64
import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

import inference

app = FastAPI(title="Chest X-ray Pneumonia Classifier",
              description=inference.DISCLAIMER, version="1.0.0")
MAX_BYTES = 5 * 1024 * 1024


async def _read_image(file: UploadFile) -> Image.Image:
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File larger than 5 MB.")
    try:
        return Image.open(io.BytesIO(data))
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Not a valid image file (use PNG or JPG).")


@app.get("/health")
def health():
    try:
        _, size = inference.get_model()
        return {"status": "ok", "input_size": size}
    except Exception as e:
        raise HTTPException(503, f"Model not loaded: {e}")


@app.post("/predict")
async def predict(file: UploadFile = File(...), include_gradcam: bool = False):
    img = await _read_image(file)
    out, overlay = inference.predict(img, with_cam=include_gradcam)
    if overlay is not None:
        out["gradcam_png_base64"] = base64.b64encode(inference.to_png_bytes(overlay)).decode()
    return out


@app.post("/predict/gradcam", response_class=Response)
async def predict_gradcam(file: UploadFile = File(...)):
    img = await _read_image(file)
    _, overlay = inference.predict(img, with_cam=True)
    return Response(inference.to_png_bytes(overlay), media_type="image/png")
