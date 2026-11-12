# Chest X-ray Pneumonia Classification (PyTorch)

Binary classifier (normal vs. pneumonia) on the public **PneumoniaMNIST** chest X-ray
benchmark (MedMNIST v3). A ResNet18 pretrained on ImageNet is fine-tuned, then evaluated
carefully: held-out test set, threshold analysis, calibration, error analysis and Grad-CAM.
The goal was to learn the full workflow of a vision project, including its limits, not to
chase a leaderboard score.

> Research/learning project on a public benchmark. **Not a medical device.**
>
> AI coding assistance (Claude) was used while building this project.

## Pipeline
| Step | Script | Output |
|---|---|---|
| Train (augmentation, class-weighted loss, best epoch by validation AUC) | `src/train.py` | `outputs/best.pt`, `history.json` |
| Test-set evaluation + Grad-CAM | `src/evaluate.py` | `metrics.json`, `roc.png`, `confusion_matrix.png`, `gradcam.png` |
| Error analysis, threshold sweep, calibration | `src/error_analysis.py` | `error_summary.json`, `threshold_sweep.png`, `calibration.png`, `errors_*.png` |
| Combine several runs (mean, std, bootstrap CI) | `src/aggregate.py` | `results_summary.json` |
| Shared inference code | `src/inference.py` | used by API and demo |
| REST API (FastAPI) | `src/api.py` | `/health`, `/predict`, `/predict/gradcam` |
| Interactive demo | `src/app.py` | Gradio web app |
| Tests + CI | `tests/`, `.github/workflows/ci.yml` | `pytest` on every push |
| Container | `Dockerfile` | one-command deployment |
| Documentation | `MODEL_CARD.md` | intended use, limits, ethics |

## Setup and run
```
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python src/train.py --epochs 10 --size 64
python src/evaluate.py
python src/error_analysis.py
python src/app.py            # demo at http://127.0.0.1:7860
pytest -q tests
```
Optional baseline without pretraining:
```
python src/train.py --scratch --epochs 10 --size 64 --out outputs_scratch
python src/evaluate.py --out outputs_scratch
python src/aggregate.py outputs outputs_scratch
```
Training on CPU takes roughly 15 minutes (10 epochs, 64x64 images).

## Results
Setup: ResNet18 (ImageNet-pretrained), 64x64 input, 10 epochs, seed 42, single run.
Test set: 624 images (390 pneumonia, 234 normal), threshold 0.5.

| Metric | Test |
|---|---|
| AUC | 0.989 |
| Accuracy | 0.957 |
| Sensitivity (recall) | 0.982 |
| Specificity | 0.915 |
| Precision | 0.950 |
| F1 | 0.966 |

**Stability across 5 random seeds** (same setup, seeds 42, 1, 2, 3, 4; `seeds_report.md`):

| Metric | Mean | Std | Range |
|---|---|---|---|
| AUC | 0.988 | 0.002 | 0.984 – 0.989 |
| Accuracy | 0.951 | 0.008 | 0.941 – 0.958 |
| Sensitivity | 0.982 | 0.005 | 0.977 – 0.990 |
| Specificity | 0.899 | 0.018 | 0.880 – 0.915 |
| F1 | 0.962 | 0.006 | 0.954 – 0.967 |

Bootstrap 95% confidence intervals for the seed-42 run (1000 test-set resamples):
AUC 0.989 [0.982, 0.995], accuracy 0.957 [0.941, 0.971], sensitivity 0.982 [0.968, 0.995],
specificity 0.915 [0.878, 0.949]. The spread across seeds is small, so the headline result
is not a lucky run. Specificity varies most, which matters if false alarms are costly.

Confusion matrix (seed 42): 383 true positives, 214 true negatives, 20 false positives, 7 false negatives.

## Analysis
**Threshold trade-off.** At the default 0.5 threshold the model favours sensitivity over
specificity. Lowering the threshold to about 0.16 raises sensitivity to 99.2% but drops
specificity to 87.6% (`threshold_sweep.png`). In a screening setting a missed case is usually
costlier than a false alarm, so the operating point should be chosen deliberately.

**Errors.** Seven pneumonia cases were missed. Six received very low pneumonia
probabilities (0.01 to 0.35), i.e. the model was confidently wrong rather than borderline; only
one was close to the threshold (0.46). Several of the missed images look hazy, over- or
under-exposed, or contain tubes/markers, but with only seven cases I cannot draw firm
conclusions about the cause (`errors_false_negatives.png`).

**Grad-CAM.** The heatmaps are coarse, broad bands concentrated in the lower chest and
central regions rather than sharp lesion localisation. This is expected: at 64x64 input the
last ResNet stage has a tiny feature map, so the maps are low resolution. They should be
read as a rough indication of where the evidence comes from, not as proof that the model
uses medically meaningful features. For missed cases the maps are mostly empty, because the
model found no evidence of pneumonia.

**Calibration.** Predictions are saturated near 0 or 1, so the calibration plot has few
distinct bins. There is mild over-confidence in the high-probability range (one bin predicts
about 0.98 but has about 0.84 observed pneumonia). No calibration method was applied.

## Limitations
- Low-resolution benchmark images, not clinical-quality data.
- Single official train/test split; no external validation on a different dataset
  (multi-seed variation and bootstrap intervals are reported above).
- Validation AUC was already about 0.99 after the first epoch and stayed flat, so the
  validation set does little to discriminate between epochs; best-epoch selection is weak.
- My test numbers are higher than the baseline figures I recall from the MedMNIST paper.
  I found no leakage in my pipeline (fixed official splits, model selected on validation only,
  threshold not tuned on test), and results are consistent across 5 seeds, but I have not
  validated on an external dataset, so I cannot say how well they generalise.
- Grad-CAM is a visual aid only.

## Possible next steps
- Evaluate on a different chest X-ray dataset without retraining (external validation).
- Compare from-scratch vs. pretrained and 64 / 128 / 224 px resolution.
- Apply temperature scaling for calibration; evaluate on a different chest X-ray dataset.

## REST API
```
uvicorn api:app --app-dir src --port 8000      # interactive docs: http://127.0.0.1:8000/docs
curl -F "file=@xray.png" "http://127.0.0.1:8000/predict?include_gradcam=true"
```
Returns the pneumonia probability, label, threshold, a disclaimer and (optionally) the
Grad-CAM overlay as base64 PNG. Files over 5 MB and non-image files are rejected.

## Docker
Train first so `outputs/best.pt` exists, then:
```
docker build -t xray-classifier .
docker run -p 8000:8000 xray-classifier
```
(Not yet tested on this machine because Docker is not installed here; CI runs the tests.)

## Demo
`python src/app.py` opens a web UI: upload an X-ray (or load a random test image) to see the
predicted probability and the Grad-CAM overlay. Add a screenshot here:
`docs/demo.png`.
