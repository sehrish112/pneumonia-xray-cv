# Model Card: Chest X-ray Pneumonia Classifier

**Author:** Sehrish Masood

## Model
- ResNet18 (ImageNet-pretrained), final layer replaced by one logit (probability of pneumonia).
- Input: grayscale chest X-ray resized to 64x64, replicated to 3 channels, normalised to [-1, 1].
- Training: 10 epochs, AdamW (lr 1e-3, cosine schedule), class-weighted binary cross-entropy,
  random rotation (10 deg) and random resized crop, seed 42, CPU.
- Best epoch chosen by validation AUC.

## Intended use
- Learning and research demonstration of an image-classification workflow
  (training, evaluation, interpretability, serving).
- **Not intended** for clinical use, diagnosis, triage or any decision about a patient.

## Data
- PneumoniaMNIST (MedMNIST v3), derived from paediatric chest X-rays (Kermany et al.).
  Official train / validation / test splits; test set has 624 images (390 pneumonia, 234 normal).
- Low resolution (64x64 used here). Single source, single population (children), so the
  model has not seen adults or other hospitals, scanners or protocols.

## Results (test set, threshold 0.5, single run)
| Metric | Value |
|---|---|
| AUC | 0.989 |
| Accuracy | 0.957 |
| Sensitivity | 0.982 |
| Specificity | 0.915 |
| Precision | 0.950 |
| F1 | 0.966 |

Across 5 seeds: AUC 0.988 +/- 0.002, accuracy 0.951 +/- 0.008, sensitivity 0.982 +/- 0.005,
specificity 0.899 +/- 0.018. Bootstrap 95% CI (seed 42): AUC [0.982, 0.995], specificity [0.878, 0.949].

At a threshold of about 0.16, sensitivity is 99.2% and specificity 87.6%.

## Known limitations
- One official data split. External validation on ChestMNIST (NIH ChestX-ray14, adults,
  pneumonia vs no finding, 12,170 images) gave AUC 0.67, sensitivity 0.65 and specificity 0.57
  at threshold 0.5, so the model does not generalise beyond its training distribution
  (mean external AUC over 5 seeds: 0.678, std 0.012). Fine-tuning on 1,956 in-domain images
  raised it only to 0.754.
- The demo shows an "unfamiliar image" warning based on feature distance to the training images
  (flags 2.6% of internal test images and 49.6% of an external sample). It is an indicator, not a guarantee.
- Validation AUC was about 0.99 from the first epoch, so it discriminates poorly between epochs.
- Predictions are saturated near 0 and 1 (over-confident); no calibration applied.
- Grad-CAM maps are coarse at this resolution and are not evidence of clinical reasoning.
- Likely to be unreliable on X-rays from other datasets, adults, or images that are not chest X-rays;
  there is no out-of-distribution detection.
- Seven pneumonia cases in the test set were missed (false negatives), six of them with
  confidently low scores.

## Ethical considerations
- Medical imaging models can fail unevenly across populations; no fairness analysis was done.
- The API and demo do not store uploaded images.
- Any real deployment would require clinical validation, regulatory approval and expert oversight.
