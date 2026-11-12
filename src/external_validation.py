"""Test the trained model on a different chest X-ray dataset, without retraining.

ChestMNIST (NIH ChestX-ray14: adult patients, a different hospital) has 14 disease labels.
We keep only images labelled "pneumonia" (positive) or with no finding at all (negative).
"""
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import medmnist
from medmnist import INFO

from evaluate import load_model
from train import predict_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    model, size = load_model(args.out)

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    info = INFO["chestmnist"]
    ds = getattr(medmnist, info["python_class"])(split="test", transform=tf, download=True,
                                                 size=size, root="data")
    names = [info["label"][str(i)].lower() for i in range(len(info["label"]))]
    pneu = names.index("pneumonia")

    labels = ds.labels  # shape (n, 14), multi-label
    is_pneu = labels[:, pneu] == 1
    no_finding = labels.sum(1) == 0
    keep = np.where(is_pneu | no_finding)[0]
    print(f"test images: {len(ds)}, pneumonia: {is_pneu.sum()}, no finding: {no_finding.sum()}")

    subset = torch.utils.data.Subset(ds, keep)
    loader = DataLoader(subset, batch_size=128)
    _, p = predict_all(model, loader, "cpu")
    y = is_pneu[keep].astype(int)

    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    auc = roc_auc_score(y, p)
    result = {
        "dataset": "ChestMNIST test (NIH ChestX-ray14), pneumonia vs no finding",
        "n_images": int(len(y)),
        "n_pneumonia": int(y.sum()),
        "n_no_finding": int((1 - y).sum()),
        "auc": float(auc),
        "sensitivity_at_0.5": float(tp / (tp + fn)),
        "specificity_at_0.5": float(tn / (tn + fp)),
        "precision_at_0.5": float(tp / max(tp + fp, 1)),
        "fraction_predicted_pneumonia_at_0.5": float(pred.mean()),
        "median_probability_no_finding": float(np.median(p[y == 0])),
        "median_probability_pneumonia": float(np.median(p[y == 1])),
    }
    with open(f"{args.out}/external_validation.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))

    fpr, tpr, _ = roc_curve(y, p)
    plt.figure()
    plt.plot(fpr, tpr, label=f"external AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("External validation (ChestMNIST)")
    plt.legend()
    plt.savefig(f"{args.out}/external_roc.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
