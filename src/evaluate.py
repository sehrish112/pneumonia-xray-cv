import argparse
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from data import get_loaders
from model import build_model
from train import predict_all


def gradcam(model, x):
    # Grad-CAM on the last resnet block: weight each feature map by the mean gradient
    feats, grads = [], []
    layer = model.layer4
    h1 = layer.register_forward_hook(lambda m, i, o: feats.append(o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
    model.zero_grad()
    model(x).sum().backward()
    h1.remove()
    h2.remove()

    weights = grads[0].mean((2, 3), keepdim=True)
    cam = torch.relu((weights * feats[0]).sum(1)).detach()
    cam = torch.nn.functional.interpolate(cam[:, None], size=x.shape[-2:], mode="bilinear")[:, 0]
    cam = cam - cam.amin((1, 2), keepdim=True)
    cam = cam / (cam.amax((1, 2), keepdim=True) + 1e-8)
    return cam.numpy()


def load_model(out_dir):
    ckpt = torch.load(f"{out_dir}/best.pt", map_location="cpu")
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["size"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    model, size = load_model(args.out)
    _, _, test_dl, _ = get_loaders(size)
    y, p = predict_all(model, test_dl, "cpu")
    np.savez(f"{args.out}/test_predictions.npz", y=y, p=p)

    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    metrics = {
        "accuracy": accuracy_score(y, pred),
        "auc": roc_auc_score(y, p),
        "precision": precision_score(y, pred),
        "recall_sensitivity": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "n_test": int(len(y)),
        "specificity": float(tn / (tn + fp)),
    }
    with open(f"{args.out}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(metrics)

    # roc curve
    fpr, tpr, _ = roc_curve(y, p)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {metrics['auc']:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.savefig(f"{args.out}/roc.png", dpi=150)
    plt.close()

    # confusion matrix
    cm = [[tn, fp], [fn, tp]]
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i][j], ha="center", va="center")
    plt.xticks([0, 1], ["Normal", "Pneumonia"])
    plt.yticks([0, 1], ["Normal", "Pneumonia"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"{args.out}/confusion_matrix.png", dpi=150)
    plt.close()

    # grad-cam on the first 8 test images
    x, labels = next(iter(test_dl))
    cams = gradcam(model, x[:8])
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(x[i, 0] * 0.5 + 0.5, cmap="gray")
        ax.imshow(cams[i], cmap="jet", alpha=0.4)
        ax.set_title(f"true={int(labels[i])}")
        ax.axis("off")
    plt.savefig(f"{args.out}/gradcam.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
