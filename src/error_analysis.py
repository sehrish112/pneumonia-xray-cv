import argparse
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

from data import get_loaders
from evaluate import gradcam, load_model
from train import predict_all


def plot_mistakes(model, images, probs, idx, title, path, max_show):
    # show the most confident mistakes first
    idx = idx[np.argsort(-np.abs(probs[idx] - 0.5))][:max_show]
    if len(idx) == 0:
        return
    cams = gradcam(model, images[idx])
    fig, axes = plt.subplots(2, len(idx), figsize=(2.2 * len(idx), 4.6), squeeze=False)
    for k, j in enumerate(idx):
        img = images[j, 0] * 0.5 + 0.5
        axes[0, k].imshow(img, cmap="gray")
        axes[0, k].set_title(f"p={probs[j]:.2f}", fontsize=9)
        axes[1, k].imshow(img, cmap="gray")
        axes[1, k].imshow(cams[k], cmap="jet", alpha=0.4)
        axes[0, k].axis("off")
        axes[1, k].axis("off")
    fig.suptitle(title)
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--max_show", type=int, default=8)
    args = parser.parse_args()

    model, size = load_model(args.out)
    _, _, test_dl, _ = get_loaders(size)
    y, p = predict_all(model, test_dl, "cpu")
    images = torch.cat([x for x, _ in test_dl])

    # sensitivity / specificity at different thresholds
    fpr, tpr, thresholds = roc_curve(y, p)
    spec = 1 - fpr
    pred = (p >= 0.5).astype(int)
    false_neg = np.where((y == 1) & (pred == 0))[0]
    false_pos = np.where((y == 0) & (pred == 1))[0]

    summary = {"threshold_0.5": {"false_negatives": int(len(false_neg)),
                                 "false_positives": int(len(false_pos))}}
    reached = np.where(tpr >= 0.99)[0]
    if len(reached) > 0:
        i = reached[0]
        summary["first_threshold_with_99pct_sensitivity"] = {
            "threshold": float(thresholds[i]),
            "sensitivity": float(tpr[i]),
            "specificity": float(spec[i]),
        }

    plt.figure()
    plt.plot(thresholds[1:], tpr[1:], label="Sensitivity")
    plt.plot(thresholds[1:], spec[1:], label="Specificity")
    plt.axvline(0.5, ls="--", c="gray")
    plt.xlabel("Decision threshold")
    plt.title("Sensitivity and specificity vs threshold")
    plt.legend()
    plt.savefig(f"{args.out}/threshold_sweep.png", dpi=150)
    plt.close()

    # calibration
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    plt.figure()
    plt.plot(mean_pred, frac_pos, "o-")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraction of pneumonia")
    plt.title("Calibration")
    plt.savefig(f"{args.out}/calibration.png", dpi=150)
    plt.close()

    plot_mistakes(model, images, p, false_neg, "False negatives",
                  f"{args.out}/errors_false_negatives.png", args.max_show)
    plot_mistakes(model, images, p, false_pos, "False positives",
                  f"{args.out}/errors_false_positives.png", args.max_show)

    with open(f"{args.out}/error_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
