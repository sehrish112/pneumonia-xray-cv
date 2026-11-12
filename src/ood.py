"""Out-of-distribution check: is this image similar to the training images?

We take the 512-d features of the trained ResNet before its last layer, fit a Gaussian to the
training images' features, and measure how far a new image is from it (Mahalanobis distance).
Run `python src/ood.py` to fit the detector and test it on in-distribution vs external images.
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import medmnist
from medmnist import INFO

from evaluate import load_model


def feature_extractor(model):
    # everything in resnet except the final linear layer
    return nn.Sequential(*list(model.children())[:-1], nn.Flatten()).eval()


def get_features(model, loader):
    fe = feature_extractor(model)
    feats = []
    with torch.no_grad():
        for x, *_ in loader:
            feats.append(fe(x))
    return torch.cat(feats).numpy()


def fit_gaussian(features):
    mean = features.mean(0)
    cov = np.cov(features, rowvar=False) + 1e-3 * np.eye(features.shape[1])
    return mean, np.linalg.inv(cov)


def mahalanobis(features, mean, precision):
    d = features - mean
    return np.sqrt(np.einsum("ij,jk,ik->i", d, precision, d))


def eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])


def make_loader(flag, split, size, n=None, seed=0):
    ds = getattr(medmnist, INFO[flag]["python_class"])(
        split=split, transform=eval_transform(), download=True, size=size, root="data")
    if n is not None and n < len(ds):
        idx = np.random.default_rng(seed).choice(len(ds), n, replace=False)
        ds = Subset(ds, idx)
    return DataLoader(ds, batch_size=128)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    model, size = load_model(args.out)

    train_f = get_features(model, make_loader("pneumoniamnist", "train", size))
    mean, precision = fit_gaussian(train_f)
    train_d = mahalanobis(train_f, mean, precision)
    threshold = float(np.percentile(train_d, 99))  # flag anything further out than 99% of training images
    np.savez(f"{args.out}/ood_stats.npz", mean=mean, precision=precision, threshold=threshold)

    internal_d = mahalanobis(get_features(model, make_loader("pneumoniamnist", "test", size)), mean, precision)
    external_d = mahalanobis(get_features(model, make_loader("chestmnist", "test", size, n=2000)), mean, precision)

    # how well does the distance separate internal from external images?
    from sklearn.metrics import roc_auc_score
    labels = np.r_[np.zeros(len(internal_d)), np.ones(len(external_d))]
    auc = roc_auc_score(labels, np.r_[internal_d, external_d])

    report = {
        "threshold_99th_percentile_of_training_distances": threshold,
        "fraction_flagged_internal_test": float((internal_d > threshold).mean()),
        "fraction_flagged_external_chestmnist_sample": float((external_d > threshold).mean()),
        "auc_internal_vs_external": float(auc),
        "n_internal": int(len(internal_d)),
        "n_external": int(len(external_d)),
    }
    with open(f"{args.out}/ood_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
