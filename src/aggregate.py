"""Combine several runs. Usage: python src/aggregate.py outputs outputs_seed1 outputs_seed2 ..."""
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

COLUMNS = ["auc", "accuracy", "recall_sensitivity", "specificity", "precision", "f1"]


def compute_metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tp = ((y == 1) & (pred == 1)).sum()
    tn = ((y == 0) & (pred == 0)).sum()
    fp = ((y == 0) & (pred == 1)).sum()
    fn = ((y == 1) & (pred == 0)).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "auc": roc_auc_score(y, p),
        "accuracy": (tp + tn) / len(y),
        "recall_sensitivity": recall,
        "specificity": tn / max(tn + fp, 1),
        "precision": precision,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
    }


def bootstrap_ci(y, p, n=1000):
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        rows.append(compute_metrics(y[idx], p[idx]))
    return {c: (np.percentile([r[c] for r in rows], 2.5),
                np.percentile([r[c] for r in rows], 97.5)) for c in COLUMNS}


runs = [d for d in sys.argv[1:] if os.path.exists(os.path.join(d, "metrics.json"))]
results = {d: json.load(open(os.path.join(d, "metrics.json"))) for d in runs}
print(f"runs found: {len(runs)} -> {runs}\n")

print("| Metric | Mean | Std | Min | Max |")
print("|---|---|---|---|---|")
summary = {}
for c in COLUMNS:
    values = np.array([results[d][c] for d in runs])
    std = values.std(ddof=1) if len(values) > 1 else 0.0
    summary[c] = {"mean": values.mean(), "std": std, "min": values.min(), "max": values.max()}
    print(f"| {c} | {values.mean():.3f} | {std:.3f} | {values.min():.3f} | {values.max():.3f} |")

first = runs[0]
pred_file = os.path.join(first, "test_predictions.npz")
if os.path.exists(pred_file):
    data = np.load(pred_file)
    ci = bootstrap_ci(data["y"], data["p"])
    print(f"\nbootstrap 95% CI for run '{first}' (1000 resamples of the test set):")
    for c in COLUMNS:
        print(f"  {c}: {results[first][c]:.3f}  [{ci[c][0]:.3f}, {ci[c][1]:.3f}]")
    summary["bootstrap_ci_first_run"] = {c: [float(ci[c][0]), float(ci[c][1])] for c in COLUMNS}

with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)
