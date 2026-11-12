"""How much of the external gap closes if we fine-tune on some in-domain (ChestMNIST) data?

Fine-tunes the trained model on a balanced sample from the ChestMNIST *train* split, then
evaluates on the ChestMNIST *test* split (same subset as external_validation.py).
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score
import medmnist
from medmnist import INFO

from evaluate import load_model
from ood import eval_transform
from train import predict_all


def pneumonia_vs_none(ds, pneu_idx):
    labels = ds.labels
    pos = np.where(labels[:, pneu_idx] == 1)[0]
    neg = np.where(labels.sum(1) == 0)[0]
    return pos, neg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    torch.manual_seed(0)

    model, size = load_model(args.out)
    info = INFO["chestmnist"]
    cls = getattr(medmnist, info["python_class"])
    names = [info["label"][str(i)].lower() for i in range(len(info["label"]))]
    pneu = names.index("pneumonia")

    train_ds = cls(split="train", transform=eval_transform(), download=True, size=size, root="data")
    test_ds = cls(split="test", transform=eval_transform(), download=True, size=size, root="data")

    # balanced fine-tuning set: all pneumonia images + the same number of "no finding" images
    pos, neg = pneumonia_vs_none(train_ds, pneu)
    rng = np.random.default_rng(0)
    neg = rng.choice(neg, len(pos), replace=False)
    idx = np.r_[pos, neg]
    train_labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(np.float32)
    print(f"fine-tuning on {len(pos)} pneumonia + {len(neg)} no-finding images")

    pos_t, neg_t = pneumonia_vs_none(test_ds, pneu)
    test_idx = np.r_[pos_t, neg_t]
    y_test = np.r_[np.ones(len(pos_t)), np.zeros(len(neg_t))].astype(int)
    test_loader = DataLoader(Subset(test_ds, test_idx), batch_size=128)

    def external_auc():
        _, p = predict_all(model, test_loader, "cpu")
        # predict_all keeps dataset order, which matches test_idx
        return float(roc_auc_score(y_test, p))

    before = external_auc()
    print("external AUC before fine-tuning:", before)

    lookup = dict(zip(idx.tolist(), train_labels.tolist()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(idx))
        for start in range(0, len(order), 64):
            batch_ids = idx[order[start:start + 64]]
            x = torch.stack([train_ds[int(i)][0] for i in batch_ids])
            y = torch.tensor([lookup[int(i)] for i in batch_ids]).unsqueeze(1)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch + 1}: external AUC {external_auc():.3f}")

    after = external_auc()
    result = {
        "finetune_images": int(len(idx)),
        "epochs": args.epochs,
        "external_auc_before": before,
        "external_auc_after": after,
        "note": "fine-tuned on the ChestMNIST train split, evaluated on its test split",
    }
    with open(f"{args.out}/finetune_external.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
