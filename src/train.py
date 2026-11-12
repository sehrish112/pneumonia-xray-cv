import argparse
import json
import os

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from data import get_loaders, pos_weight
from model import build_model


def predict_all(model, loader, device):
    model.eval()
    labels, probs = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.append(torch.sigmoid(model(x.to(device))).cpu())
            labels.append(y)
    return torch.cat(labels).numpy().ravel(), torch.cat(probs).numpy().ravel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--size", type=int, default=128, choices=[28, 64, 128, 224])
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scratch", action="store_true", help="train without imagenet weights")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    train_dl, val_dl, _, train_ds = get_loaders(args.size, args.batch)
    model = build_model(pretrained=not args.scratch).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_ds).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_auc = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}"):
            x, y = x.to(device), y.float().to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)
        scheduler.step()

        y_val, p_val = predict_all(model, val_dl, device)
        val_auc = roc_auc_score(y_val, p_val)
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_ds), "val_auc": val_auc})
        print(history[-1])

        # keep the checkpoint with the best validation auc
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"model": model.state_dict(), "size": args.size}, f"{args.out}/best.pt")

    with open(f"{args.out}/history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("best val auc:", best_auc)


if __name__ == "__main__":
    main()
