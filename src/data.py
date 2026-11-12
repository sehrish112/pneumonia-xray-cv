import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO

FLAG = "pneumoniamnist"


def get_loaders(size=128, batch_size=64, root="data"):
    DataClass = getattr(medmnist, INFO[FLAG]["python_class"])
    os.makedirs(root, exist_ok=True)

    # the images are grayscale but resnet expects 3 channels
    to_3ch = transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    norm = transforms.Normalize([0.5] * 3, [0.5] * 3)

    train_tf = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(size, scale=(0.85, 1.0)),
        transforms.ToTensor(),
        to_3ch,
        norm,
    ])
    eval_tf = transforms.Compose([transforms.ToTensor(), to_3ch, norm])

    train_ds = DataClass(split="train", transform=train_tf, download=True, size=size, root=root)
    val_ds = DataClass(split="val", transform=eval_tf, download=True, size=size, root=root)
    test_ds = DataClass(split="test", transform=eval_tf, download=True, size=size, root=root)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    test_dl = DataLoader(test_ds, batch_size=batch_size)
    return train_dl, val_dl, test_dl, train_ds


def pos_weight(train_ds):
    # there are more pneumonia images than normal ones, so weight the loss
    labels = np.array(train_ds.labels).ravel()
    n_neg = (labels == 0).sum()
    n_pos = (labels == 1).sum()
    return torch.tensor([n_neg / n_pos], dtype=torch.float32)
