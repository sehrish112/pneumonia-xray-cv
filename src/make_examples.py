"""Save a few PneumoniaMNIST test images as PNGs for the demo's example gallery.

Gallery: 2 normal and 2 pneumonia images the model gets right, plus 1 normal image it gets
wrong (a false alarm), so the demo also shows a mistake.
"""
import os

import numpy as np
import medmnist
from medmnist import INFO
from PIL import Image

import inference


def main(size=64, out_dir="docs/examples"):
    os.makedirs(out_dir, exist_ok=True)
    ds = getattr(medmnist, INFO["pneumoniamnist"]["python_class"])(
        split="test", download=True, size=size, root="data")
    labels = np.array(ds.labels).ravel()
    rng = np.random.default_rng(0)

    # predicted pneumonia probability for every test image
    probs = np.array([inference.predict(ds[i][0])[0]["probability_pneumonia"] for i in range(len(ds))])
    pred = (probs >= 0.5).astype(int)

    picks = [
        ("1_normal_correct", np.where((labels == 0) & (pred == 0))[0]),
        ("2_normal_correct", np.where((labels == 0) & (pred == 0))[0]),
        ("3_pneumonia_correct", np.where((labels == 1) & (pred == 1))[0]),
        ("4_pneumonia_correct", np.where((labels == 1) & (pred == 1))[0]),
        ("5_normal_model_mistake", np.where((labels == 0) & (pred == 1))[0]),
    ]
    used = set()
    for name, pool in picks:
        pool = [i for i in pool if i not in used]
        i = int(rng.choice(pool))
        used.add(i)
        img, _ = ds[i]
        # enlarge so it looks fine in the browser; the demo shrinks it back before predicting
        img.resize((256, 256), Image.NEAREST).save(f"{out_dir}/{name}.png")
    print("saved examples to", out_dir)


if __name__ == "__main__":
    main()
