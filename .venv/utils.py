import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_images(folder):
    files = sorted(os.listdir(folder))
    imgs = []

    for f in files:
        path = os.path.join(folder, f)

        try:
            img = Image.open(path).convert("RGB")
        except:
            continue

        img = np.array(img).astype(np.float32) / 255.0
        imgs.append(img)

    return np.array(imgs)


def show_results(originals, background, foregrounds):
    N = len(originals)
    idx = N // 2

    plt.figure(figsize=(12, 4))

    # Background
    plt.subplot(1, 3, 1)
    plt.title("Background (low-rank)")
    plt.imshow(np.clip(background, 0, 1))
    plt.axis("off")

    # Foreground
    plt.subplot(1, 3, 2)
    plt.title("Exemplo Foreground")
    plt.imshow(np.clip(np.abs(foregrounds[idx]), 0, 1))
    plt.axis("off")

    # Original
    plt.subplot(1, 3, 3)
    plt.title("Original")
    plt.imshow(originals[idx])
    plt.axis("off")

    plt.tight_layout()
    plt.show()
