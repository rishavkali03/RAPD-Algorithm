import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import matplotlib.pyplot as plt

from preprocessing.load_image import load_image
from preprocessing.normalization import normalize_image
from preprocessing.clahe_he import apply_he, apply_clahe, apply_he_clahe

def show(title, image, pos):
    plt.subplot(1, 5, pos)
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")


if __name__ == "__main__":

    # 1. Load sample image (for understanding only)
    img_path = "../images/Sample.jpg"
    original = load_image(
        img_path,
        resize_dim=(256, 256),
        grayscale=True
    )

    # 2. Normalization
    normalized = normalize_image(original)

    # 3. Preprocessing steps
    he_img = apply_he(normalized)
    clahe_img = apply_clahe(normalized)
    he_clahe_img = apply_he_clahe(normalized)

    # 4. Visualization (self understanding)
    plt.figure(figsize=(18, 4))

    show("Original", original, 1)
    show("Normalized", normalized, 2)
    show("HE", he_img, 3)
    show("CLAHE", clahe_img, 4)
    show("HE + CLAHE", he_clahe_img, 5)

    plt.suptitle("Preprocessing Sanity Check")
    plt.tight_layout()
    plt.show()
