import sys
import os
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.load_image import load_image
from preprocessing.normalization import normalize_image
from preprocessing.clahe_he import apply_he, apply_clahe, apply_he_clahe
from optimization.fitness import calculate_fitness


def show(title, image, fitness, pos):
    plt.subplot(1, 4, pos)
    plt.imshow(image, cmap="gray")
    plt.title(f"{title}\nFitness = {fitness:.2f}")
    plt.axis("off")


if __name__ == "__main__":

    img_path = "images/Sample.jpg"

    # Load & normalize
    original = load_image(img_path, resize_dim=(256, 256), grayscale=True)
    normalized = normalize_image(original)

    # Generate enhanced versions
    he_img = apply_he(normalized)
    clahe_img = apply_clahe(normalized)
    he_clahe_img = apply_he_clahe(normalized)

    # Compute fitness
    fitness_he = calculate_fitness(original, he_img)
    fitness_clahe = calculate_fitness(original, clahe_img)
    fitness_he_clahe = calculate_fitness(original, he_clahe_img)

    # Visualization
    plt.figure(figsize=(16, 4))

    show("HE", he_img, fitness_he, 1)
    show("CLAHE", clahe_img, fitness_clahe, 2)
    show("HE + CLAHE", he_clahe_img, fitness_he_clahe, 3)

    plt.tight_layout()
    plt.show()

    # Print numeric comparison
    print("Fitness comparison:")
    print(f"HE        : {fitness_he:.2f}")
    print(f"CLAHE    : {fitness_clahe:.2f}")
    print(f"HE+CLAHE : {fitness_he_clahe:.2f}")
