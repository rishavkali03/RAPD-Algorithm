import cv2
import matplotlib.pyplot as plt
from rapd_algorithm import rapd_optimize


def main():

    # Load image
    image = cv2.imread("images/Sample.jpg", cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Image not found")
        return

    # Run RAPD
    best_image, fitness, params = rapd_optimize(image)

    print("\nBest RAPD Fitness:", fitness)
    print("Best Parameters:", params)

    # Show result
    plt.imshow(best_image, cmap="gray")
    plt.title("RAPD Enhanced Image")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()