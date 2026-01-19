import numpy as np
import cv2


def calculate_entropy(image):
    """
    Calculate Shannon entropy of an image.

    Entropy measures the amount of information
    and detail present in an image.

    Parameters:
        image (numpy.ndarray): Input image (grayscale or color)

    Returns:
        float: Entropy value
    """

    # Convert to grayscale if image is color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Compute histogram (256 intensity levels)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # Normalize histogram to get probability distribution
    hist = hist.ravel() / hist.sum()

    # Remove zero entries to avoid log2(0)
    hist = hist[hist > 0]

    # Compute Shannon entropy
    entropy = -np.sum(hist * np.log2(hist))

    return entropy

