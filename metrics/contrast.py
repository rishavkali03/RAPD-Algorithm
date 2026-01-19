import numpy as np
import cv2


def calculate_contrast(image):
    """
    Calculate image contrast using standard deviation
    of pixel intensity values.

    Higher contrast indicates better visibility of details.

    Parameters:
        image (numpy.ndarray): Input image (grayscale or color)

    Returns:
        float: Contrast value
    """

    # Convert to grayscale if image is color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Convert to float for numerical stability
    gray = gray.astype(np.float64)

    # Compute contrast as standard deviation
    contrast = np.std(gray)

    return contrast
