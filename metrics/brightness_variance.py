import numpy as np
import cv2


def calculate_brightness_variance(image):
    """
    Calculate Brightness Variance (BV) of an image.

    Brightness Variance measures the spread of intensity values
    and helps in preserving natural illumination consistency.

    Parameters:
        image (numpy.ndarray): Input image (grayscale or color)

    Returns:
        float: Brightness variance value
    """

    # Convert to grayscale if image is color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Convert to float for numerical stability
    gray = gray.astype(np.float64)

    # Compute variance of pixel intensities
    brightness_variance = np.var(gray)

    return brightness_variance
