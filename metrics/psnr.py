import numpy as np
import cv2


def calculate_psnr(original, enhanced):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between
    original and enhanced images.

    PSNR evaluates the distortion introduced during enhancement.

    Parameters:
        original (numpy.ndarray): Original input image
        enhanced (numpy.ndarray): Enhanced image

    Returns:
        float: PSNR value in decibels (dB)
    """

    # Convert images to grayscale if they are color
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original.copy()

    if len(enhanced.shape) == 3:
        enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    else:
        enhanced_gray = enhanced.copy()

    # Convert to float for accurate computation
    original_gray = original_gray.astype(np.float64)
    enhanced_gray = enhanced_gray.astype(np.float64)

    # Compute Mean Squared Error (MSE)
    mse = np.mean((original_gray - enhanced_gray) ** 2)

    # Avoid division by zero
    if mse == 0:
        return float('inf')

    # Maximum pixel intensity value
    max_pixel = 255.0

    # Compute PSNR
    psnr = 10 * np.log10((max_pixel ** 2) / mse)

    return psnr
