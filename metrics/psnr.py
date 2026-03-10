import numpy as np
import cv2


def calculate_psnr(original, enhanced):

    # Convert to grayscale
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original.copy()

    if len(enhanced.shape) == 3:
        enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    else:
        enhanced_gray = enhanced.copy()

    # Convert to float32
    original_gray = original_gray.astype(np.float32)
    enhanced_gray = enhanced_gray.astype(np.float32)

    # Mean Squared Error
    mse = np.mean((original_gray - enhanced_gray) ** 2)

    if mse == 0:
        return float("inf")

    max_pixel = 255.0

    psnr = 10 * np.log10((max_pixel ** 2) / mse)

    return psnr