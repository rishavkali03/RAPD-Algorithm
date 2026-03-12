import cv2
from skimage.metrics import structural_similarity as ssim


def calculate_ssim(original, enhanced):
    """
    Calculate Structural Similarity Index (SSIM) between
    original and enhanced images.

    SSIM measures perceptual and structural similarity.

    Parameters:
        original (numpy.ndarray): Original input image
        enhanced (numpy.ndarray): Enhanced image

    Returns:
        float: SSIM value (range: -1 to 1, typically 0 to 1)
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

    # Compute SSIM
    ssim_value = ssim(
    original_gray,
    enhanced_gray,
    data_range=255
)
    return ssim_value

