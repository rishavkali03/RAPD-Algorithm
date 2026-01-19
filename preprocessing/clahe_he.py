import cv2
import numpy as np


def apply_he(image):
    """
    Apply Histogram Equalization (HE) to an image.

    Parameters:
        image (numpy.ndarray): Input image (grayscale or color)

    Returns:
        numpy.ndarray: HE-enhanced image
    """

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply histogram equalization
    he_image = cv2.equalizeHist(gray)

    return he_image


def apply_clahe(
    image,
    clip_limit=2.0,
    tile_grid_size=(8, 8)
):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Parameters:
        image (numpy.ndarray): Input image (grayscale or color)
        clip_limit (float): Threshold for contrast limiting
        tile_grid_size (tuple): Size of grid for local equalization

    Returns:
        numpy.ndarray: CLAHE-enhanced image
    """

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Create CLAHE object
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    clahe_image = clahe.apply(gray)

    return clahe_image


def apply_he_clahe(
    image,
    clip_limit=2.0,
    tile_grid_size=(8, 8)
):
    """
    Sequentially apply HE followed by CLAHE.

    This provides global contrast normalization
    followed by controlled local enhancement.

    Parameters:
        image (numpy.ndarray): Input image
        clip_limit (float): CLAHE clip limit
        tile_grid_size (tuple): CLAHE tile grid size

    Returns:
        numpy.ndarray: Preprocessed image
    """

    he_image = apply_he(image)
    clahe_image = apply_clahe(
        he_image,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size
    )

    return clahe_image
