import cv2
import numpy as np


def normalize_image(
    image,
    out_range=(0, 255),
    dtype=np.uint8
):
    """
    Normalize image intensity values to a specified range.

    Parameters:
        image (numpy.ndarray): Input image (grayscale or color)
        out_range (tuple): Desired output intensity range
        dtype (numpy.dtype): Output data type

    Returns:
        numpy.ndarray: Normalized image
    """

    # Convert image to float for stable normalization
    image_float = image.astype(np.float64)

    # Normalize to desired range
    normalized = cv2.normalize(
        image_float,
        None,
        alpha=out_range[0],
        beta=out_range[1],
        norm_type=cv2.NORM_MINMAX
    )

    return normalized.astype(dtype)
