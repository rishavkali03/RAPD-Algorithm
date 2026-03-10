import cv2
import numpy as np

def normalize_image(image):

    norm = cv2.normalize(
        image,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return norm.astype(np.uint8)