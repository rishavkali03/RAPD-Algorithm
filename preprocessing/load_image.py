import cv2


def load_image(path, grayscale=True):

    if grayscale:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(path)

    if image is None:
        raise ValueError("Image not found at path")

    return image