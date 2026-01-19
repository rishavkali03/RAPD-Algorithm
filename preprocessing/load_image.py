import cv2


def load_image(
    image_path,
    resize_dim=(256, 256),
    grayscale=False
):
    """
    Load and preprocess an image for RAPD pipeline.

    Parameters:
        image_path (str): Path to input image
        resize_dim (tuple): Target image size (width, height)
        grayscale (bool): Convert image to grayscale if True

    Returns:
        numpy.ndarray: Loaded image
    """

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Resize image
    image = cv2.resize(image, resize_dim)

    # Convert color space if required
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image
