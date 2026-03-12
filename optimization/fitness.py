from metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_entropy,
    calculate_brightness_variance,
    calculate_contrast
)


def calculate_fitness(
    original_image,
    enhanced_image,
    weights=None
):
    """
    Calculate RAPD multi-metric fitness value.

    The fitness function jointly optimizes multiple
    image quality metrics to achieve balanced enhancement
    while preserving natural illumination.

    Parameters:
        original_image (numpy.ndarray): Original input image
        enhanced_image (numpy.ndarray): Enhanced image
        weights (dict): Optional weights for metrics

    Returns:
        float: Fitness score
    """

    # Default weights (experimentally chosen)
    if weights is None:
        weights = {
            "psnr": 0.25,
            "ssim": 0.25,
            "entropy": 0.20,
            "contrast": 0.15,
            "brightness": 0.15
        }



    # --- Compute quality metrics ---
    psnr_value = calculate_psnr(original_image, enhanced_image)
    ssim_value = calculate_ssim(original_image, enhanced_image)
    entropy_value = calculate_entropy(enhanced_image)
    contrast_value = calculate_contrast(enhanced_image)

    # --- Brightness consistency ---
    bv_original = calculate_brightness_variance(original_image)
    bv_enhanced = calculate_brightness_variance(enhanced_image)

    # Penalize excessive brightness deviation
    brightness_penalty = abs(bv_enhanced - bv_original)
    # --- Safety handling ---
    # Prevent instability if PSNR becomes infinite
    if psnr_value == float("inf"):
        psnr_value = 100.0

    # --- Fitness computation ---
    fitness_score = (
        weights["psnr"] * psnr_value +
        weights["ssim"] * ssim_value +
        weights["entropy"] * entropy_value +
        weights["contrast"] * contrast_value -
        weights["brightness"] * brightness_penalty
    )

    return fitness_score


