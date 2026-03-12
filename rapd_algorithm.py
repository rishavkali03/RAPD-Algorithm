import cv2
import random
import numpy as np
from skimage.metrics import structural_similarity as ssim



# Apply CLAHE Enhancement

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    enhanced = clahe.apply(image)

    return enhanced



# PSNR Calculation

def calculate_psnr(original, enhanced):

    mse = np.mean((original - enhanced) ** 2)

    if mse == 0:
        return 100

    psnr = 20 * np.log10(255.0 / np.sqrt(mse))

    return psnr



# Entropy Calculation

def calculate_entropy(image):

    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    histogram = histogram.ravel() / histogram.sum()

    histogram = histogram[histogram > 0]

    entropy = -np.sum(histogram * np.log2(histogram))

    return entropy



# Brightness Variance

def calculate_brightness_variance(image):

    return np.var(image)



# Multi-Metric Fitness Function

def calculate_fitness(original, enhanced):

    # SSIM
    ssim_score = ssim(original, enhanced, data_range=255)

    # PSNR
    psnr_score = calculate_psnr(original, enhanced)

    # Entropy
    entropy_score = calculate_entropy(enhanced)

    # Brightness Variance
    bv_original = calculate_brightness_variance(original)
    bv_enhanced = calculate_brightness_variance(enhanced)

    brightness_penalty = abs(bv_enhanced - bv_original)

    # RAPD Weights
    w1 = 0.4   # SSIM
    w2 = 0.3   # PSNR
    w3 = 0.2   # Entropy
    w4 = 0.1   # Brightness penalty

    # Normalize metrics
    psnr_norm = psnr_score / 100
    entropy_norm = entropy_score / 8
    bv_penalty = brightness_penalty / 1000

    fitness = (
        w1 * ssim_score +
        w2 * psnr_norm +
        w3 * entropy_norm -
        w4 * bv_penalty
    )

    return fitness



# RAPD Optimization

def rapd_optimize(original_image, iterations=20):

    best_fitness = float("-inf")
    best_image = None
    best_params = None

    # NEW: Track optimization progress
    fitness_history = []

    tile_options = [(4,4), (8,8), (12,12), (16,16)]

    for i in range(iterations):

        clip = random.uniform(0.5, 5.0)
        tile = random.choice(tile_options)

        enhanced = apply_clahe(
            original_image,
            clip_limit=clip,
            tile_grid_size=tile
        )

        fitness = calculate_fitness(original_image, enhanced)

        # Store fitness for plotting
        fitness_history.append(fitness)

        print(f"Iteration {i+1}: clip={clip:.2f}, tile={tile}, fitness={fitness:.4f}")

        if fitness > best_fitness:
            best_fitness = fitness
            best_image = enhanced
            best_params = (round(clip,2), tile)

    # Return fitness history for visualization
    return best_image, best_fitness, best_params, fitness_history