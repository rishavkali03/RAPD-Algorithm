import cv2
import random
from skimage.metrics import structural_similarity as ssim


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8,8)):

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    enhanced = clahe.apply(image)

    return enhanced


def calculate_fitness(original, enhanced):

    ssim_score = ssim(original, enhanced)

    fitness = ssim_score * 100

    return fitness


def rapd_optimize(original_image, iterations=20):

    best_fitness = float("-inf")
    best_image = None
    best_params = None

    tile_options = [(4,4),(8,8),(12,12),(16,16)]

    for i in range(iterations):

        clip = random.uniform(0.5,5.0)
        tile = random.choice(tile_options)

        enhanced = apply_clahe(
            original_image,
            clip_limit=clip,
            tile_grid_size=tile
        )

        fitness = calculate_fitness(original_image, enhanced)

        print(f"Iteration {i+1}: clip={clip:.2f}, tile={tile}, fitness={fitness:.2f}")

        if fitness > best_fitness:
            best_fitness = fitness
            best_image = enhanced
            best_params = (round(clip,2), tile)

    return best_image, best_fitness, best_params