import cv2

# Metric imports
from metrics.psnr import calculate_psnr
from metrics.ssim import calculate_ssim
from metrics.entropy import calculate_entropy
from metrics.brightness_variance import calculate_brightness_variance

# Enhancement methods
from preprocessing.clahe_he import apply_he
from preprocessing.clahe_he import apply_clahe

# RAPD algorithm
from rapd_algorithm import rapd_optimize

# Visualization
from utils.visualize import plot_fitness_curve
from utils.visualize import show_image_comparison
from utils.visualize import plot_histogram_comparison
from utils.visualize import plot_algorithm_comparison
from utils.visualize import plot_radar_chart
from utils.visualize import plot_metrics_table


def main():

    
    # Load Image
   
    image = cv2.imread("images/Sample.jpg", cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Image not found")
        return

    print("Image loaded successfully.")

   
    # Run RAPD Optimization
   
    best_image, fitness, params, history = rapd_optimize(image)

    print("\nBest RAPD Fitness:", round(fitness, 4))
    print("Best Parameters:", params)

    # Save enhanced image
    cv2.imwrite("rapd_enhanced_result.jpg", best_image)
    print("Enhanced image saved as rapd_enhanced_result.jpg")

   
    # Visualization of RAPD results
   
    plot_fitness_curve(history)

    show_image_comparison(image, best_image)

    plot_histogram_comparison(image, best_image)

    
    # Run Classical Enhancement Methods
    
    he_image = apply_he(image)

    clahe_image = apply_clahe(image)

    # WCA approximation (temporary)
    wca_image = apply_clahe(image)

   
    # Compute Metrics Automatically
    
    metrics = {

        "HE": [
            calculate_psnr(image, he_image),
            calculate_ssim(image, he_image),
            calculate_entropy(he_image),
            calculate_brightness_variance(he_image)
        ],

        "CLAHE": [
            calculate_psnr(image, clahe_image),
            calculate_ssim(image, clahe_image),
            calculate_entropy(clahe_image),
            calculate_brightness_variance(clahe_image)
        ],

        "WCA": [
            calculate_psnr(image, wca_image),
            calculate_ssim(image, wca_image),
            calculate_entropy(wca_image),
            calculate_brightness_variance(wca_image)
        ],

        "RAPD": [
            calculate_psnr(image, best_image),
            calculate_ssim(image, best_image),
            calculate_entropy(best_image),
            calculate_brightness_variance(best_image)
        ]
    }

    
    # Print Metrics in Terminal
    
    for algo, values in metrics.items():

        print(f"\n{algo} Metrics")

        print(f"PSNR: {values[0]:.2f}")

        print(f"SSIM: {values[1]:.4f}")

        print(f"Entropy: {values[2]:.2f}")

        print(f"Brightness Variance: {values[3]:.2f}")

    
    # Plot Comparison Charts
   
    plot_algorithm_comparison(metrics)

    plot_radar_chart(metrics)

    plot_metrics_table(metrics)


if __name__ == "__main__":
    main()