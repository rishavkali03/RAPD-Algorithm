import matplotlib.pyplot as plt
import numpy as np
import cv2


# -------------------------------------------------
# Fitness vs Iteration Curve
# -------------------------------------------------
def plot_fitness_curve(history):

    iterations = list(range(1, len(history) + 1))

    plt.figure(figsize=(6,4))

    plt.plot(iterations, history, marker='o', color='blue', linewidth=2)

    plt.title("RAPD Optimization Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness Score")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig("fitness_curve.png", dpi=300)

    plt.show()


# -------------------------------------------------
# Original vs RAPD Enhanced Image
# -------------------------------------------------
def show_image_comparison(original, enhanced):

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.imshow(original, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(enhanced, cmap="gray")
    plt.title("RAPD Enhanced Image")
    plt.axis("off")

    plt.tight_layout()

    plt.savefig("image_comparison.png", dpi=300)

    plt.show()


# -------------------------------------------------
# Histogram Comparison
# -------------------------------------------------
def plot_histogram_comparison(original, enhanced):

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.hist(original.ravel(), bins=256, color='gray')
    plt.title("Histogram of Original Image")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    plt.subplot(1,2,2)
    plt.hist(enhanced.ravel(), bins=256, color='blue')
    plt.title("Histogram of RAPD Enhanced Image")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig("histogram_comparison.png", dpi=300)

    plt.show()


# -------------------------------------------------
# Algorithm Comparison Bar Chart
# -------------------------------------------------
def plot_algorithm_comparison(metrics):

    labels = ["PSNR", "SSIM", "Entropy", "BV"]
    algorithms = list(metrics.keys())

    values = np.array(list(metrics.values()))

    x = np.arange(len(labels))
    width = 0.2

    plt.figure(figsize=(9,5))

    colors = ["#999999", "#66c2a5", "#fc8d62", "#8da0cb"]

    for i, algo in enumerate(algorithms):
        plt.bar(x + i*width, values[i], width, label=algo, color=colors[i])

    plt.xticks(x + width*1.5, labels)

    plt.title("Performance Comparison of Image Enhancement Algorithms")
    plt.xlabel("Evaluation Metrics")
    plt.ylabel("Metric Value")

    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()

    plt.savefig("algorithm_comparison.png", dpi=300)

    plt.show()


# -------------------------------------------------
# Radar Chart (Spider Chart)
# -------------------------------------------------
def plot_radar_chart(metrics):

    labels = ["PSNR", "SSIM", "Entropy", "BV"]

    num_vars = len(labels)

    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

    for algo, values in metrics.items():

        values = values + values[:1]

        ax.plot(angles, values, label=algo)
        ax.fill(angles, values, alpha=0.1)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)

    plt.title("Radar Chart Comparison of Enhancement Algorithms")

    plt.legend(loc='upper right', bbox_to_anchor=(1.3,1.1))

    plt.tight_layout()

    plt.savefig("radar_comparison.png", dpi=300)

    plt.show()


# -------------------------------------------------
# Metrics Table Visualization (NEW)
# -------------------------------------------------
def plot_metrics_table(metrics):

    algorithms = list(metrics.keys())
    values = np.array(list(metrics.values()))

    columns = ["PSNR", "SSIM", "Entropy", "BV"]

    fig, ax = plt.subplots(figsize=(8,3))

    ax.axis('off')

    table = ax.table(
        cellText=[[f"{v:.2f}" for v in row] for row in values],
        rowLabels=algorithms,
        colLabels=columns,
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    # Highlight best values
    for col in range(values.shape[1]):
        best_index = np.argmax(values[:, col])
        table[(best_index+1, col)].set_facecolor("#b6fcb6")

    plt.title("Performance Metrics Comparison", fontsize=14)

    plt.savefig("table.png", dpi=300, bbox_inches='tight')

    plt.show()