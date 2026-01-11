# RAPD Algorithm  
### (RAPD Optimization Algorithm)  
**Robust Adaptive Perceived-Detail Driven Multi-Metric Optimization Algorithm**

---

## 📌 Overview

Image enhancement is a fundamental task in image processing aimed at improving visual quality while preserving important structural and perceptual details. Classical enhancement techniques such as Histogram Equalization (HE) and CLAHE improve contrast but often suffer from over-enhancement, noise amplification, and brightness distortion.

To address these limitations, this project proposes a novel image enhancement framework called the **Rapid Algorithm (RAPD Optimization Algorithm)**. RAPD is designed as a **multi-metric optimization-based image enhancement approach** that jointly optimizes multiple perceptual and statistical image quality parameters to achieve balanced and visually natural enhancement.

---

## 🎯 Key Idea

Most existing image enhancement techniques focus on optimizing a **single image quality metric**, which often leads to degradation in other aspects of image quality. RAPD overcomes this limitation by treating image enhancement as a **multi-metric optimization problem**, where multiple quality measures are optimized simultaneously within a unified framework.

---

## 🔍 Image Quality Metrics Optimized

The RAPD algorithm jointly optimizes the following metrics:

- **PSNR (Peak Signal-to-Noise Ratio)**  
  Controls noise amplification and distortion

- **SSIM (Structural Similarity Index)**  
  Preserves structural and perceptual similarity

- **Entropy**  
  Enhances information richness and fine detail visibility

- **Brightness Variance (BV)**  
  Maintains natural illumination consistency

- **Contrast Measures**  
  Improves visibility of important image regions

---

## 🧠 Core Contribution

- A **novel multi-metric fitness function** that balances perceptual and statistical image quality parameters
- A **robust and adaptive optimization framework** for image enhancement
- Reduction of over-enhancement and brightness distortion commonly observed in classical methods
- A scalable and modular implementation suitable for research and real-world applications

---
