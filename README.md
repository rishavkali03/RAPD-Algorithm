# RAPD Algorithm  
### Rapid Algorithm (RAPD Optimization Algorithm)  
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

## ⚙️ Project Structure

RAPD_Algorithm/
│
├── main.py
├── rapd_algorithm.py
│
├── preprocessing/
│ ├── load_image.py
│ ├── normalization.py
│ └── clahe_he.py
│
├── metrics/
│ ├── psnr.py
│ ├── ssim.py
│ ├── entropy.py
│ ├── brightness_variance.py
│ └── contrast.py
│
├── optimization/
│ ├── fitness.py
│ └── parameter_update.py
│
├── utils/
│ └── visualize.py
│
├── images/
├── results/
└── requirements.txt


---

## 🚀 Installation & Setup

Clone the repository and set up the environment:

```bash
git clone https://github.com/rishavkali03/RAPD-Algorithm.git
cd RAPD-Algorithm
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

🧪 Experimental Evaluation (Planned)

The RAPD algorithm will be evaluated and compared against:

Histogram Equalization (HE)

CLAHE

GA-based image enhancement

WCA-based image enhancement

Evaluation will be performed using PSNR, SSIM, Entropy, Brightness Variance, and qualitative visual analysis.

🔮 Future Scope

Extension to color image and video enhancement

Integration with machine learning and deep learning pipelines

Application in medical image analysis and disease detection

Real-time implementation using GPU acceleration

📄 Research & Publication

This project is intended as a research-oriented implementation with the goal of producing a novel, publishable image enhancement algorithm. The modular design supports reproducibility, experimentation, and extension for academic dissemination.

Acknowledgement

Made with ❤️ by
Rishav Kali, Ankana Banerjee, Poulami Das, Debanjan Mandal
under the guidance of Dip Kumar Saha