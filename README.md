# Adversarial Attack and Defense in Medical Imaging

This repository contains the codebase for research on Adversarial Machine Learning, with a specific focus on Medical Imaging Security. It explores the vulnerabilities of deep learning models in medical tasks and implements state-of-the-art defenses against them.

## Overview

Deep learning models, while highly accurate, are susceptible to adversarial attacks—small, often imperceptible perturbations that cause misclassification. This project investigates:
- **Universal Adversarial Perturbations (UAP):** Single noise patterns capable of fooling models across many different images.
- **Backdoor/Poisoning Attacks:** Methods to inject hidden vulnerabilities during the training phase.
- **Adversarial Training Defenses:** Robust training methodologies (like AWP, OAAT, and FBF) to mitigate these threats.

### Target Domains & Datasets
- **ISIC:** Skin Cancer Detection (ISIC 2018/2019)
- **OCT:** Retinal Optical Coherence Tomography (OCT 2017)
- **COVID-Net:** Chest X-Ray Analysis for COVID-19 (CXRAY)
- **CIFAR-10:** Standard benchmark for baseline evaluations

## Repository Structure

```
.
├── data/               # Datasets, CSV metadata, and data splits (ignored by git)
├── experiments/        # Results, logs, generated noise patterns, and output figures (ignored by git)
├── src/                # Core source code
│   ├── attacks/        # Implementation of UAP and other attack scripts
│   ├── defenses/       # Adversarial training scripts (AWP, OAAT, etc.)
│   ├── models/         # Model architectures and standard training scripts
│   └── utils/          # Data loaders, helper functions, and shared utilities
└── third_party/        # External toolboxes (ART, backdoor-toolbox, UAP-COVID-Net) (ignored by git)
```

## Setup

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the corresponding datasets inside the `data/` directory as required by the scripts.

## Usage

*(Detailed usage instructions to be added depending on the specific experiment to reproduce)*

- To run standard training, navigate to `src/models/` and execute the appropriate script (e.g., `train_OCT.py`).
- Attack scripts can be found in `src/attacks/` (e.g., `OCT2017_UAP.py`).
- Defense testing and adversarial training scripts are in `src/defenses/`.
