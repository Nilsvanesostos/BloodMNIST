# Efficient Deep Learning Architectures for Blood Cell Classification

This repository contains the code, utilities, and report for the classification of blood cells from the BloodMNIST dataset. The project focuses on optimizing convolutional neural network (CNN) architectures to achieve high accuracy while maintaining computational efficiency and interpretability.

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Key Features](#key-features)
- [Results](#results)

---

## Dataset

We use the **BloodMNIST dataset**, a publicly available dataset that provides labeled images of blood cells for medical imaging tasks. The dataset is suitable for building and evaluating machine learning models for blood cell classification.

For more information about BloodMNIST, visit: [BloodMNIST Dataset](https://medmnist.com/)

---

## Project Structure

The repository is organized as follows:

```plaintext
├── Gupta_vanEsOstos.ipynb      # Main code file
│ 
├── utils/
│   ├── models.py                # Definitions of CNN architectures (e.g., VGG, ResNet, etc.)
│   ├── training.py              # Functions for training and validation pipelines
│   ├── metrics.py               # Metric computation for model evaluation
│   ├── cnn_performance_metrics.py  # Analysis of CNN performance metrics
│   ├── cnn_experiments.py       # Experimental setups and configurations for CNNs
├── Gupta_vanEsOstos.pdf         # PDF of the report on the analysis
│  
└── README.md  # This README file
```

---

## Usage

Clone the repository:

   ```bash
   git clone https://github.com/Nilsvanesostos/BloodMNIST.git
   cd BloodMNIST
   ```

---

## Key Features

- **BloodMNIST Dataset Integration:** Leverages the BloodMNIST dataset for effective training and testing of models.
- **Class Imbalance:** Explores the balace of the data and possible solutions using data augmentation and resampling techniques.
- **Advanced Neural Networks Architectures:** Explores architectures like CNN, VGG, Autoencoder, InceptionNet, ResNet, and Attention Mechanism.
- **Detailed Metrics Analysis:** Provides insights into model performance through robust metrics and error analysis.

---

## Results

- Achieved a classification **accuracy of 98.3%** on the BloodMNIST dataset.
- Successfully optimized CNN and VGG architectures with attention mechanisms for a balance of accuracy, interpretability, and computational efficiency.
- Detailed error analysis and performance evaluation results are provided in the report.

<img width="415" alt="GradCam_Attention" src="https://github.com/user-attachments/assets/be674315-650e-4fca-aaf1-259de7e857b7" />

