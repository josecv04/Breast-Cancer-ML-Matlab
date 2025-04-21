# Breast-Cancer-ML-Matlab

A MATLAB-based machine learning pipeline for early and accurate breast cancer diagnosis using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. This project combines statistical analysis, dimensionality reduction, and both supervised and unsupervised learning techniques to enhance clinical decision-making.

## 📊 Project Overview

This project aims to automate and improve breast cancer diagnostics using classic machine learning workflows in MATLAB. It involves:

- **Data Preprocessing:** Cleaning and preparing 569 patient records.
- **Dimensionality Reduction:** Applying Principal Component Analysis (PCA) to reduce feature space.
- **Unsupervised Learning:** Clustering with K-Means to separate malignant and benign tumors.
- **Supervised Learning:** Training logistic regression models to classify tumor types.
- **Evaluation:** Measuring performance via accuracy scores, confusion matrices, and visualization.

## 🚀 Features

- PCA-based dimensionality reduction with variance visualization
- K-Means clustering with label remapping and evaluation
- Logistic regression with hyperparameter tuning and performance comparison
- Interactive visualizations: heatmaps, histograms, accuracy plots
- Reproducibility ensured through fixed random seeds

## 📁 Dataset

**Source:** [UCI Machine Learning Repository - WDBC Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))  
**Records:** 569 samples with 30 numerical features and binary target (`M` = Malignant, `B` = Benign)

## 📈 Results

- Achieved **96.49% classification accuracy** using optimized logistic regression on PCA-transformed features.
- Clear separability achieved through PCA and K-Means visualization.

## 🛠️ Tech Stack

- MATLAB (R2023a or later recommended)
- PCA, K-Means Clustering, Logistic Regression
- Data visualization tools: heatmaps, scatter plots, confusion matrices

## 📌 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/josecv04/Breast-Cancer-ML-Matlab.git
