# MRI-based Parkinson's Disease Classification

[View on GitHub](https://github.com/Promico-Git/Parkinson-s-Disease-MRI-Classification)

This project implements and evaluates two deep learning models, a 3D Convolutional Neural Network (3D-CNN) and a CNN-LSTM, for the classification of Parkinson's disease from 3D MRI scans. The goal is to distinguish between healthy individuals and patients with Parkinson's.

## Table of Contents

* [File Structure](#file-structure)
* [About the Project](#about-the-project)
* [Key Features](#key-features)
* [Models Implemented](#models-implemented)
* [Getting Started](#getting-started)
* [Usage](#usage)
* [Dependencies](#dependencies)

## File Structure

The project is organized as follows. The MRI data (`.nii` files) should be placed in the `healthy` and `patient` subdirectories.

```
Parkinson-s-Disease-MRI-Classification/
│
├── Datasets/
│   └── MRI/
│       ├── healthy/
│       │   ├── healthy_001.nii
│       │   ├── healthy_002.nii
│       │   └── ...
│       │
│       └── patient/
│           ├── patient_001.nii
│           ├── patient_002.nii
│           └── ...
│
├── MRI Classification Models.py
├── requirements.txt
├── mri_cnn_model.pth         (Generated after training)
└── mri_cnnlstm_model.pth     (Generated after training)
```

## About The Project

This repository contains the Python script for a complete pipeline to train and evaluate deep learning models on MRI data. It includes steps for data loading, preprocessing, augmentation, model definition, training with best practices like learning rate scheduling and early stopping, and comprehensive evaluation.

## Key Features

* **MRI Preprocessing**: Normalizes and resizes 3D MRI scans from `.nii` format.
* **Data Augmentation**: Includes random flips, rotations, and affine transformations to improve model generalization.
* **Two Model Architectures**: Implements both a 3D-CNN for spatial feature extraction and a CNN-LSTM to process MRI scans as sequences of 2D slices.
* **Comprehensive Training Loop**: Features a robust training function with a learning rate scheduler (`ReduceLROnPlateau`) and early stopping to prevent overfitting.
* **In-depth Evaluation**: Calculates and visualizes key metrics, including:
  * Accuracy, Precision, Recall, and F1-Score
  * Area Under the Curve (AUC)
  * Confusion Matrix
  * ROC Curve

## Models Implemented

1. **3D Convolutional Neural Network (MRI3DCNN)**: This model processes the entire 3D volume of the MRI scan at once, capturing spatial features in three dimensions. It consists of a sequence of 3D convolutional and max-pooling layers followed by fully connected layers for classification.

2. **CNN-LSTM (MRI_CNN_LSTM)**: This hybrid model treats the 3D MRI scan as a sequence of 2D slices.
   * A **2D-CNN** acts as a feature extractor for each individual slice.
   * An **LSTM** (Long Short-Term Memory) network then processes the sequence of extracted features to capture dependencies across the slices.

## Getting Started

To get a local copy up and running, follow these simple steps.

1. Clone the repo:
   ```
   git clone [https://github.com/Promico-Git/Parkinson-s-Disease-MRI-Classification.git](https://github.com/Promico-Git/Parkinson-s-Disease-MRI-Classification.git)
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Organize your dataset according to the [File Structure](#file-structure) section above.

2. Run the script from your terminal:
   ```
   python "MRI Classification Models.py"
   ```
The script will automatically handle the data splitting, training, and evaluation for both models, printing the results and saving the trained model weights (`mri_cnn_model.pth` and `mri_cnnlstm_model.pth`).

## Dependencies

* Python 3.x
* PyTorch
* NumPy
* Matplotlib
* NiBabel
* OpenCV-Python
* Scikit-learn
* Glob
