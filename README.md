# ECG Classification Project

This repository contains code and notebooks for ECG signal classification using a Vision Transformer (ViT) model. The project involves preprocessing ECG data, training the model, and visualizing the results.

## Repository Structure

- **ViT-ECG-5-class-classification-model.ipynb**  
  This Jupyter Notebook contains the implementation of the ViT model for classifying ECG signals into 5 classes. It includes the training, evaluation, and testing processes.

- **ecg-ptb-data-preprocessing.ipynb**  
  This Jupyter Notebook handles the preprocessing of ECG data from the PTB dataset. It includes steps such as data normalization, segmentation, and preparation for model input.

- **train.py**  
  This script is used for training the ViT model on the processed ECG dataset. It includes functionalities for model training, validation, and saving the trained model.

- **plot.py**  
  This script contains functions for plotting various results from the training process, such as loss curves and accuracy metrics. It helps visualize the model's performance over time.

- **plot_attention.py**  
  This script visualizes the attention maps from the ViT model. It is useful for understanding how the model focuses on different parts of the ECG signal during classification.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/ecg-classification.git
cd ecg-classification
