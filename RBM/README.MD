# Restricted Boltzmann Machine (RBM) for MNIST

A PyTorch implementation of a Restricted Boltzmann Machine (RBM) trained on the MNIST dataset using Contrastive Divergence.

## Overview

This implementation includes:

* A basic RBM model with configurable architecture
* CD-k training (default k=2)
* MNIST data preprocessing
* Training visualization with reconstruction samples
* Training progress monitoring

## Requirements

```python
torch
torchvision
matplotlib

## Project Structure

├── rbm.py      # RBM model and configuration
├── data.py     # Data loading utilities  
├── run.py      # Training loop and visualization
└── README.md

## Usage
* Clone the repository:
   git clone <repository-url>
   cd rbm-mnist

* Run the training:
   python run.py

The script will:

* Download MNIST dataset automatically
* Train the RBM model
* Save reconstruction visualizations every 5 epochs
* Plot and save the training curve

## Configuration

You can modify the RBM hyperparameters in __rbm.py__. The default configuration is:

@dataclass
class RBMConfig:
    v_dim: int = 784      # Input dimension (28x28 MNIST images)
    h_dim: int = 256      # Hidden layer dimension  
    num_steps: int = 2    # Number of Gibbs steps (CD-k)
    batch_size: int = 64
    num_epochs: int = 20
    learning_rate: float = 1e-2

## Model Details
The RBM implementation uses:

*Binary visible and hidden units
*Contrastive Divergence (CD) training
* Mean squared error for reconstruction monitoring


## Example Outputs
After training, you should see:
[Generated](https://github.com/alirezaghl/Architectures/blob/main/RBM/results/reconstructions_epoch_20.png)

Reconstruction visualizations showing the original images (top row) and their reconstructions (bottom row)
A training curve showing the decrease in reconstruction error over epochs
