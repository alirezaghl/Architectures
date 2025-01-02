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

├── rbm.py      # RBM model and configuration
├── data.py     # Data loading utilities  
├── run.py      # Training loop and visualization
└── README.md

git clone <repository-url>
cd rbm-mnist

python run.py

@dataclass
class RBMConfig:
    v_dim: int = 784      # Input dimension (28x28 MNIST images)
    h_dim: int = 256      # Hidden layer dimension  
    num_steps: int = 2    # Number of Gibbs steps (CD-k)
    batch_size: int = 64
    num_epochs: int = 20
    learning_rate: float = 1e-2
