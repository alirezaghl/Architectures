Restricted Boltzmann Machine (RBM) for MNIST
A PyTorch implementation of a Restricted Boltzmann Machine (RBM) trained on the MNIST dataset using Contrastive Divergence.
Overview
This implementation includes:

A basic RBM model with configurable architecture
CD-k training (default k=1)
MNIST data preprocessing
Training visualization with reconstruction samples
Training progress monitoring

Requirements
Copytorch
torchvision
matplotlib
Install dependencies using:
bashCopypip install torch torchvision matplotlib
Project Structure
Copy.
├── rbm.py         # RBM model and configuration
├── data.py        # Data loading utilities
├── run.py         # Training loop and visualization
└── README.md
Usage

Clone the repository:

bashCopygit clone <repository-url>
cd rbm-mnist

Run the training:

bashCopypython run.py
The script will:

Download MNIST dataset automatically
Train the RBM model
Save reconstruction visualizations every 5 epochs
Plot and save the training curve

Configuration
You can modify the RBM hyperparameters in rbm.py. The default configuration is:
pythonCopy@dataclass
class RBMConfig:
    v_dim: int = 784        # Input dimension (28x28 MNIST images)
    h_dim: int = 256        # Hidden layer dimension
    num_steps: int = 2      # Number of Gibbs steps (CD-k)
    batch_size: int = 64
    num_epochs: int = 20
    learning_rate: float = 1e-2
Output Files
The training process generates:

reconstructions_epoch_X.png: Visualization of original and reconstructed images every 5 epochs
training_curve.png: Plot of reconstruction error over training epochs

Model Details
The RBM implementation uses:

Binary visible and hidden units
Sigmoid activation function
Contrastive Divergence (CD) training
Adam optimizer
Mean squared error for reconstruction monitoring

Example Outputs
After training, you should see:

[generated](https://github.com/alirezaghl/Architectures/blob/main/RBM/results/reconstructions_epoch_20.png)

Reconstruction visualizations showing the original images (top row) and their reconstructions (bottom row)
A training curve showing the decrease in reconstruction error over epochs

Customization
To modify the training process:

Adjust model architecture in rbm.py:

pythonCopyconfig = RBMConfig(
    v_dim=784,
    h_dim=512,  # Change hidden layer size
    num_steps=1  # Change number of Gibbs steps
)

Modify training parameters in run.py:

pythonCopyconfig = RBMConfig(
    num_epochs=20,  # Train for more epochs
    learning_rate=1e-2  # Adjust learning rate
)
