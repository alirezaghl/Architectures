import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        batch_size, num_channels, height, width = x.shape
        x = x.reshape(batch_size, height//self.patch_size * width//self.patch_size, (self.patch_size**2) * num_channels)
        return x