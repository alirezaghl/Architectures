import torch
import torch.nn as nn
from MLPblock import MLPBlock

class MixerBlock(nn.Module):
  def __init__(self, image_size, embed_dim, hidden_dim, patch_size):
    super().__init__()

    self.image_size = image_size
    self.patch_size = patch_size
    num_patches = (self.image_size * self.image_size) // self.patch_size**2
    self.layernorm1 = nn.LayerNorm(embed_dim)
    self.layernorm2 = nn.LayerNorm(embed_dim)
    self.token_mlp = MLPBlock(num_patches, hidden_dim)
    self.channel_mlp = MLPBlock(embed_dim, hidden_dim)
    
  def forward(self,x):
    y = self.layernorm1(x)
    y = y.permute(0,2,1)
    y = self.token_mlp(y)
    y = y.permute(0,2,1)
    x = x + y
    y = self.layernorm2(x)
    output = x + self.channel_mlp(y)

    return output