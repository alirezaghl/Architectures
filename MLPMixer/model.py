import torch
import torch.nn as nn
from MLPblock import MLPBlock
from MixerBlock import MixerBlock
from PatchEmbedding import PatchEmbedding

class MLPMixer(nn.Module):
  def __init__(self, image_size, embed_dim, hidden_dim, num_channels, num_classes, patch_size, num_mixer):
    super().__init__()

    self.image_size = image_size
    self.num_channels = num_channels
    num_patches = (self.image_size * self.image_size) // patch_size**2
    self.patch_embedding = PatchEmbedding(patch_size)
    self.patch2embed = nn.Linear(self.num_channels * patch_size * patch_size, embed_dim)
    self.mixer = nn.ModuleList(
        [
            MixerBlock(
                image_size=self.image_size,
                hidden_dim=hidden_dim,
                embed_dim=embed_dim,
                patch_size=patch_size
            )
            for _ in range(num_mixer)
        ]
    )

    self.classifier = nn.Linear(embed_dim, num_classes)

  
  def forward(self, x):
    x = self.patch2embed(self.patch_embedding(x))
    for mixer in self.mixer:
      x = mixer(x)

    x = x.mean(dim=1)
    output = self.classifier(x)

    return output