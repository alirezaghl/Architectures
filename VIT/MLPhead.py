import torch
import torch.nn as nn

class MLPhead(nn.Module):
  def __init__(self, embed_dimension, num_classes):
    super().__init__()
    
    self.fc = nn.Sequential(
        nn.LayerNorm(embed_dimension),
        nn.Linear(embed_dimension, num_classes)
    )
  
  def forward(self, x):
    return self.fc(x)