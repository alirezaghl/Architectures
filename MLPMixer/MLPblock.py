import torch
import torch.nn as nn

class MLPBlock(nn.Module):
  def __init__(self, embed_dim, hidden_dim):
    super().__init__()

    self.fc1 = nn.Linear(embed_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, embed_dim)
    self.activation = nn.GELU()
    
  
  def forward(self,x):
    return self.fc2(self.activation(self.fc1(x)))