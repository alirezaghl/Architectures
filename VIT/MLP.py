import torch
import torch.nn as nn

class MLP(nn.Module):
  def __init__(self, embed_dimension, hidden_dimension, dropout=0.0):
    super().__init__()
    
    self.fc1 = nn.Linear(embed_dimension, hidden_dimension)
    self.fc2 = nn.Linear(hidden_dimension, embed_dimension)
    self.activation = nn.GELU()
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    return self.dropout(self.fc2(self.dropout(self.activation(self.fc1(x)))))


  def forward(self, x):
        input = self.norm_1(x)
        x = x + self.attention(input, input, input)[0]
        output = x + self.mlp(self.norm_2(x))
        return output