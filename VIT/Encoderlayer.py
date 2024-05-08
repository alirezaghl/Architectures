import torch
import torch.nn as nn
from MLP import MLP

class Encoderlayer(nn.Module):
  def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):

        super().__init__()

        self.norm_1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout)
        self.norm_2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, hidden_dim, dropout=dropout)



  def forward(self, x):
        input = self.norm_1(x)
        x = x + self.attention(input, input, input)[0]
        output = x + self.mlp(self.norm_2(x))
        return output