import torch
import torch.nn as nn

class VIT(nn.Module):

    def __init__(self, image_dim, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, dropout=0.0):

        super().__init__()
        
        self.patch_size = patch_size  
        self.image_size = image_dim
        num_patches = (self.image_size * self.image_size) // self.patch_size**2
        self.image_size = image_dim
        self.patch2embed = nn.Linear(num_channels * self.patch_size * self.patch_size, embed_dim)
        transformer_encoder_list = [
            Encoderlayer(embed_dim, hidden_dim, num_heads, dropout) 
                    for _ in range(num_layers)] 
        self.transformer = nn.Sequential(*transformer_encoder_list)

        self.mlp_head = MLPhead(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1,num_patches+1,embed_dim))
        self.patch_embedding = PatchEmbedding(self.patch_size)
        

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.patch2embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.dropout(x + self.pos_embedding)
        x = self.transformer(x)
        output = self.mlp_head(x[:, 0])

        return output