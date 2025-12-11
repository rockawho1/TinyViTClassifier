import torch
import torch.nn as nn

class PositionEmbeds(nn.Module):
    def __init__(self, num_patches: int, embed_dim: int):
        super().__init__()
        # Define the positional embeddings as a parameter (learnable)
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # +1 for cls token

        # Initialize its weights using a truncated normal distribution
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x: torch.Tensor):
        # Add the positional embeddings to the patch embeddings
        return x + self.pos_emb