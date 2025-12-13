import torch
import torch.nn as nn
from model.MultiHeadSelfAttention import MultiHeadSelfAttention
from model.MLP import MLP

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, attn_drop: float = 0.0, proj_drop: float = 0.0, mlp_drop: float = 0.0):
        super().__init__()

        # layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Attention block
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attn_drop, proj_drop)

        # MLP block
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, hidden_dim, mlp_drop)

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> tuple[torch.Tensor, any]:
        # Self attention with residual
        attn_out, attn = self.attn(self.norm1(x), return_attn)
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x, attn
