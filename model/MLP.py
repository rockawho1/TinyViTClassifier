import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, drop_prob: float = 0.0):
        super().__init__()
        # Linear layers
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        # GELU
        self.gelu = nn.GELU()
        # Dropout
        self.drop = nn.Dropout(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
        return x
