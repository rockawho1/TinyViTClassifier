import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()

        # Make sure that each head can get an equal slice of the embedding
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.embed_dim = embed_dim

        # The dimensionality of each head
        self.head_dim = self.embed_dim // self.num_heads

        # The scaling factor for the qkv matrix --> sqrt(head_dim)
        self.matrix_scale = self.head_dim ** -0.5

        # define qkv linear layer
        # Notes:
        #  - We use a linear layer since this is just a matrix multiplication
        #  - Same layer stores all 3 matrices (that why "*3"). Done for efficiency
        #  - There is a hidden dimension here which splits the data to each head.
        #    For efficiency, we use a single matrix mult instead of many and then
        #    reshape to split for each head.
        self.qkv_proj = nn.Linear(self.embed_dim, self.embed_dim * 3)

        # define output projection.
        # Notes:
        #  - We use a single linear layer (so it's just one big mat mult)
        #  - This lets us combine the information from all heads
        #  - Outputs back an embedding of the original size (for the residual connection)
        self.output_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # Define the dropouts that we will use
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> tuple[torch.Tensor, any]:
        # Batch size, Num patches, Patch embedding size
        B, N, D = x.shape

        # Output is: (B, N, D * 3)
        qkv_mat = self.qkv_proj(x)

        # For each patch split into q,k,v
        qkv_mat = qkv_mat.reshape(B, N, 3, self.num_heads, self.head_dim)

        # Permute so we have q,k,v for each head (qkv, B, num heads, N, head dim)
        qkv_mat = qkv_mat.permute(2, 0, 3, 1, 4)
        q, k, v = qkv_mat[0], qkv_mat[1], qkv_mat[2]

        # Compute attention scores: scaled dot-product attention
        attn_scores = (q @ k.transpose(-2,-1)) * self.matrix_scale

        # Convert to a distribution
        attn = attn_scores.softmax(dim=-1)

        # Apply dropout (regularization / robustness / standard practice)
        attn_dropped = self.attn_drop(attn)

        # Output shape is: (B, heads, N, head_dim)
        output = (attn_dropped @ v)

        # Reshape to get (B, N, D)
        output = output.transpose(1, 2).reshape(B, N, D)

        # Project back to original embedding dim
        output = self.output_proj(output)

        # Apply dropout
        output_dropped = self.proj_drop(output)

        if return_attn:
            # return attention for visualization
            return output_dropped, attn

        return output_dropped, None
