import torch
import torch.nn as nn
from model.PatchEmbed import PatchEmbed
from model.PositionEmbeds import PositionEmbeds
from model.TransformerEncoderBlock import TransformerBlock

class TinyViT(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        embed_dim = config["model"]["embed_dim"]
        img_size = config["model"]["img_size"]
        patch_size = config["model"]["patch_size"]
        in_chans = config["model"]["in_chans"]
        lin_type = config["model"]["lin_type"]
        num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            lin_type=lin_type,
        )

        # cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # pos embedding
        self.pos_embed = PositionEmbeds(num_patches, embed_dim)

        # dropout
        self.dropout = nn.Dropout(config["model"]["drop"])

        # Transformer blocks
        blocks = []
        for _ in range(config["model"]["block_depth"]):
            block = TransformerBlock(
                embed_dim,
                config["model"]["num_heads"],
                config["model"]["mlp_ratio"],
                config["model"]["attn_drop"],
                config["model"]["proj_drop"],
                config["model"]["mlp_drop"]
            )

            blocks.append(block)

        self.layers = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(embed_dim)

        # classification head
        self.head = nn.Linear(embed_dim, config["model"]["num_classes"])
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

        # Debug visualization
        # self.return_attn = config["return_attn"]

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> tuple[torch.Tensor, any]:
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # Embed patches. Output shape: (B, num patches, embed_dim)
        x = self.patch_embed(x)

        # cls_tokens: (B, 1, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # Concat cls tokens
        x = torch.cat((cls_tokens, x), dim=1)

        # Create position embeddings
        x = self.pos_embed(x)

        # Apply dropout
        x = self.dropout(x)

        # Apply blocks
        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x, return_attn)
            if attn is not None:
                attn_maps.append(attn.detach().cpu())

        # Normalize
        x = self.norm(x)

        # CLS -> logits
        cls_output = x[:, 0]
        logits = self.head(cls_output)

        return logits, attn_maps
