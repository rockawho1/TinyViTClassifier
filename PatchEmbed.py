import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """
    Class splits the image into patches and embeds them linearly.
    There are two options on how to do the embedding:
        - linear: uses a linear layer
        - conv: uses a convolution layer with a kernel the size of the patch
    """
    def __init__(self, img_size: int = 32, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 64, lin_type: str = "linear") -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.lin_embed_type = lin_type

        # Compute num_patches
        assert self.img_size % patch_size == 0
        num_patches_w = self.img_size // patch_size
        num_patches_h = self.img_size // patch_size
        self.num_patches = num_patches_w * num_patches_h

        # Define the linear projection layer
        assert self.lin_embed_type == "linear" or self.lin_embed_type == "conv"
        if self.lin_embed_type == "linear":
            # In this case the input is: (B, num patches, C * patch size * patch size)
            # and the output is: (B, num_patches, embed dim)
            patch_dim = self.patch_size * self.patch_size * self.in_chans
            self.lin_proj = nn.Linear(
                in_features=patch_dim,
                out_features=self.embed_dim
            )
        elif self.lin_embed_type == "conv":
            # In this case the input is: (B, C, H, W)
            # and the output is: (B, embed dim, H / patch size, W / patch size)
            self.lin_proj = nn.Conv2d(
                in_channels=self.in_chans,
                out_channels=self.embed_dim,
                kernel_size=(self.patch_size, self.patch_size),
                stride=(self.patch_size, self.patch_size)
            )
        else:
            assert False

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        if self.lin_embed_type == "linear":
            # Reshape the image into patches (B, C, H / patch size, patch size, W / patch size, patch size)
            patches = x.reshape(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
            # Reorder to get all patch data in consecutive dims: (B, H / patch size, W / patch size, C, patch size, patch size)
            patches = patches.permute(0, 2, 4, 1, 3, 5)
            # Flatten patches
            patches = patches.reshape(B, self.num_patches, C * self.patch_size * self.patch_size)
            # Project patches to embedding dim
            patches_embeddings = self.lin_proj(patches)
            return patches_embeddings

        if self.lin_embed_type == "conv":
            # Patches shape: (B, embed dim, H / patch size, W / patch size)
            patches_embeddings = self.lin_proj(x)
            # Reshape to (B, embed dim, num patches)
            patches_embeddings = patches_embeddings.flatten(2)
            # Transpose to (B, num patches, embed dim)
            patches_embeddings = patches_embeddings.transpose(1,2)
            return patches_embeddings

        return None
