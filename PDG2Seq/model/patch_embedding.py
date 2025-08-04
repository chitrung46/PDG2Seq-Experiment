import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, stride, in_dim, embed_dim):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.embed_dim = embed_dim
        self.proj = nn.Linear(patch_len * in_dim, embed_dim)

    def forward(self, x):
        # x: (B, T, N, C)
        B, T, N, C = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, N, C, T)
        x = x.reshape(B * N * C, T)  # (B*N*C, T)
        patches = x.unfold(1, self.patch_len, self.stride)  # (B*N*C, num_patches, patch_len)
        patches = patches.permute(0, 2, 1)  # (B*N*C, patch_len, num_patches)
        patches = patches.reshape(B * N, C, self.patch_len, -1)  # (B*N, C, patch_len, num_patches)
        patches = patches.permute(0, 3, 1, 2)  # (B*N, num_patches, C, patch_len)
        patches = patches.reshape(B * N, patches.shape[1], C * self.patch_len)  # (B*N, num_patches, C*patch_len)
        embed = self.proj(patches)  # (B*N, num_patches, embed_dim)
        return embed