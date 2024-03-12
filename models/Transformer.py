import torch
import torch.nn as nn
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def patchify(data, n_patches):
    N, C, T = data.shape

    patches = torch.zeros(N, n_patches, C * T // n_patches)
    patch_width = T // n_patches

    for idx, single in enumerate(data):
        for i in range(n_patches):
            patch = single[:, i * patch_width: (i+1) * patch_width]
            patches[idx, i] = patch.flatten()

    return patches

def vectorized_patchify(data, n_patches):
    N, C, T = data.shape
    patch_width = T // n_patches

    # Reshape data to split the last dimension into patch_width and n_patches
    # New shape will be [N, C, n_patches, patch_width]
    data_reshaped = data.view(N, C, n_patches, patch_width)

    # Transpose to bring n_patches to the second dimension and then flatten the last two dimensions
    # This effectively turns the patch data into the flattened form you want
    # New shape will be [N, n_patches, C * patch_width]
    patches = data_reshaped.transpose(1, 2).flatten(start_dim=2)

    return patches


def positional_embedding(num_tokens, token_dim):
    result = torch.ones(num_tokens, token_dim)
    for i in range(num_tokens):
        for j in range(token_dim):
            result[i][j] = np.sin(i / (10000 ** (j / token_dim))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / token_dim)))
    return result

def vectorized_positional_embedding(num_tokens, token_dim):
    # Generate a grid of indices for tokens and dimensions
    i = torch.arange(num_tokens).unsqueeze(1)  # Shape: [num_tokens, 1]
    j = torch.arange(token_dim).unsqueeze(0)  # Shape: [1, token_dim]

    # Compute the argument for the sinusoidal functions
    factor = torch.pow(10000, (2 * (j // 2)) / token_dim)
    
    # Use broadcasting to apply the sinusoidal functions across the grid
    embeddings = torch.zeros(num_tokens, token_dim)
    embeddings[:, 0::2] = torch.sin(i / factor[:, 0::2])  # Even indices
    embeddings[:, 1::2] = torch.cos(i / factor[:, 1::2])  # Odd indices

    return embeddings



class ViTforEEG(nn.Module):
    def __init__(self, input_dim=(22, 1000), out_dim=4, n_patches=1000, hidden_dims=64, num_heads=8, ff_dim=64, dropout=0.3, num_layers=2, device='mps'):
        super(ViTforEEG, self).__init__()

        self.input_dim = input_dim
        self.n_patches = n_patches
        self.hidden_dims = hidden_dims
        self.device = device

        self.patch_size = input_dim[0] * input_dim[1] // n_patches

        # Linear Embedding
        self.lin_emb = nn.Linear(self.patch_size, self.hidden_dims)

        # Classification Token (learnable)
        self.cls_token = nn.Parameter(torch.rand(1, self.hidden_dims))

        # Positional Embedding
        self.pos_emb = nn.Parameter(vectorized_positional_embedding(self.n_patches + 1, self.hidden_dims).clone().detach())
        self.pos_emb.requires_grad = False

        encoder_layer = TransformerEncoderLayer(d_model=self.hidden_dims, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dims, out_dim),
            nn.Softmax(dim=-1)
        )

    
    def forward(self, data):
        N, C, T = data.shape
        patches = vectorized_patchify(data, self.n_patches)
        tokens = self.lin_emb(patches.to(self.device))

        # Add Tokens to Token Stack
        tokens = torch.cat((self.cls_token.expand(N, 1, -1), tokens), dim=1)

        # Add Positional Embedding
        out = tokens + self.pos_emb.repeat(N, 1, 1)

        out = self.transformer_encoder(out)

        out = out[:, 0]

        return self.mlp(out)