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

    data_reshaped = data.view(N, C, n_patches, patch_width)
    patches = data_reshaped.transpose(1, 2).flatten(start_dim=2)

    return patches


def positional_embedding(num_tokens, token_dim):
    embedding = torch.ones(num_tokens, token_dim)
    for i in range(num_tokens):
        for j in range(token_dim):
            embedding[i][j] = np.sin(i / (10000 ** (j / token_dim))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / token_dim)))
    return embedding

def vectorized_positional_embedding(num_tokens, token_dim):
    # Indices
    i = torch.arange(num_tokens).unsqueeze(1)  # (num_tokens, 1)
    j = torch.arange(token_dim).unsqueeze(0)  # (1, token_dim)

    factor = torch.pow(10000, (2 * (j // 2)) / token_dim)
    
    embedding = torch.zeros(num_tokens, token_dim)
    embedding[:, 0::2] = torch.sin(i / factor[:, 0::2])
    embedding[:, 1::2] = torch.cos(i / factor[:, 1::2])

    return embedding


class ViTforEEG(nn.Module):
    """
    Vision Transformer tailored for our EEG data.

    Architecture
    - Input: (N, C, T), C (num of channels), T (num of time steps)
    - Patched along T axis
        - Linear - Positional Embedding - Transformer Encoder x N - Linear
    """

    def __init__(self, input_dim=(22, 1000), out_dim=4, n_patches=500, hidden_dims=64, num_heads=8, ff_dim=64, dropout=0.5, num_layers=2, device='mps'):
        """
        Initializes the ViTforEEG model with specified configurations.

        Parameters:
        - input_dim (tuple): Dimensions of the input data. (number of EEG channels (C), num time steps (T)). Default is (22, 1000).
        - out_dim (int): Number of output classes. Default is 4.
        - n_patches (int): Number of patches the input time series is divided into. Default is 500.
        - hidden_dims (int): Dimension of the hidden layers in the Transformer Encoder. Default is 64.
        - num_heads (int): Number of attention heads in the Transformer Encoder. Default is 8.
        - ff_dim (int): Dimension of the feedforward network within the Transformer Encoder. Default is 64.
        - dropout (float): Dropout rate in the Transformer Encoder. Default is 0.5.
        - num_layers (int): Number of Transformer Encoder layers. Default is 2.
        - device (str): Computing device for model execution. Default is 'mps'.
        """
        
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
        # self.register_buffer('pos_emb', vectorized_positional_embedding(self.n_patches + 1, self.hidden_dims), persistent=False)
        self.pos_emb = nn.Parameter(vectorized_positional_embedding(self.n_patches + 1, self.hidden_dims).clone().detach())
        self.pos_emb.requires_grad = False

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(d_model=self.hidden_dims, nhead=num_heads, dim_feedforward=ff_dim, 
                                                dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dims, out_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, data):
        N, C, T = data.shape
        # patches = vectorized_patchify(data, self.n_patches).to(self.pos_emb.device)
        # tokens = self.lin_emb(patches)
        patches = vectorized_patchify(data, self.n_patches)
        tokens = self.lin_emb(patches.to(self.device))

        # Add Tokens to Token Stack
        tokens = torch.cat((self.cls_token.expand(N, 1, -1), tokens), dim=1)

        # Add Positional Embedding
        tokenized_input = tokens + self.pos_emb.repeat(N, 1, 1)

        # Transformer
        transformer_out = self.transformer_encoder(tokenized_input)[:, 0]

        return self.mlp(transformer_out)
    

class ConvViTforEEG(nn.Module):
    def __init__(self, input_dim=(22, 1000), out_dim=4, hidden_dims=64, num_heads=8, ff_dim=64, dropout=0.5, num_layers=2, device='mps'):
        super(ConvViTforEEG, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.device = device

        # Convolution
        filter_num1 = 20
        conv_out_dim = (40, 250)

        self.conv = nn.Sequential(
            nn.Conv2d(1, filter_num1, kernel_size=(1, 64), padding=(0, 32)),
            nn.BatchNorm2d(filter_num1),
            nn.ELU(),
            nn.Conv2d(filter_num1, conv_out_dim[0], kernel_size=(22, 1), groups=10),
            nn.BatchNorm2d(conv_out_dim[0]),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=0.3)
        )

        # Calculate Number of Patches and Patch Size
        t = torch.rand(2, 1, *input_dim)
        t = self.conv(t)
        self.n_patches = (t.shape[3] + 1) // 2
        self.patch_size = conv_out_dim[0] * 2

        # Linear Embedding
        self.lin_emb = nn.Linear(self.patch_size, self.hidden_dims)

        # Classification Token (learnable)
        self.cls_token = nn.Parameter(torch.rand(1, self.hidden_dims))

        # Positional Embedding
        self.pos_emb = nn.Parameter(vectorized_positional_embedding(self.n_patches + 1, self.hidden_dims).clone().detach())
        self.pos_emb.requires_grad = False

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(d_model=self.hidden_dims, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dims, out_dim),
            nn.Softmax(dim=-1)
        )

    
    def forward(self, data):
        N, C, T = data.shape

        data = data.unsqueeze(1).to(self.device)
        data = self.conv(data)
        if data.shape[3] % 2 == 1:
            data = nn.functional.pad(data, (0, 1), "constant", 0)
        data = data.squeeze(2)

        patches = vectorized_patchify(data, self.n_patches)
        tokens = self.lin_emb(patches.to(self.device))

        # Add Tokens to Token Stack
        tokens = torch.cat((self.cls_token.expand(N, 1, -1), tokens), dim=1)

        # Add Positional Embedding
        tokenized_input = tokens + self.pos_emb.repeat(N, 1, 1)

        # Transformer
        transformer_out = self.transformer_encoder(tokenized_input)[:, 0]

        return self.mlp(transformer_out)
    


