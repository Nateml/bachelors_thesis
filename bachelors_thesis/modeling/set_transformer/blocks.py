import torch
import torch.nn as nn


class SetAttentionBlock(nn.Module):
    def __init__(self, d, h, ffn_expansion=4, mha_dropout=0):
        """
        Args
        ----
        d: int
            The input dimension.
        h: int
            The number of attention heads.
        """
        super(SetAttentionBlock, self).__init__()

        self.mha = nn.MultiheadAttention(d, h, batch_first=True, dropout=mha_dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_expansion*d),
            nn.GELU(), # GELU activation function, not sure why this is used instead of ReLU
            nn.Linear(ffn_expansion*d, d)
        )
        self.layernorm1 = nn.LayerNorm(d)
        self.layernorm2 = nn.LayerNorm(d)

    def forward(self, x):
        """
        Args
        ----
        x: torch.Tensor, shape (batch_size, num_electrodes, feature_dim)
            The input data.

        Output
        ------
        torch.Tensor, shape (batch_size, num_electrodes, feature_dim)
            The input features enriched by attention.
        """

        # Self attention inside the set
        attn, _ = self.mha(x, x, x)

        # Enrich the input with the attention output 
        # and normalize the result
        x = self.layernorm1(x + attn)

        # Feed forward network on each element seperately
        x = self.layernorm2(x + self.ffn(x))
        return x


class PMA(nn.Module):
    """
    Pooling Multihead Attention (PMA) block.
    """
    def __init__(self, d, h, k):
        """
        Args
        ----
        d: int
            The input dimension.
        h: int
            The number of attention heads.
        k: int
            The number of learnable query vectors.
        """
        super(PMA, self).__init__()

        self.seed = nn.Parameter(torch.randn(k, d))  # (k, d)
        self.mha = nn.MultiheadAttention(d, h, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, Z):
        B, *_ = Z.shape
        S = self.seed.unsqueeze(0).expand(B, -1, -1)  # (B, k, d)
        Y, _ = self.mha(S, Z, Z)  # queries = seeds
        return self.norm(Y + S)  # (B, k, d)
