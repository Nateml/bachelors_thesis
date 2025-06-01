import torch
import torch.nn as nn


class SetAttentionBlock(nn.Module):
    def __init__(self, d, h, ffn_expansion=4, mha_dropout=0, activation=nn.GELU()):
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
            activation,
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


class InducedSetAttentionBlock(nn.Module):
    def __init__(self, d, h, m, ffn_expansion=4, mha_dropout=0):
        """
        Args
        ----
        d: int
            The input dimension.
        h: int
            The number of attention heads.
        """
        super(InducedSetAttentionBlock, self).__init__()
        
        # Learnable incuding points (m x d)
        self.I = nn.Parameter(torch.randn(m, d))

        # Two distinct MHA layers
        # 1) induce: I attends to X
        # 2) project: X attends to induced output
        self.mha1 = nn.MultiheadAttention(d, h, batch_first=True, dropout=mha_dropout)
        self.mha2 = nn.MultiheadAttention(d, h, batch_first=True, dropout=mha_dropout)

        # Feed forward
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_expansion*d),
            nn.GELU(), # GELU activation function, not sure why this is used instead of ReLU
            nn.Linear(ffn_expansion*d, d)
        )
        
        # Layernorm
        self.layernorm1 = nn.LayerNorm(d)
        self.layernorm2 = nn.LayerNorm(d)
        self.layernorm3 = nn.LayerNorm(d)

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

        B, N, d = x.shape

        # 1) Induce
        I = self.I.unsqueeze(0).expand(B, -1, -1)  # (B, m, d)  # noqa: E741
        H, _ = self.mha1(I, x, x)  # (B, m, d)
        H = self.layernorm1(H + I)

        # 2) Project
        Z, _ = self.mha2(x, H, H)  # (B, N, d)
        Z = self.layernorm2(Z + x)

        # 3) Feed forward
        Y = self.layernorm3(Z + self.ffn(Z))

        return Y


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
