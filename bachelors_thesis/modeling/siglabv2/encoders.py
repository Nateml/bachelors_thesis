from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn


class SimpleCNNEncoder(nn.Module):

    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.feature_dim = cfg.feature_dim

        layers = []
        for i in range(cfg.encoder.num_cnn_layers):
            layers.append(
                nn.Conv1d(
                    in_channels=cfg.encoder.in_channels[i],
                    out_channels=cfg.encoder.out_channels[i],
                    kernel_size=cfg.encoder.kernel_sizes[i],
                    stride=cfg.strides[i],
                    padding=cfg.paddings[i]
                )
            )
            layers.append(nn.BatchNorm1d(cfg.encoder.out_channels[i]))
            layers.append(nn.ReLU())
            # Do not dropout on the last layer
            if i < cfg.encoder.num_cnn_layers - 1:
                if cfg.encoder.light_dropout > 0:
                    layers.append(nn.Dropout(cfg.encoder.light_dropout))

        layers.append(nn.AdaptiveAvgPool1d(1))  # pool down to T=1
        layers.append(nn.Flatten())  # flatten the channels to form feature vector
        if cfg.encoder.heavy_dropout > 0:
            layers.append(nn.Dropout(cfg.encoder.heavy_dropout))  # dropout layer
        layers.append(nn.Linear(cfg.encoder.out_channels[-1], self.feature_dim))  # project to feature dimension
        layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args
        ----
        x: torch.Tensor, shape (B, N, T)
            The input data.
            B: batch size, N: number of electrodes, T: length of the signal
        """
        B, N, T = x.shape

        # Reshape to (B * N, 1, T)
        x = x.view(B * N, 1, T)

        # Forward pass
        x = self.encoder(x)  # (B * N, feature_dim)

        # Reshape back to (B, N, feature_dim)
        x = x.view(B, N, -1)

        return x

class GruEncoder(nn.Module):
    """
    GRU Encoder for ECG signals.
    """

    def __init__(self, cfg: DictConfig):
        super(GruEncoder, self).__init__()

        self.feature_dim = cfg.feature_dim
        self.input_features = cfg.input_features
        self.hidden_size = cfg.encoder.hidden_size
        self.num_layers = cfg.encoder.num_layers
        self.dropout = cfg.encoder.dropout
        self.bidirectional = cfg.encoder.bidirectional

        self.gru = nn.GRU(
            input_size=self.input_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional
        )

        # Output size is doubled if bidirectional
        self.gru_output_size = self.hidden_size * (2 if self.bidirectional else 1)

        # Fully connected layer to project the GRU output to the desired feature dimension
        self.fc = nn.Sequential(
            nn.Linear(self.gru_output_size, self.feature_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x is of shape (B, num_electrodes, T):
        # B: batch size, num_electrodes: 6, T: length of the signal
        # Reshape to (B * num_electrodes, T, feature_dim) for GRU
        # Each electrode signal is processed independently
        # and in parallel

        assert x.dim() == 3, "Input must be of shape (B, N, T)"

        B, N, T = x.shape
        # Flatten to (B * N, T) and add a channel dimension
        x = x.view(B * N, T, -1)

        out, _ = self.gru(x)

        # Take the last output of the GRU
        out = out[:, -1, :]
        # out: (B * N, output_size)

        # Project to the desired feature dimension
        out = self.fc(out)  # (B * N, feature_dim)

        # Reshape back to (B, N, output_size)
        out = out.view(B, N, -1)

        return out

class GRUAttention(nn.Module):
    def __init__(self, hidden_dim, attn_dim):
        super(GRUAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn_dim   = attn_dim

        self.w_q = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.w_k = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.v   = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, H):  # H: (B, T, H)
        q = self.w_q(H[:, -1:, :])  # (B, 1, A)
        k = self.w_k(H)  # (B, T, A)
        e = self.v(torch.tanh(q + k)).squeeze(-1)  # (B, T)
        a = torch.softmax(e, dim=1)  # (B, T)
        c = torch.bmm(a.unsqueeze(1), H).squeeze(1)  # (B, H)
        return c, a  # c: context vector, a: attention weights

class InceptionBlock(nn.Module):
    """
    Performs multiple convolutions in parallel with different kernel sizes.
    Each branch has its own kernel size and output channels.
    The outputs of all branches are concatenated along the channel dimension.
    """
    def __init__(self, in_channels, bottleneck_channels, cfg: DictConfig):
        super().__init__()

        # Create the branches

        # Branch 1: 1x1 convolution
        self.branch1 = nn.Sequential(
            # kernel size 1
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=cfg.encoder.inception_block.branch1.out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            # Instance norm
            nn.BatchNorm1d(cfg.encoder.inception_block.branch1.out_channels),
            nn.ReLU()
        )

        # Branch 2: 5x1 convolution
        self.branch2 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm1d(bottleneck_channels),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=bottleneck_channels,
                out_channels=cfg.encoder.inception_block.branch2.out_channels,
                kernel_size=OmegaConf.select(cfg, "encoder.inception_block.branch2.kernel_size", default=5),
                stride=1,
                padding=OmegaConf.select(cfg, "encoder.inception_block.branch2.padding", default=2)
            ),
            nn.BatchNorm1d(cfg.encoder.inception_block.branch2.out_channels),
            nn.ReLU()
        )

        # Branch 3: 11x1 convolution
        self.branch3 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm1d(bottleneck_channels),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=bottleneck_channels,
                out_channels=cfg.encoder.inception_block.branch3.out_channels,
                kernel_size=OmegaConf.select(cfg, "inception_block.branch3.kernel_size", default=11),
                stride=1,
                padding=OmegaConf.select(cfg, "inception_block.branch3.padding", default=5)
            ),
            nn.BatchNorm1d(cfg.encoder.inception_block.branch3.out_channels),
            nn.ReLU(),
        )

        # Branch 4: Max pooling
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=OmegaConf.select(cfg, "inception_block.branch4.maxpool_size", default=3),
                         stride=1,
                         padding=1),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=cfg.encoder.inception_block.branch4.out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm1d(cfg.encoder.inception_block.branch4.out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # Parallel branches
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        # Concatenate along the channel dimension
        out = torch.cat([b1, b2, b3, b4], dim=1)

        return out

class InceptionEncoder(nn.Module):

    def __init__(self, cfg: DictConfig):
        super(InceptionEncoder, self).__init__()

        self.feature_dim = cfg.feature_dim
        layers = [] 
        current_channels = 1
        for i in range(cfg.encoder.num_inception_blocks):
            layers.append(InceptionBlock(
                in_channels=current_channels,
                bottleneck_channels=cfg.encoder.bottleneck_channels,
                cfg=cfg
                ))
            
            current_channels = (
                cfg.encoder.inception_block.branch1.out_channels +
                cfg.encoder.inception_block.branch2.out_channels +
                cfg.encoder.inception_block.branch3.out_channels +
                cfg.encoder.inception_block.branch4.out_channels
            )

            # Dropout
            if cfg.encoder.dropout > 0:
                layers.append(nn.Dropout(cfg.encoder.dropout))

        # Pooling layer
        layers.append(nn.AdaptiveAvgPool1d(1))  # pool down to T=1
        layers.append(nn.Flatten())  # flatten the channels to form feature vector
        layers.append(nn.Linear(current_channels, self.feature_dim))  # project to feature dimension
        layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args
        ----
        x: torch.Tensor, shape (B, N, T)
            The input data.
            B: batch size, N: number of electrodes, T: length of the signal
        """
        B, N, T = x.shape

        # Reshape to (B * N, 1, T)
        x = x.view(B * N, 1, T)

        # Forward pass
        x = self.encoder(x)

        # (B * N, feature_dim)
        # Reshape back to (B, N, feature_dim)
        x = x.view(B, N, -1)

        return x

class CNNGRUEncoder(nn.Module):
    """
    CNN + GRU Encoder for ECG signals.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()

        # Create the encoder
        encoder_layers = []
        current_channels = 1
        for i in range(cfg.encoder.num_inception_blocks):
            encoder_layers.append(InceptionBlock(
                in_channels=current_channels,
                bottleneck_channels=cfg.encoder.inception_block.bottleneck_channels,
                cfg=cfg
                ))
            
            current_channels = (
                cfg.encoder.inception_block.branch1.out_channels +
                cfg.encoder.inception_block.branch2.out_channels +
                cfg.encoder.inception_block.branch3.out_channels +
                cfg.encoder.inception_block.branch4.out_channels
            )

            # Downsampling layer
            encoder_layers.append(nn.MaxPool1d(kernel_size=2))

        # Small dropout layer
        encoder_layers.append(nn.Dropout(cfg.encoder.cnn_dropout))

        self.cnn_encoder = nn.Sequential(*encoder_layers)

        self.cnn_post_layers = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # pooling layerm
            nn.Flatten(),
            nn.Linear(current_channels, cfg.feature_dim)
        )

        # alpha is a learnable parameter for the skip connection
        self.alpha = nn.Parameter(torch.tensor(-2.0))  # Initialize to -2.0 to make the skip connection negligible at first

        # GRU layer
        self.gru = nn.GRU(
            input_size=current_channels,
            hidden_size=cfg.feature_dim,
            num_layers=cfg.encoder.gru.num_layers,
            batch_first=True,
            dropout=cfg.encoder.gru.dropout if cfg.encoder.gru.num_layers > 1 else 0.0,
            bidirectional=cfg.encoder.gru.bidirectional
        )

        self.gru_projection = nn.Linear(
            cfg.feature_dim * (2 if cfg.encoder.gru.bidirectional else 1),
            cfg.feature_dim
        )

        self.merge_norm = nn.LayerNorm(cfg.feature_dim)

    def forward(self, x):
        B, N, T = x.shape

        x = x.view(B * N, 1, T)  # Reshape to (B * N, 1, T)

        cnn_out = self.cnn_encoder(x)  # (B * N, C, T')

        gru_in = cnn_out.permute(0, 2, 1)  # (B * N, T', C)

        # Forward pass through GRU
        out, _ = self.gru(gru_in)  # (B * N, T', H)
        out = out[:, -1, :]  # Take the last output of the GRU

        # Project to the desired feature dimension
        out = self.gru_projection(out)

        # Reshape back to (B, N, D)
        out = out.view(B, N, -1)

        # Add the cnn output with the GRU output
        # so that the cnn blocks receive enough gradient
        cnn_out = self.cnn_post_layers(cnn_out)  # (B * N, feature_dim)
        # Reshape back to (B, N, feature_dim)
        cnn_out = cnn_out.view(B, N, -1)

        merged = out + torch.sigmoid(self.alpha) * cnn_out  # (B, N, feature_dim)
        merged = self.merge_norm(merged)

        return merged