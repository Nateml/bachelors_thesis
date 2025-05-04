import torch
import torch.nn as nn
from omegaconf import DictConfig

class SimpleCNN1D(nn.Module):
    """
    Simple 1D CNN for encoding ECG signals.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()

        # Construct layer
        layers = []
        in_channels = cfg.local_encoder.in_channels
        for layer_cfg in cfg.local_encoder.layers:
            layers.append(nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=layer_cfg.out_channels,
                    kernel_size=layer_cfg.kernel_size,
                    stride=layer_cfg.stride,
                    padding=layer_cfg.padding
                ))
            layers.append(nn.ReLU())
            in_channels = layer_cfg.out_channels

        # Pooling Layer
        layers.append(nn.AdaptiveAvgPool1d(1))  # pooling layer
        # Flatten
        layers.append(nn.Flatten())
        # Fully Connected Layyer
        layers.append(nn.Linear(in_channels, cfg.feature_dim))
        layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        # x is of shape (B, num_electrodes, T):
        # B: batch size, num_electrodes: 6, T: length of the signal
        # Reshape to (B * num_electrodes, 1, T) for Conv1d
        # Each electrode signal is processed independently
        # and in parallel

        assert x.dim() == 3, "Input must be of shape (B, N, T)"
        assert x.size(1) == 6, f"Input must have 6 electrodes. Input shape: {x.shape}"

        B, N, T = x.shape
        # Flatten to (B * N, T) and add a channel dimension
        x = x.view(B * N, 1, T)

        out = self.encoder(x)  # (B * N, feature_dim)
        # Reshape back to (B, N, feature_dim)
        out = out.view(B, N, -1)

        return out

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
                out_channels=cfg.inception_block.branch1.out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            # Instance norm
            nn.BatchNorm1d(cfg.inception_block.branch1.out_channels),
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
                out_channels=cfg.inception_block.branch2.out_channels,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm1d(cfg.inception_block.branch2.out_channels),
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
                out_channels=cfg.inception_block.branch3.out_channels,
                kernel_size=11,
                stride=1,
                padding=5
            ),
            nn.BatchNorm1d(cfg.inception_block.branch3.out_channels),
            nn.ReLU(),
        )

        # Branch 4: 41x1 convolution
        self.branch4 = nn.Sequential(
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
                out_channels=cfg.inception_block.branch4.out_channels,
                kernel_size=41,
                stride=1,
                padding=20
            ),
            nn.BatchNorm1d(cfg.inception_block.branch4.out_channels),
            nn.ReLU(),
        )

        # Branch 5: Max pooling
        self.branch5 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=cfg.inception_block.branch5.out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm1d(cfg.inception_block.branch5.out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # Parallel branches
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        b5 = self.branch5(x)

        # Concatenate along the channel dimension
        out = torch.cat([b1, b2, b3, b4, b5], dim=1)

        return out

class InceptionEncoder(nn.Module):

    def __init__(self, cfg: DictConfig):
        super().__init__()

        layers = []
        current_channels = 1
        for i in range(cfg.inception_encoder.num_blocks):
            layers.append(InceptionBlock(
                in_channels=current_channels,
                bottleneck_channels=cfg.inception_encoder.bottleneck_channels,
                cfg=cfg
                ))
            
            current_channels = (
                cfg.inception_block.branch1.out_channels +
                cfg.inception_block.branch2.out_channels +
                cfg.inception_block.branch3.out_channels +
                cfg.inception_block.branch4.out_channels +
                cfg.inception_block.branch5.out_channels
            )

            # Small dropout layer
            layers.append(nn.Dropout(cfg.inception_encoder.dropout))

        # Pooling Layer
        layers.append(nn.AdaptiveAvgPool1d(1))  # pooling layer
        # Flatten
        layers.append(nn.Flatten())
        # Fully Connected Layyer
        layers.append(nn.Linear(current_channels, cfg.feature_dim))
        layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        # x is of shape (B, num_electrodes, T):
        # B: batch size, num_electrodes: 6, T: length of the signal
        # Reshape to (B * num_electrodes, 1, T) for Conv1d
        # Each electrode signal is processed independently
        # and in parallel

        assert x.dim() == 3, "Input must be of shape (B, N, T)"

        B, N, T = x.shape
        # Flatten to (B * N, T) and add a channel dimension
        x = x.view(B * N, 1, T)

        out = self.encoder(x)

        # Reshape back to (B, N, feature_dim)
        out = out.view(B, N, -1)

        return out

class InceptionEncoderWithRes(nn.Module):

    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg

        pre_skip_layers = []
        post_skip_layers = []
        current_channels = 1
        for i in range(cfg.inception_encoder.num_blocks):
            pre_skip_layers.append(InceptionBlock(
                in_channels=current_channels,
                bottleneck_channels=cfg.inception_encoder.bottleneck_channels,
                cfg=cfg
                ))
            
            current_channels = (
                cfg.inception_block.branch1.out_channels +
                cfg.inception_block.branch2.out_channels +
                cfg.inception_block.branch3.out_channels +
                cfg.inception_block.branch4.out_channels +
                cfg.inception_block.branch5.out_channels
            )

            # Small dropout layer
            if i < cfg.inception_encoder.num_blocks - 1:  # don't add dropout after the last block
                pre_skip_layers.append(nn.Dropout(cfg.inception_encoder.dropout))

        # Residual projection layer
        self.residual_proj = nn.Sequential(
            nn.Conv1d(
            in_channels=1,
            out_channels=current_channels,
            kernel_size=1,
            stride=1,
            padding=0
            ),
            nn.BatchNorm1d(current_channels),
            nn.ReLU()
        )
        
        # Pooling Layer
        post_skip_layers.append(nn.AdaptiveAvgPool1d(1))  # pooling layer
        # Flatten
        post_skip_layers.append(nn.Flatten())
        # Fully Connected Layyer
        post_skip_layers.append(nn.Linear(current_channels, cfg.feature_dim))
        post_skip_layers.append(nn.ReLU())

        # self.encoder = nn.Sequential(*layers)
        self.pre_skip = nn.Sequential(*pre_skip_layers)
        self.post_skip = nn.Sequential(*post_skip_layers)


    def forward(self, x):
        # x is of shape (B, num_electrodes, T):
        # B: batch size, num_electrodes: 6, T: length of the signal
        # Reshape to (B * num_electrodes, 1, T) for Conv1d
        # Each electrode signal is processed independently
        # and in parallel

        assert x.dim() == 3, "Input must be of shape (B, N, T)"

        B, N, T = x.shape
        # Flatten to (B * N, T) and add a channel dimension
        x = x.view(B * N, 1, T)

        # Pass through the Inception blocks
        out = self.pre_skip(x)

        # Add skip connection
        # OmegaConf.select() returns None if the key is not found,
        # keeping the model backward compatible with older configs
        residual = self.residual_proj(x)  # (B * N, current_channels, T)

        out = out + residual  # (B * N, current_channels, T)

        # Pass through the post-skip layers
        out = self.post_skip(out)  # (B * N, feature_dim)

        # Reshape back to (B, N, feature_dim)
        out = out.view(B, N, -1)

        return out

class GruEncoder(nn.Module):
    """
    GRU Encoder for ECG signals.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.gru = nn.GRU(
            input_size=cfg.feature_dim,
            hidden_size=cfg.gru.hidden_size,
            num_layers=cfg.gru.num_layers,
            batch_first=True,
            dropout=cfg.gru.dropout if cfg.gru.num_layers > 1 else 0.0,
            bidirectional=cfg.gru.bidirectional
        )

        # Output size is doubled if bidirectional
        self.output_size = cfg.gru.hidden_size * (2 if cfg.gru.bidirectional else 1)

        # Fully connected layer to project the GRU output to the desired feature dimension
        self.fc = nn.Linear(self.output_size, cfg.feature_dim)

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
        # Reshape back to (B, N, output_size)
        out = out.view(B, N, -1)

        return out

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
                bottleneck_channels=cfg.inception_block.bottleneck_channels,
                cfg=cfg
                ))
            
            current_channels = (
                cfg.inception_block.branch1.out_channels +
                cfg.inception_block.branch2.out_channels +
                cfg.inception_block.branch3.out_channels +
                cfg.inception_block.branch4.out_channels +
                cfg.inception_block.branch5.out_channels
            )

            # Downsampling layer
            encoder_layers.append(nn.MaxPool1d(kernel_size=2))

        # Small dropout layer
        encoder_layers.append(nn.Dropout(cfg.encoder.cnn_dropout))

        self.cnn_encoder = nn.Sequential(*encoder_layers)

        # GRU layer
        self.gru = nn.GRU(
            input_size=current_channels,
            hidden_size=cfg.feature_dim,
            num_layers=cfg.gru.num_layers,
            batch_first=True,
            dropout=cfg.gru.dropout if cfg.gru.num_layers > 1 else 0.0,
            bidirectional=cfg.gru.bidirectional
        )

    def forward(self, x):
        B, N, T = x.shape

        x = x.view(B * N, 1, T)  # Reshape to (B * N, 1, T)

        x = self.cnn_encoder(x)  # (B * N, C, T')

        x = x.permute(0, 2, 1)  # (B * N, T', C)

        # Forward pass through GRU
        out, _ = self.gru(x)  # (B * N, T', D)
        # Take the last output of the GRU
        out = out[:, -1, :]  # (B * N, D)

        # Reshape back to (B, N, D)
        out = out.view(B, N, -1)

        return out


