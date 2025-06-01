from omegaconf import DictConfig, OmegaConf
from src.registries.activation_registry import get_activation
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNNEncoder(nn.Module):

    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.feature_dim = cfg.feature_dim

        layers = []
        for i in range(len(cfg.encoder.in_channels)):
            layers.append(
                nn.Conv1d(
                    in_channels=cfg.encoder.in_channels[i],
                    out_channels=cfg.encoder.out_channels[i],
                    kernel_size=cfg.encoder.kernel_sizes[i],
                    stride=cfg.encoder.strides[i],
                    padding=cfg.encoder.paddings[i]
                )
            )
            layers.append(nn.BatchNorm1d(cfg.encoder.out_channels[i]))
            layers.append(nn.ReLU())
            # Do not dropout on the last layer
            if i < len(cfg.encoder.in_channels) - 1:
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
        self.input_features = cfg.encoder.input_features
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

        # Reshape back to (B, N, feature_dim)
        out = out.view(B, N, -1)

        return out

class GruAttentionEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.feature_dim = cfg.feature_dim
        self.input_features = cfg.encoder.input_features
        self.hidden_size = cfg.encoder.hidden_size
        self.num_layers = cfg.encoder.num_layers
        self.dropout = cfg.encoder.dropout
        self.bidirectional = cfg.encoder.bidirectional
        self.attn_dim = cfg.encoder.attn_dim
        self.num_directions = 2 if self.bidirectional else 1

        self.gru = nn.GRU(
            input_size=self.input_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional
        )

        # Output size is doubled if bidirectional
        self.gru_output_size = self.hidden_size * self.num_directions

        # Attention
        self.attn_proj = nn.Linear(self.gru_output_size, self.attn_dim, bias=False)
        self.attn_vec = nn.Linear(self.attn_dim, 1, bias=False)

        # Fully connected layer to project the GRU output to the desired feature dimension
        self.fc = nn.Sequential(
            nn.Linear(self.gru_output_size, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(OmegaConf.select(cfg, "encoder.final_dropout", default=0.0))
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

        h_seq, _ = self.gru(x)

        # Compute attention scores
        e = torch.tanh(self.attn_proj(h_seq))  # (B * N, attn_dim)
        scores = self.attn_vec(e).squeeze(-1)
        alpha = F.softmax(scores, dim=1)

        # The context vector is the weighted sum of the GRU outputs over time
        # alpha.unsqueeze(1) is (B * N, 1, T)
        # out is (B * N, T, H)
        context = torch.bmm(alpha.unsqueeze(1), h_seq).squeeze(1)  # (B * N, H)

        # Project to the desired feature dimension
        out = self.fc(context)  # (B * N, feature_dim)

        # Reshape back to (B, N, feature_dim)
        out = out.view(B, N, -1)

        return out

class LSTMEncoder(nn.Module):
    """
    LSTM Encoder for ECG signals.
    """

    def __init__(self, cfg: DictConfig):
        super(LSTMEncoder, self).__init__()

        self.feature_dim = cfg.feature_dim
        self.input_features = cfg.encoder.input_features
        self.hidden_size = cfg.encoder.hidden_size
        self.num_layers = cfg.encoder.num_layers
        self.dropout = cfg.encoder.dropout
        self.final_dropout = cfg.encoder.final_dropout
        self.bidirectional = cfg.encoder.bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=self.input_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional
        )

        # Output size is doubled if bidirectional
        self.lstm_output_size = self.hidden_size * self.num_directions

        # Fully connected layer to project the LSTM output to the desired feature dimension
        self.fc = nn.Sequential(
            nn.Linear(self.lstm_output_size, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(self.final_dropout)
        )

    def forward(self, x):
        # x is of shape (B, num_electrodes, T):
        # B: batch size, num_electrodes: 6, T: length of the signal
        # Reshape to (B * num_electrodes, T, feature_dim) for LSTM
        # Each electrode signal is processed independently
        # and in parallel

        assert x.dim() == 3, "Input must be of shape (B, N, T)"

        B, N, T = x.shape
        # Flatten to (B * N, T) and add a channel dimension
        x = x.view(B * N, T, -1)

        out, _ = self.lstm(x)

        # Take the last output of the LSTM
        out = out[:, -1, :]
        # out: (B * N, output_size)

        # Project to the desired feature dimension
        out = self.fc(out)  # (B * N, feature_dim)

        # Reshape back to (B, N, output_size)
        out = out.view(B, N, -1)

        return out

class LSTMAttentionEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.feature_dim = cfg.feature_dim
        self.input_features = cfg.encoder.input_features
        self.hidden_size = cfg.encoder.hidden_size
        self.num_layers = cfg.encoder.num_layers
        self.dropout = cfg.encoder.dropout
        self.bidirectional = cfg.encoder.bidirectional
        self.attn_dim = cfg.encoder.attn_dim
        self.num_directions = 2 if self.bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=self.input_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional
        )

        # Output size is doubled if bidirectional
        self.lstm_output_size = self.hidden_size * self.num_directions

        # Attention
        self.attn_proj = nn.Linear(self.lstm_output_size, self.attn_dim, bias=False)
        self.attn_vec = nn.Linear(self.attn_dim, 1, bias=False)

        # Fully connected layer to project the LSTM output to the desired feature dimension
        self.fc = nn.Sequential(
            nn.Linear(self.lstm_output_size, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(OmegaConf.select(cfg, "encoder.final_dropout", default=0.0))
        )

    def forward(self, x):
        # x is of shape (B, num_electrodes, T):
        # B: batch size, num_electrodes: 6, T: length of the signal
        # Reshape to (B * num_electrodes, T, feature_dim) for LSTM
        # Each electrode signal is processed independently
        # and in parallel

        assert x.dim() == 3, "Input must be of shape (B, N, T)"

        B, N, T = x.shape
        # Flatten to (B * N, T) and add a channel dimension
        x = x.view(B * N, T, -1)

        h_seq, _ = self.lstm(x)

        # Compute attention scores
        e = torch.tanh(self.attn_proj(h_seq))  # (B * N, attn_dim)
        scores = self.attn_vec(e).squeeze(-1)
        alpha = F.softmax(scores, dim=1)

        # The context vector is the weighted sum of the LSTM outputs over time
        # alpha.unsqueeze(1) is (B * N, 1, T)
        # out is (B * N, T, H)
        context = torch.bmm(alpha.unsqueeze(1), h_seq).squeeze(1)  # (B * N, H)

        # Project to the desired feature dimension
        out = self.fc(context)  # (B * N, feature_dim)

        # Reshape back to (B, N, feature_dim)
        out = out.view(B, N, -1)

        return out

class SimpleCNNGRUEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(SimpleCNNGRUEncoder, self).__init__()

        self.feature_dim = cfg.feature_dim
        self.gru_num_directions = 2 if cfg.encoder.gru.bidirectional else 1
        self.gru_output_size = cfg.encoder.gru.hidden_size * self.gru_num_directions

        # CNN
        cnn_layers = []
        for i in range(len(cfg.encoder.cnn.in_channels)):
            cnn_layers.append(
                nn.Conv1d(
                    in_channels=cfg.encoder.cnn.in_channels[i],
                    out_channels=cfg.encoder.cnn.out_channels[i],
                    kernel_size=cfg.encoder.cnn.kernel_sizes[i],
                    stride=cfg.encoder.cnn.strides[i],
                    padding=cfg.encoder.cnn.paddings[i]
                )
            )
            cnn_layers.append(nn.BatchNorm1d(cfg.encoder.cnn.out_channels[i]))
            cnn_layers.append(nn.ReLU())

            # Max pool
            cnn_layers.append(nn.MaxPool1d(kernel_size=2))

            # Do not dropout on the last layer
            if i < len(cfg.encoder.cnn.in_channels) - 1:
                if cfg.encoder.cnn.light_dropout > 0:
                    cnn_layers.append(nn.Dropout(cfg.encoder.cnn.light_dropout))

        self.cnn = nn.Sequential(*cnn_layers)

        # GRU
        self.gru = nn.GRU(
            input_size=cfg.encoder.cnn.out_channels[-1],
            hidden_size=cfg.encoder.gru.hidden_size,
            num_layers=cfg.encoder.gru.num_layers,
            batch_first=True,
            dropout=cfg.encoder.gru.dropout if cfg.encoder.gru.num_layers > 1 else 0.0,
            bidirectional=cfg.encoder.gru.bidirectional
        )

        self.attn_proj = nn.Linear(self.gru_output_size, cfg.encoder.gru.attn_dim, bias=False)
        self.attn_vec = nn.Linear(cfg.encoder.gru.attn_dim, 1, bias=False)

        # Fully connected layer to project the GRU output to the desired feature dimension
        self.fc = nn.Sequential(
            nn.Linear(self.gru_output_size, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(OmegaConf.select(cfg, "encoder.gru.final_dropout", default=0.0))
        )

    def forward(self, x):
        # x is of shape (B, num_electrodes, T):
        # B: batch size, num_electrodes: 6, T: length of the signal
        # Reshape to (B * num_electrodes, T, feature_dim) for GRU
        # Each electrode signal is processed independently
        # and in parallel

        assert x.dim() == 3, "Input must be of shape (B, N, T)"

        B, N, T = x.shape

        # Reshape to (B * N, 1, T)
        x = x.view(B * N, 1, T)

        # CNN forward pass
        cnn_out = self.cnn(x)  # (B * N, C, T')

        gru_in = cnn_out.permute(0, 2, 1)  # (B * N, T', C)

        h_seq, _ = self.gru(gru_in)

        # Compute attention scores
        e = torch.tanh(self.attn_proj(h_seq))  # (B * N, attn_dim)
        scores = self.attn_vec(e).squeeze(-1)
        alpha = F.softmax(scores, dim=1)

        # The context vector is the weighted sum of the GRU outputs over time
        # alpha.unsqueeze(1) is (B * N, 1, T)
        # out is (B * N, T, H)
        context = torch.bmm(alpha.unsqueeze(1), h_seq).squeeze(1)  # (B * N, H)

        # Project to the desired feature dimension
        out = self.fc(context)  # (B * N, feature_dim)

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
        self.activation = get_activation(OmegaConf.select(cfg, "encoder.inception_block.activation", default="relu"))()

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
            self.activation
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
            self.activation,
            nn.Conv1d(
                in_channels=bottleneck_channels,
                out_channels=cfg.encoder.inception_block.branch2.out_channels,
                kernel_size=OmegaConf.select(cfg, "encoder.inception_block.branch2.kernel_size", default=5),
                stride=1,
                padding=OmegaConf.select(cfg, "encoder.inception_block.branch2.padding", default=2)
            ),
            nn.BatchNorm1d(cfg.encoder.inception_block.branch2.out_channels),
            self.activation
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
            self.activation,
            nn.Conv1d(
                in_channels=bottleneck_channels,
                out_channels=cfg.encoder.inception_block.branch3.out_channels,
                kernel_size=OmegaConf.select(cfg, "inception_block.branch3.kernel_size", default=11),
                stride=1,
                padding=OmegaConf.select(cfg, "inception_block.branch3.padding", default=5)
            ),
            nn.BatchNorm1d(cfg.encoder.inception_block.branch3.out_channels),
            self.activation
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
            self.activation
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

    class LocalInceptionBlock(nn.Module):
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

            # Branch 4: 17x1 convolution
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
                    out_channels=cfg.encoder.inception_block.branch4.out_channels,
                    kernel_size=OmegaConf.select(cfg, "inception_block.branch4.kernel_size", default=17),
                    stride=1,
                    padding=OmegaConf.select(cfg, "inception_block.branch4.padding", default=8)
                ),
                nn.BatchNorm1d(cfg.encoder.inception_block.branch4.out_channels),
                nn.ReLU(),
            )

            # Branch 5: Max pooling
            self.branch5 = nn.Sequential(
                nn.MaxPool1d(kernel_size=OmegaConf.select(cfg, "inception_block.branch5.maxpool_size", default=3),
                            stride=1,
                            padding=1),
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=cfg.encoder.inception_block.branch5.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.BatchNorm1d(cfg.encoder.inception_block.branch5.out_channels),
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

    def __init__(self, cfg: DictConfig):
        super(InceptionEncoder, self).__init__()

        self.feature_dim = cfg.feature_dim
        layers = [] 
        current_channels = 1
        for i in range(cfg.encoder.num_inception_blocks):
            layers.append(self.LocalInceptionBlock(
                in_channels=current_channels,
                bottleneck_channels=cfg.encoder.bottleneck_channels if i > 0 else 1,
                cfg=cfg
                ))
            
            current_channels = (
                cfg.encoder.inception_block.branch1.out_channels +
                cfg.encoder.inception_block.branch2.out_channels +
                cfg.encoder.inception_block.branch3.out_channels +
                cfg.encoder.inception_block.branch4.out_channels + 
                cfg.encoder.inception_block.branch5.out_channels
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
    Inception CNN + GRU Encoder for ECG signals.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.gru_num_directions = 2 if cfg.encoder.gru.bidirectional else 1
        self.gru_output_size = cfg.encoder.gru.hidden_dim * self.gru_num_directions
        self.feature_dim = cfg.feature_dim

        self.activation = get_activation(OmegaConf.select(cfg, "encoder.activation", default="relu"))()

        # Create the encoder
        encoder_layers = []
        current_channels = 1
        for i in range(cfg.encoder.num_inception_blocks):
            encoder_layers.append(InceptionBlock(
                in_channels=current_channels,
                bottleneck_channels=cfg.encoder.inception_block.bottleneck_channels if i > 0 else 1,
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
            nn.AdaptiveAvgPool1d(1),  # pooling layer
            nn.Flatten(),
            nn.Linear(current_channels, cfg.feature_dim)
        )

        # alpha is a learnable parameter for the skip connection
        self.alpha = nn.Parameter(torch.tensor(-2.0))  # Initialize to -2.0 to make the skip connection negligible at first

        # GRU layer
        self.gru = nn.GRU(
            input_size=current_channels,
            hidden_size=cfg.encoder.gru.hidden_dim,
            num_layers=cfg.encoder.gru.num_layers,
            batch_first=True,
            dropout=cfg.encoder.gru.dropout if cfg.encoder.gru.num_layers > 1 else 0.0,
            bidirectional=cfg.encoder.gru.bidirectional
        )

        self.attn_proj = nn.Linear(self.gru_output_size, cfg.encoder.gru.attn_dim, bias=False)
        self.attn_vec = nn.Linear(cfg.encoder.gru.attn_dim, 1, bias=False)

        # Fully connected layer to project the GRU output to the desired feature dimension
        self.fc = nn.Sequential(
            nn.Linear(self.gru_output_size, self.feature_dim),
            self.activation,
            nn.Dropout(OmegaConf.select(cfg, "encoder.gru.final_dropout", default=0.0))
        )

        self.merge_norm = nn.LayerNorm(cfg.feature_dim)

    def forward(self, x):
        B, N, T = x.shape

        x = x.view(B * N, 1, T)  # Reshape to (B * N, 1, T)

        cnn_out = self.cnn_encoder(x)  # (B * N, C, T')

        gru_in = cnn_out.permute(0, 2, 1)  # (B * N, T', C)

        # Forward pass through GRU
        h_seq, _ = self.gru(gru_in)

        # Compute attention scores
        e = torch.tanh(self.attn_proj(h_seq))  # (B * N, attn_dim)
        scores = self.attn_vec(e).squeeze(-1)
        alpha = F.softmax(scores, dim=1)

        # The context vector is the weighted sum of the GRU outputs over time
        # alpha.unsqueeze(1) is (B * N, 1, T)
        # out is (B * N, T, H)
        context = torch.bmm(alpha.unsqueeze(1), h_seq).squeeze(1)  # (B * N, H)

        # Project to the desired feature dimension
        out = self.fc(context)  # (B * N, feature_dim)

        # Reshape back to (B, N, feature_dim)
        out = out.view(B, N, -1)

        return out
