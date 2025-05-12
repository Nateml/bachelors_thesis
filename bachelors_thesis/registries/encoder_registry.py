from typing import Type

from bachelors_thesis.modeling.siglabv2.encoders import (
    CNNGRUEncoder,
    GruAttentionEncoder,
    GruEncoder,
    InceptionEncoder,
    LSTMAttentionEncoder,
    LSTMEncoder,
    SimpleCNNEncoder,
    SimpleCNNGRUEncoder,
)


def get_encoder(encoder_name: str) -> Type:
    """
    Get the encoder and loss function based on the encoder name.
    :param encoder_name: Name of the encoder to get.
    """
    if encoder_name == "cnngru":
        return CNNGRUEncoder  # Replace with actual loss function if needed
    elif encoder_name == "simplecnn":
        return SimpleCNNEncoder
    elif encoder_name == "gru":
        return GruEncoder
    elif encoder_name == "gruattention":
        return GruAttentionEncoder
    elif encoder_name == "simplecnn_gru":
        return SimpleCNNGRUEncoder
    elif encoder_name == "inception":
        return InceptionEncoder
    elif encoder_name == "lstm":
        return LSTMEncoder
    elif encoder_name == "lstmattention":
        return LSTMAttentionEncoder
    else:
        raise ValueError(f"Encoder {encoder_name} not found. Available encoders: cnngru, simplecnn, gru, gruattention, simplecnn_gru, inception.")