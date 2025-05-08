from typing import Type

from bachelors_thesis.modeling.siglabv2.encoders import CNNGRUEncoder


def get_encoder(encoder_name: str) -> Type:
    """
    Get the encoder and loss function based on the encoder name.
    :param encoder_name: Name of the encoder to get.
    """
    if encoder_name == "cnngru":
        return CNNGRUEncoder  # Replace with actual loss function if needed
    else:
        raise ValueError(f"Encoder {encoder_name} not found. Available encoders: cnngru")