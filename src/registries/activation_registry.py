from torch import nn


def get_activation(name: str):
    """
    Get an activation function by name.
    
    Args:
        name (str): The name of the activation function.
    
    Returns:
        nn.Module: The activation function.
    """

    valid_activations = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "swish": nn.SiLU,
    }

    if name not in valid_activations:
        raise ValueError(f"Activation function '{name}' is not supported. Supported functions are: {', '.join(valid_activations.keys())}")
    
    return valid_activations[name]
