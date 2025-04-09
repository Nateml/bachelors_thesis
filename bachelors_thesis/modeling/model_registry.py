from typing import Callable, Tuple, Type

import torch.nn as nn

from bachelors_thesis.modeling.aura12 import AURA12, loss_step


def get_model(model_name: str) -> Tuple[Type, Callable]:
    """
    Get the model and loss function based on the model name.
    :param model_name: Name of the model to get.
    """
    if model_name == "aura12":
        return AURA12, loss_step
