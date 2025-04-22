from typing import Callable, Tuple, Type

from bachelors_thesis.modeling.aura12 import AURA12
from bachelors_thesis.modeling.aura12 import loss_step as aura12_loss_step
from bachelors_thesis.modeling.siglab import SigLab
from bachelors_thesis.modeling.siglab import loss_step as siglab_loss_step
from bachelors_thesis.modeling.sigloc import SigLoc12, SigLocNolan
from bachelors_thesis.modeling.sigloc import loss_step as sigloc_loss_step


def get_model(model_name: str) -> Tuple[Type, Callable]:
    """
    Get the model and loss function based on the model name.
    :param model_name: Name of the model to get.
    """
    if model_name == "aura12":
        return AURA12, aura12_loss_step
    elif model_name == "sigloc12":
        return SigLoc12, sigloc_loss_step
    elif model_name == "siglab":
        return SigLab, siglab_loss_step
    elif model_name == "sigloc-nolan":
        return SigLocNolan, sigloc_loss_step


