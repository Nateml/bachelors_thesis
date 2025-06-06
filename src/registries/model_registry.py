from typing import Callable, Tuple, Type

from src.modeling.old.aura12 import AURA12
from src.modeling.old.aura12 import loss_step as aura12_loss_step
from src.modeling.old.cnngru import CNNGru
from src.modeling.old.cnngru import loss_step as cnngru_loss_step
from src.modeling.old.gru import GruModel
from src.modeling.old.gru import loss_step as gru_loss_step
from src.modeling.old.sigloc import SigLoc12, SigLocNolan
from src.modeling.old.sigloc import loss_step as sigloc_loss_step
from src.modeling.siglabv2.siglab_deepsets import SigLabDeepsets
from src.modeling.siglabv2.siglab_deepsets import (
    loss_step as siglab_deepsets_loss_step,
)
from src.modeling.siglabv2.siglab_nocontext import SigLabNoContext
from src.modeling.siglabv2.siglab_nocontext import (
    loss_step as siglab_nocontext_loss_step,
)
from src.modeling.siglabv2.siglabv2 import SigLabV2
from src.modeling.siglabv2.siglabv2 import loss_step as siglabv2_loss_step


def get_model(model_name: str) -> Tuple[Type, Callable]:
    """
    Get the model and loss function based on the model name.
    :param model_name: Name of the model to get.
    """
    if model_name == "aura12":
        return AURA12, aura12_loss_step
    elif model_name == "sigloc12":
        return SigLoc12, sigloc_loss_step
    elif model_name == "sigloc-nolan":
        return SigLocNolan, sigloc_loss_step
    elif model_name == "gru":
        return GruModel, gru_loss_step
    elif model_name == "cnngru":
        return CNNGru, cnngru_loss_step
    elif model_name == "siglabv2":
        return SigLabV2, siglabv2_loss_step
    elif model_name == "siglab_deepsets":
        return SigLabDeepsets, siglab_deepsets_loss_step
    elif model_name == "siglab_nocontext":
        return SigLabNoContext, siglab_nocontext_loss_step
    else:
        raise ValueError(f"Model {model_name} not found. Available models: "
                         f"aura12, sigloc12, sigloc-nolan, gru, cnngru, siglabv2, siglab_deepsets")


