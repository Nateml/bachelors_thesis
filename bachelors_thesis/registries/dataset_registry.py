from bachelors_thesis.modeling.datasets.aura12_dataset import AURA12Dataset
from bachelors_thesis.modeling.datasets.siglab_dataset import SigLabDataset


def get_dataset(cfg):
    if cfg.model.model_name == "aura12":
        return AURA12Dataset
    elif cfg.model.model_name in ["sigloc", "sigloc12", "siglab", "siglab12", "sigloc-nolan", "gru", "cnngru", "siglabv2", "siglab_deepsets", "siglab_nocontext"]:
        return SigLabDataset
    else:
        raise ValueError(f"Unknown dataset for: {cfg.model.model_name}")