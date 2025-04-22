from bachelors_thesis.modeling.datasets.aura12_dataset import AURA12Dataset
from bachelors_thesis.modeling.datasets.sigloc_dataset import SigLocDataset


def get_dataset(cfg):
    if cfg.model.model_name == "aura12":
        return AURA12Dataset
    elif cfg.model.model_name in ["sigloc", "sigloc12", "siglab", "siglab12", "sigloc-nolan"]:
        return SigLocDataset
    else:
        raise ValueError(f"Unknown dataset for: {cfg.model.model_name}")