from bachelors_thesis.modeling.datasets.aura12_dataset import AURA12Dataset


def get_dataset(cfg):
    if cfg.model.model_name == "aura12":
        return AURA12Dataset
    else:
        raise ValueError(f"Unknown dataset for: {cfg.model.name}")