from loguru import logger
import numpy as np

from bachelors_thesis.config import PROCESSED_DATA_DIR


def main(
        X: np.ndarray,
        Y: np.ndarray
):
    data_train = X[np.where(Y.strat_fold < 9)]
    data_val = X[np.where(Y.strat_fold == 9)]
    data_test = X[np.where(Y.strat_fold == 10)]

    meta_train = Y.iloc[np.where(Y.strat_fold < 9)]
    meta_val = Y.iloc[np.where(Y.strat_fold == 9)]
    meta_test = Y.iloc[np.where(Y.strat_fold == 10)]

    # Save numpy arrays to .npy
    OUTPUT_DIR = PROCESSED_DATA_DIR / "aura12"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "train.npy", data_train)
    np.save(OUTPUT_DIR / "val.npy", data_val)
    np.save(OUTPUT_DIR / "test.npy", data_test)
    np.save(OUTPUT_DIR / "meta_train.npy", meta_train)
    np.save(OUTPUT_DIR / "meta_val.npy", meta_val)
    np.save(OUTPUT_DIR / "meta_test.npy", meta_test)

    logger.success("Data for AURA12 processed and saved successfully.")
