import os
from typing import List, Tuple

import numpy as np
from src.config import RAW_DATA_DIR
from src.data.load_ptbdata_new import PRECORDIAL_LEADS
import wfdb


def get_ecg_signals_from_file(
        filename: str,
        only_precordial_leads: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Load ECG signals from a WFDB file and return them as a numpy array.

    Args
    ----
    filename: str
        Name of the WFDB file (without extension) to load.
    only_precordial_leads: bool, default True
        If True, only load precordial leads (V1-V6).

    Returns
    -------
    signal: np.ndarray
        ECG signal matrix of shape (N, T), where N is the number of leads and T is length of the signals.
    lead_names: list of str
        List of lead names corresponding to the columns of the signal matrix.
    """

    full_path = os.path.join(RAW_DATA_DIR / "ptb-xl", filename)
    header = wfdb.rdheader(full_path)

    channels = (
        [i for i, lead in enumerate(header.sig_name) if lead in PRECORDIAL_LEADS]
         if only_precordial_leads else None
    )

    signal, _ = wfdb.rdsamp(full_path, channels=channels)
    signal = signal.astype(np.float32)

    lead_names = (
        [header.sig_name[i] for i in channels]
        if channels is not None else header.sig_name
    )

    return signal, lead_names
    