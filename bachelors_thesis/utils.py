from typing import List, Optional

from matplotlib import pyplot as plt
import numpy as np


def _plot_traces(
        signal: np.ndarray,
        lead_names: List[str],
        sampling_rate: int, 
        title: str,
        ax: Optional[plt.Axes] = None
) -> plt.Axes:
    n_samples, n_ch = signal.shape

    if ax is None:
        fig_h = max(3, n_ch * 1.5)
        fig, ax = plt.subplots(figsize=(12, fig_h))

    # Vertical spacing to avoid overlap
    spacing = max(np.ptp(signal[:, i]) for i in range(n_ch)) * 1.2

    baselines = [(n_ch - 1 - i) * spacing for i in range(n_ch)]

    for i, base in enumerate(baselines):
        ax.plot(signal[:, i] + base)

    ax.set_title(title)
    ax.set_xlabel("Samples")
    ax.set_yticks(baselines)
    ax.set_yticklabels(lead_names)
    ax.set_xlim(0, n_samples)
    ax.grid(True, alpha=0.3)

def plot_ecg(
        signals: np.ndarray,
        sampling_rate: int =100,
        leads: List[str] = ["V1", "V2", "V3", "V4", "V5", "V6"],
        ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plots the ECG signals for the given leads.

    Args
    ----
    signals: np.ndarray, shape (N, T) **or** (T, N)
        ECG signal matrix. If the first dimension equals the number of
        *leads*, we treat the array as (N, T); otherwise, we assume (T, N)
        and transpose.
    sampling_rate: int, default 100
        Sampling frequency in Hz (used for the title / x-axis scale).
    leads: list of str, default ["V1", ..., "V6"]
        Names corresponding to each row of *signals*.
    ax: matplotlib.axes.Axes, optional
        If provided, the traces are drawn onto this axis.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the plot.
    """
    
    if signals.ndim != 2:
        raise ValueError("Signals should be 2D array of shape (N, T)")
    
    n_rows, n_cols = signals.shape
    if n_rows != len(leads):
        if n_cols == len(leads):
            signals = signals.T
            n_rows, n_cols = signals.shape
        else:
            raise ValueError(
                f"Signals shape {signals.shape} does not match leads length {len(leads)}"
            )

    signal = signals.T.astype(np.float32)  # (T, N)

    return _plot_traces(
        signal,
        leads,
        sampling_rate,
        f"ECG signals array | sr={sampling_rate} Hz",
        ax
    )