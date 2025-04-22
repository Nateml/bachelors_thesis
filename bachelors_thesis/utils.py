from typing import List, Optional

from matplotlib import pyplot as plt
import numpy as np
from bachelors_thesis.registries.preprocessor_registry import get_preprocessor

PRECORDIAL_LEAD_NAMES = [
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]

def _plot_traces(
        signal: np.ndarray,
        lead_names: List[str],
        sampling_rate: int, 
        title: str,
        predictions: Optional[np.ndarray] = None,
        ax: Optional[plt.Axes] = None
) -> plt.Axes:
    n_samples, n_ch = signal.shape

    if ax is None:
        fig_h = max(3, n_ch * 1.5)
        _, ax = plt.subplots(figsize=(12, fig_h))

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
    ax.set_ylabel("Leads")

    if predictions is not None:
        assert len(predictions) == n_ch, "Predictions length must match number of leads."

        # Extend the visible x-range to make room for the text
        extra = int(n_samples * 0.15)
        ax.set_xlim(0, n_samples + extra)
        x_text = n_samples + extra * 0.05

        for i, base in enumerate(baselines):
            pred = str(predictions[i])
            true = lead_names[i]
            txt = f"Predicted: {pred}"
            color = "green" if true == pred else "red"

            ax.text(x_text, base, txt, color=color, fontsize=9, ha="left", va="center",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))

    return ax

def plot_ecg(
        signals: np.ndarray,
        sampling_rate: int =100,
        leads: List[str] = ["V1", "V2", "V3", "V4", "V5", "V6"],
        predictions: Optional[np.ndarray] = None,
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
    predictions : Sequence[str|int], optional
        Per-lead model predictions (same ordering as *leads*).
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
        predictions=predictions,
        ax=ax
    )

def confusion_matrix(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Computes the confusion matrix for the given predictions and targets.

    Args
    ----
    predictions: np.ndarray, shape (N,)
        Predicted labels.
    targets: np.ndarray, shape (N,)
        True labels.

    Returns
    -------
    np.ndarray, shape (C, C)
        Confusion matrix of shape (C, C), where C is the number of classes.
        Access the value at (i, j) to get the number of samples with true label i
        that were predicted as j.
    """
    
    assert predictions.shape == targets.shape, "Predictions and targets must have the same shape."
    assert predictions.ndim == 2, "Predictions and targets must be 2D arrays."

    n_classes = np.max(targets) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for p, t in zip(predictions, targets):
        cm[t, p] += 1

    return cm

def apply_preprocessors(data: np.ndarray, sampling_rate: int, preprocessors: list):
    """
    Applies a list of preprocessing functions to the data.

    Args
    ----
    data: np.ndarray, shape (N, T, C)
        Input data to be preprocessed. N is the number of samples, 
        T is the number of time points, and C is the number of channels.
    sampling_rate: int
        Sampling frequency of the data in Hz.
    preprocessors: list of dict
        List of preprocessor configuration objects. Each object should have a field
        *_preprocessor_* that contains the name of the preprocessor function to be fetched
        using the preprocessor_registry. The rest of the fields are passed as keyword
        arguments to the preprocessor function.

    Returns
    -------
    np.ndarray, shape (N, T, C)
        Copy of *data* with the preprocessor functions applied.
    """
    data_filtered = data.copy()
    for prep_param in preprocessors:
        prep = get_preprocessor(prep_param._preprocessor_)
        data_filtered = prep(data_filtered, sampling_rate=sampling_rate, **prep_param)
    return data_filtered