from typing import List, Optional

from matplotlib import pyplot as plt
import numpy as np
from prettytable import PrettyTable
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn

from bachelors_thesis.registries.preprocessor_registry import get_preprocessor

PRECORDIAL_LEAD_NAMES = [
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]
LEAD_NAMES = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6"
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

    # Change datatype to int
    predictions = predictions.astype(int)
    targets = targets.astype(int)

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

def plot_saliency(
        model: nn.Module,
        signals: torch.Tensor,
        lead_idx: List[int] = [0, 1, 2, 3, 4, 5],
        leads: List[str] = PRECORDIAL_LEAD_NAMES,
        title: str = "Saliency Map",
        time_window: tuple = None,
        smoothing_sigma: int = 3
):
    """
    Plots the saliency map for the given signals and model.

    Args
    ----
    model: nn.Module
        The model used for prediction. It should be a PyTorch model.
    signals: torch.Tensor, shape (C, T)
        The input signals to be analyzed. Only works with a single ECG sample. T is the number of time points,
        and C is the number of channels (leads).
    lead_idx: list of int, default [0, 1, 2, 3, 4, 5]
        The indices of the leads to plot. These index the channels in *signals*. The order of this list shoulld
        correspond with the names in *leads*.
    leads: list of str, default PRECORDIAL_LEAD_NAMES
        The names of the leads corresponding to the lead indices in *lead_idx*.
    title: str, default "Saliency Map"
        The title of the plot.
    time_window: tuple of int, default None
        The time window to be plotted. If None, the entire signal is plotted.
        Should be a tuple of (start, end) indices.
    smoothing_sigma: int, default 3
        The standard deviation for the Gaussian smoothing applied to the saliency map.
    """
    x = signals.unsqueeze(0)  # Add batch dimension
    x.requires_grad = True  # Enable gradient computation

    y = model(x)
    if isinstance(y, tuple): # The tuple normally returns the embeddings firsts
        y = y[1]

    if y.dim() == 3: # Check if there is a batch dimension
        y = y[0]

    if not time_window:
        time_window = (0, signals.shape[-1])

    assert time_window[0] >= 0 and time_window[1] <= signals.shape[-1], "Time window out of bounds"
    assert time_window[0] < time_window[1], "Invalid time window"

    assert len(leads) == len(lead_idx), "The number of lead indices should match the number of lead names"

    num_leads = len(leads)
    fig, axes = plt.subplots(num_leads, 1, figsize=(12, 2.5 * num_leads), sharex=True,
                            gridspec_kw={'hspace': 0.15})
    
    for i, lead in enumerate(lead_idx):
        lead_output = y[lead]
        target_idx = lead_output.argmax()
        target = lead_output[target_idx]
        
        model.zero_grad()
        x.grad = None
        target.backward(retain_graph=True)

        saliency = x.grad.data.abs()[0, lead].cpu().numpy()
        saliency = saliency / saliency.max()
        saliency = saliency[time_window[0]:time_window[1]]
        saliency_smooth = gaussian_filter1d(saliency, sigma=smoothing_sigma)
        saliency_2d = saliency_smooth[np.newaxis, :]

        ecg = x[0, lead].cpu().detach().numpy()
        ecg = ecg[time_window[0]:time_window[1]]
        ecg_scaled = (ecg - ecg.min()) / (ecg.max() - ecg.min())

        ax = axes[i] if num_leads > 1 else axes
        im = ax.imshow(saliency_2d,
                    cmap='inferno',
                    aspect='auto',
                    extent=[0, len(ecg), 0, 1],
                    norm=plt.Normalize(vmin=0.01, vmax=saliency.max()))

        ax.plot(np.arange(len(ecg)), ecg_scaled, color='cyan', linewidth=1.5, linestyle='-', alpha=0.8)

        # Minimal labeling
        ax.set_yticks([])
        ax.set_ylabel(f"Lead {leads[i]}", color='white', rotation=0, labelpad=30, fontsize=10, va='center')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.axhline(y=0, color='white', linewidth=2, alpha=0.8)

        # Clean borders
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Hide x-ticks on all but last subplot
        if i < num_leads - 1:
            ax.set_xticks([])
        else:
            ax.set_xlabel('Time', color='white')

    # Add a single colorbar to the side
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.01)
    cbar.set_label('Saliency', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.get_yticklabels(), color='white')

    # Title for the whole figure
    fig.suptitle(title, color='white', fontsize=18)
    fig.patch.set_facecolor('black')

    #plt.tight_layout(rect=[0, 0, 0.97, 0.90])
    plt.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.1)
    plt.show()

    return fig, axes

def hungarian_predictions(logits):
    if logits.ndim == 2:
        logits = logits[np.newaxis, :, :]
    elif logits.ndim != 3:
        raise ValueError("Logits must be 2D or 3D array")

    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()

    B, N, C = logits.shape

    predictions = np.zeros((B, N), dtype=np.int64)

    for b in range(logits.shape[0]):
        # Compute the optimal assignment of signals to leads
        # Effectively we are choosing a one to one assignment of 
        # signals to classes which maximizes the confidence of the model
        # row_ind is the indices of the signals
        # col_ind is the corresponding indices of the targets
        # so, (row_ind[i], col_ind[i]) is the assignment of signal i to class col_ind[i]
        # since logits is square, row_ind corresponds to np.arange(N) (e.g. [0, 1, 2, 3, 4, 5])
        # ideally col_ind should be the same as targets (i.e. [0, 1, 2, 3, 4, 5] as well)
        row_ind, col_ind = linear_sum_assignment(logits[b], maximize=True)

        # Assign the predicted class to the corresponding signal
        predictions[b, row_ind] = col_ind

    return predictions


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
