from loguru import logger as log
import numpy as np
from scipy.signal import butter, lfilter


def bandpass(data, *, sampling_rate, order=4, low=0.4, high=40, **kwargs):
    """
    Apply a butterworth bandpass filter to signals.

    data: np.ndarray

    Accepts data of shape:
        - (T,) for a 1D signal
        - (N, T) for multiple signals
        - (N, T, C) for multiple signals with multiple channels
    """
    b, a = butter(order, [low, high], btype='band', fs=sampling_rate)
    data = np.asarray(data)

    if data.ndim == 1:
        # Shape (T,)
        return lfilter(b, a, data)
    
    elif data.ndim == 2:
        # Shape (N, T)
        return np.array([lfilter(b, a, sig) for sig in data])
    
    elif data.ndim == 3:
        # Shape (N, T, C)
        N, T, C = data.shape

        if C > T:
            log.warning("Preprocessing warning: Number of channels is greater than number of samples." \
            "Are you sure the data is in the right shape?")

        reshaped = data.transpose(0, 2, 1).reshape(-1, T)  # (N*C, T)
        filtered = np.array([lfilter(b, a, sig) for sig in reshaped])
        return filtered.reshape(N, C, T).transpose(0, 2, 1) # (N, T, C)

    else:
        raise ValueError(f"Unsupported data shape: {data.shape}. Expected 1D, 2D, or 3D array.")

def highpass(data, *, sampling_rate, order=4, low=0.5, **kwargs):
    """
    Apply a butterworth highpass filter to signals.

    data: np.ndarray

    Accepts data of shape:
        - (T,) for a 1D signal
        - (N, T) for multiple signals
        - (N, T, C) for multiple signals with multiple channels
    """
    b, a = butter(order, low, btype='highpass', fs=sampling_rate)

    if data.ndim == 1:
        # Shape (T,)
        return lfilter(b, a, data)

    elif data.ndim == 2:
        # Shape (N, T)
        return np.array([lfilter(b, a, sig) for sig in data])
    
    elif data.ndim == 3:
        # Shape (N, T, C)
        N, T, C = data.shape

        if C > T:
            log.warning("Preprocessing warning: Number of channels is greater than number of samples." \
            "Are you sure the data is in the right shape?")

        reshaped = data.transpose(0, 2, 1).reshape(-1, T)
        filtered = np.array([lfilter(b, a, sig) for sig in reshaped])
        return filtered.reshape(N, C, T).transpose(0, 2, 1)

    else:
        raise ValueError(f"Unsupported data shape: {data.shape}. Expected 1D, 2D, or 3D array.")