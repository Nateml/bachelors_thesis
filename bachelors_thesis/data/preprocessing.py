from loguru import logger as log
import numpy as np
from scipy.signal import butter, lfilter, medfilt
from scipy.signal import resample as _resample


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

def highpass(data, *, sampling_rate, order=4, cutoff=0.5, logger=None, **kwargs):
    """
    Apply a butterworth highpass filter to signals.

    data: np.ndarray

    Accepts data of shape:
        - (T,) for a 1D signal
        - (N, T) for multiple signals
        - (N, T, C) for multiple signals with multiple channels
    """
    if logger:
        logger.info(f"Applying highpass filter with {order} order and cutoff {cutoff} Hz")

    b, a = butter(order, cutoff, btype='highpass', fs=sampling_rate)

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

def median_filter(data, *, sampling_rate, kernel_ms1=200, kernel_ms2=600, **kwargs):
    k1 = int(np.round(kernel_ms1 * sampling_rate / 1000.0))
    # Make sure kernel size is odd
    if k1 % 2 == 0:
        k1 += 1
    k2 = int(np.round(kernel_ms2 * sampling_rate / 1000.0))
    if k2 % 2 == 0:
        k2 += 1

    if data.ndim == 1:
        baseline1 = medfilt(data, kernel_size=k1)
        baseline2 = medfilt(baseline1, kernel_size=k2)
        filtered = data - baseline2
        return filtered

    elif data.ndim == 2:
        # Shape (N, T)
        return np.array([median_filter(sig, sampling_rate=sampling_rate) for sig in data])

    elif data.ndim == 3:
        # Shape (N, T, C)
        N, T, C = data.shape

        if C > T:
            log.warning("Preprocessing warning: Number of channels is greater than number of samples." \
            "Are you sure the data is in the right shape?")

        reshaped = data.transpose(0, 2, 1).reshape(-1, T)
        filtered = np.array([median_filter(sig, sampling_rate=sampling_rate) for sig in reshaped])
        return filtered.reshape(N, C, T).transpose(0, 2, 1)

    else:
        raise ValueError(f"Unsupported data shape: {data.shape}. Expected 1D, 2D, or 3D array.")


def z_score(data, **kwargs):
    """
    Apply a z-score normalization to signals.

    data: np.ndarray

    Accepts data of shape:
        - (N, T, C) for multiple signals with multiple channels
    """
    data = np.asarray(data)

    # Shape (N, T, C)
    N, T, C = data.shape

    if C > T:
        log.warning("Preprocessing warning: Number of channels is greater than number of samples." \
        "Are you sure the data is in the right shape?")

    reshaped = data.transpose(0, 2, 1).reshape(-1, T)
    normalized = np.array([(sig - np.mean(sig)) / np.std(sig) for sig in reshaped])
    return normalized.reshape(N, C, T).transpose(0, 2, 1)


def resample(data, sampling_rate, target_rate):
    """
    Resample the data to the target sampling rate.

    data: np.ndarray

    Accepts data of shape:
        - (N, T, C) for multiple signals with multiple channels
    """
    data = np.asarray(data)

    # Shape (N, T, C)
    N, T, C = data.shape

    if C > T:
        log.warning("Preprocessing warning: Number of channels is greater than number of samples." \
        "Are you sure the data is in the right shape?")


    num = int(T * target_rate / sampling_rate)
    resampled = np.array(
        [_resample(sig, num, axis=1) for sig in data.transpose(0, 2, 1)]
    )

    return resampled.transpose(0, 2, 1)  # (N, C, T) -> (N, T, C)
