import numpy as np


def lead_level_accuracy(predictions: np.ndarray = None, logits: np.ndarray = None, targets: np.ndarray = None):
    assert predictions is not None or logits is not None, "Either predictions or logits must be provided."
    assert predictions is None or logits is None, "Only one of predictions or logits can be provided."

    num_samples = logits.shape[0] if logits is not None else predictions.shape[0]
    num_classes = logits.shape[1] if logits is not None else predictions.shape[1]

    if targets is None:
        targets = np.broadcast_to(np.arange(num_classes), (num_samples, num_classes))

    assert targets.shape == (num_samples, num_classes), "Targets must have the same shape as the first two dimensions of logits."

    if logits is not None:
        # -------------------------------
        # Compute predictions from logits
        # -------------------------------
        assert logits.ndim == 3, "Logits must be 3D."
        assert logits.shape[1] == logits.shape[2], "Last two dimensions of logits must be square."

        predictions: np.ndarray = logits.argmax(axis=2)  # (N, C)

    # ------------------------------------
    # Calculate accuracy using predictions
    # ------------------------------------
    assert predictions.ndim == 2, "Predictions must be 2D."

    # predictions is (N, C) and targets is (N, C)
    if targets is None:
        targets = np.broadcast_to(np.arange(num_classes), (num_samples, num_classes))

    assert targets.shape == (num_samples, num_classes), "Targets must have the same shape as the first two dimensions of logits."

    correct: np.ndarray = predictions == targets
    acc = correct.mean()
    return acc


def set_level_accuracy(predictions: np.ndarray = None, logits: np.ndarray = None, targets: np.ndarray = None):
    assert predictions is not None or logits is not None, "Either predictions or logits must be provided."
    assert predictions is None or logits is None, "Only one of predictions or logits can be provided."

    num_samples = logits.shape[0] if logits is not None else predictions.shape[0]
    num_classes = logits.shape[1] if logits is not None else predictions.shape[1]

    if targets is None:
        targets = np.broadcast_to(np.arange(num_classes), (num_samples, num_classes))

    assert targets.shape == (num_samples, num_classes), "Targets must have the same shape as the first two dimensions of logits."

    if predictions is None:
        # -------------------------------
        # Compute predictions from logits
        # -------------------------------
        assert logits.ndim == 3, "Logits must be 3D."
        assert logits.shape[1] == logits.shape[2], "Last two dimensions of logits must be square."

        predictions: np.ndarray = logits.argmax(axis=2)  # (N, C)

    # ------------------------------------
    # Calculate accuracy using predictions
    # ------------------------------------
    assert predictions.ndim == 2, "Predictions must be 2D."

    # predictions is (N, C) and targets is (N, C)
    if targets is None:
        targets = np.broadcast_to(np.arange(num_classes), (num_samples, num_classes))

    assert targets.shape == (num_samples, num_classes), "Targets must have the same shape as the first two dimensions of logits."

    acc = np.mean(np.all(predictions == targets, axis=1))
    return acc

