from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


def expected_calibration_error(confidences, correct, n_bins: int = 15) -> float:
    """Compute standard ECE for classification confidence.

    Parameters
    ----------
    confidences: array-like, shape [n]
        Max softmax probabilities or detection confidences.
    correct: array-like, shape [n]
        Binary correctness indicators.
    n_bins: int
        Number of equal-width confidence bins.
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if not np.any(mask):
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        ece += mask.mean() * abs(bin_acc - bin_conf)
    return float(ece)


def maximum_calibration_error(confidences, correct, n_bins: int = 15) -> float:
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    gaps = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if np.any(mask):
            gaps.append(abs(correct[mask].mean() - confidences[mask].mean()))
    return float(max(gaps) if gaps else 0.0)


def multiclass_brier_score(probabilities, labels) -> float:
    probs = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n, k = probs.shape
    y = np.zeros((n, k), dtype=float)
    y[np.arange(n), labels] = 1.0
    return float(np.mean(np.sum((probs - y) ** 2, axis=1)))


def error_detection_auroc(confidences, correct) -> float:
    """AUROC for detecting errors with uncertainty score = 1 - confidence."""
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=int)
    errors = 1 - correct
    if len(np.unique(errors)) < 2:
        return float('nan')
    return float(roc_auc_score(errors, 1.0 - confidences))
