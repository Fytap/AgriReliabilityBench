from __future__ import annotations

import numpy as np


def risk_coverage_curve(confidences, correct):
    """Return coverage and risk when retaining samples above descending confidence."""
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=int)
    order = np.argsort(-confidences)
    sorted_correct = correct[order]
    n = len(correct)
    coverage = np.arange(1, n + 1) / n
    cumulative_errors = np.cumsum(1 - sorted_correct)
    risk = cumulative_errors / np.arange(1, n + 1)
    return coverage, risk


def risk_coverage_auc(confidences, correct):
    coverage, risk = risk_coverage_curve(confidences, correct)
    if hasattr(np, "trapezoid"):
        trapezoid = np.trapezoid
    else:
        trapezoid = np.trapz
    return float(trapezoid(risk, coverage))


def selective_accuracy_at_coverage(confidences, correct, target_coverage: float):
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=int)
    n_keep = max(1, int(round(len(confidences) * target_coverage)))
    order = np.argsort(-confidences)[:n_keep]
    return float(correct[order].mean())


def risk_at_coverage(confidences, correct, target_coverage: float):
    return float(1.0 - selective_accuracy_at_coverage(confidences, correct, target_coverage))
