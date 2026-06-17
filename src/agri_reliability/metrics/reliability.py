from __future__ import annotations

import numpy as np


def absolute_drop(in_domain_score: float, external_score: float) -> float:
    return float(in_domain_score - external_score)


def retention(external_score: float, in_domain_score: float) -> float:
    if in_domain_score == 0:
        return float("nan")
    return float(external_score / in_domain_score)


def cross_dataset_summary(in_domain_score: float, external_score: float) -> dict[str, float]:
    return {
        "in_domain_score": float(in_domain_score),
        "external_score": float(external_score),
        "absolute_drop": absolute_drop(in_domain_score, external_score),
        "retention": retention(external_score, in_domain_score),
    }


def high_confidence_wrong_rate(confidences, correct, threshold: float = 0.9) -> float:
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=int)
    mask = confidences >= threshold
    if not np.any(mask):
        return 0.0
    return float(np.mean(correct[mask] == 0))


def risk_at_coverage(confidences, correct, target_coverage: float) -> float:
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=int)
    if len(confidences) == 0:
        return float("nan")
    n_keep = max(1, int(round(len(confidences) * target_coverage)))
    order = np.argsort(-confidences)[:n_keep]
    return float(np.mean(1 - correct[order]))
