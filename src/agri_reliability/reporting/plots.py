from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_reliability_diagram(confidences, correct, out_path, n_bins=15, title='Reliability diagram'):
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if np.any(mask):
            xs.append(confidences[mask].mean())
            ys.append(correct[mask].mean())
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(xs, ys, marker='o')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_risk_coverage(coverage, risk, out_path, title='Risk--coverage curve'):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.plot(coverage, risk)
    plt.xlabel('Coverage')
    plt.ylabel('Risk')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
