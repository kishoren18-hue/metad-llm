"""Standard calibration metrics used in the ML literature, for side-by-side comparison."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

__all__ = ["ece", "adaptive_ece", "brier", "auroc2", "overconfidence", "reliability_table"]


def _arrays(correct, confidence):
    y = np.asarray(correct, dtype=float).ravel()
    p = np.asarray(confidence, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError("correct and confidence must have the same length")
    if y.size == 0:
        raise ValueError("empty input")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("confidence must lie in [0, 1]")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("correct must be 0/1 or boolean")
    return y, p


def reliability_table(correct, confidence, n_bins: int = 10, strategy: str = "uniform"):
    """Per-bin (mean confidence, accuracy, count). strategy: 'uniform' or 'quantile'."""
    y, p = _arrays(correct, confidence)
    if strategy == "uniform":
        edges = np.linspace(0, 1, n_bins + 1)
    elif strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            rows.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return rows


def ece(correct, confidence, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    rows = reliability_table(correct, confidence, n_bins, "uniform")
    n = sum(r[2] for r in rows)
    return float(sum(r[2] * abs(r[0] - r[1]) for r in rows) / n)


def adaptive_ece(correct, confidence, n_bins: int = 10) -> float:
    """ECE with equal-count bins, less sensitive to empty high-confidence bins."""
    rows = reliability_table(correct, confidence, n_bins, "quantile")
    n = sum(r[2] for r in rows)
    return float(sum(r[2] * abs(r[0] - r[1]) for r in rows) / n)


def brier(correct, confidence) -> float:
    y, p = _arrays(correct, confidence)
    return float(np.mean((p - y) ** 2))


def auroc2(correct, confidence) -> float:
    """Type-2 AUROC: P(confidence on a correct trial > confidence on an error).

    Ties count half. Returns nan if all trials are correct or all wrong.
    """
    y, p = _arrays(correct, confidence)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def overconfidence(correct, confidence) -> float:
    """Mean confidence minus accuracy. Positive = overconfident."""
    y, p = _arrays(correct, confidence)
    return float(p.mean() - y.mean())
