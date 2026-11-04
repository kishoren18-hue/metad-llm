"""Optional plotting helpers (requires matplotlib)."""

from __future__ import annotations

import numpy as np

from .metrics import reliability_table


def type2_roc(correct, confidence):
    """Points of the type-2 ROC: P(conf >= t | correct) vs P(conf >= t | error)."""
    y = np.asarray(correct).astype(int)
    p = np.asarray(confidence, dtype=float)
    ts = np.unique(p)[::-1]
    hr = [0.0] + [float((p[y == 1] >= t).mean()) for t in ts]
    far = [0.0] + [float((p[y == 0] >= t).mean()) for t in ts]
    return np.array(far), np.array(hr)


def report_figure(models: dict, n_bins: int = 10):
    """models: {name: (correct, confidence)} -> matplotlib Figure with 2 panels."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for name, (y, p) in models.items():
        rows = reliability_table(y, p, n_bins, "quantile")
        ax1.plot([r[0] for r in rows], [r[1] for r in rows], marker="o", label=name)
        far, hr = type2_roc(y, p)
        ax2.plot(far, hr, label=name)
    for ax in (ax1, ax2):
        ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    ax1.set(xlabel="mean confidence", ylabel="accuracy", title="Reliability (calibration)")
    ax2.set(xlabel="P(high conf | error)", ylabel="P(high conf | correct)",
            title="Type-2 ROC (metacognitive sensitivity)")
    fig.tight_layout()
    return fig
