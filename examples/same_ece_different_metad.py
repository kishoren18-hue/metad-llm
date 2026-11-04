"""Two simulated "models" with near-identical calibration error but very different
metacognitive sensitivity.

Both models answer the same 4,000 questions with the same accuracy. Model A's
confidence is computed from a sharp internal signal; model B's comes from a much
noisier one. Each model's confidence is *calibrated against its own signal*
(it equals P(correct | signal), estimated on held-out data), so both get a low
ECE. Only the type-2 measures show that B barely knows when it is wrong.

Run:  python examples/same_ece_different_metad.py
"""

from pathlib import Path

import numpy as np

from metad_llm import assess
from metad_llm.plot import report_figure

rng = np.random.default_rng(7)


def make_model(signal_noise: float, n: int, n_ref: int = 200_000):
    """Return (correct, confidence) for a model whose confidence is calibrated
    to a noisy view of each item's true difficulty."""

    def draw(m):
        ability = rng.normal(0.6, 1.3, m)          # latent log-odds of being right
        correct = (rng.uniform(size=m) < 1 / (1 + np.exp(-ability))).astype(int)
        signal = ability + rng.normal(0, signal_noise, m)
        return correct, signal

    # build the calibration map P(correct | signal) on a large reference sample
    ref_y, ref_s = draw(n_ref)
    edges = np.quantile(ref_s, np.linspace(0, 1, 101))
    bin_acc = np.array([ref_y[(ref_s >= lo) & (ref_s <= hi)].mean()
                        for lo, hi in zip(edges[:-1], edges[1:])])
    y, s = draw(n)
    idx = np.clip(np.searchsorted(edges, s) - 1, 0, 99)
    return y, np.clip(bin_acc[idx], 0.01, 0.99)


models = {
    "Model A (sharp signal)": make_model(signal_noise=0.3, n=4000),
    "Model B (noisy signal)": make_model(signal_noise=2.5, n=4000),
}

for name, (y, p) in models.items():
    print(f"== {name}")
    print(assess(y, p, n_ratings=4, n_boot=200))
    print()

out = Path(__file__).resolve().parent.parent / "docs" / "same_ece_different_metad.png"
out.parent.mkdir(exist_ok=True)
report_figure(models).savefig(out, dpi=150, bbox_inches="tight")
print(f"figure saved to {out}")
