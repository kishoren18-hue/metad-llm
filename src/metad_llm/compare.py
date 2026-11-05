"""Paired comparison of two models (or two confidence methods) on the same items.

The question this answers: "Is model B's metacognition actually better than
model A's, or is the difference noise?" Items are resampled jointly, so each
bootstrap replicate compares the two models on the same questions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import metrics
from .core import bin_confidence, counts_from_correctness
from .sdt import fit_meta_d

__all__ = ["MetricDiff", "Comparison", "compare", "confidence_from_logprobs"]


@dataclass
class MetricDiff:
    name: str
    a: float
    b: float
    diff: float          # b - a
    ci_low: float
    ci_high: float
    p_value: float       # two-sided paired bootstrap
    better: str          # "higher" or "lower"

    @property
    def winner(self) -> str:
        if self.ci_low <= 0 <= self.ci_high:
            return "no clear difference"
        b_up = self.diff > 0
        return "B" if (b_up and self.better == "higher") or (not b_up and self.better == "lower") else "A"


@dataclass
class Comparison:
    n: int
    n_boot: int
    diffs: list[MetricDiff]

    def as_dict(self) -> dict:
        return {"n": self.n, "n_boot": self.n_boot,
                "metrics": [{**d.__dict__, "winner": d.winner} for d in self.diffs]}

    def __str__(self) -> str:
        rows = [f"{self.n} paired items, {self.n_boot} bootstrap resamples",
                f"{'metric':<10}{'A':>8}{'B':>8}{'B - A':>9}   {'95% CI':<18}{'p':>7}   verdict"]
        for d in self.diffs:
            rows.append(f"{d.name:<10}{d.a:>8.3f}{d.b:>8.3f}{d.diff:>+9.3f}   "
                        f"[{d.ci_low:+.3f}, {d.ci_high:+.3f}]  {d.p_value:>6.3f}   {d.winner}")
        return "\n".join(rows)


_BETTER = {"accuracy": "higher", "ece": "lower", "brier": "lower", "auroc2": "higher",
           "meta_d": "higher", "m_ratio": "higher"}


def _scores(y, p, ratings, k, with_metad: bool) -> dict:
    out = {
        "accuracy": float(np.mean(y)),
        "ece": metrics.ece(y, p),
        "brier": metrics.brier(y, p),
        "auroc2": metrics.auroc2(y, p),
    }
    if with_metad:
        try:
            f = fit_meta_d(*counts_from_correctness(y, ratings, k))
            out["meta_d"], out["m_ratio"] = f.meta_d, f.m_ratio
        except ValueError:
            out["meta_d"] = out["m_ratio"] = float("nan")
    return out


def compare(correct_a, conf_a, correct_b, conf_b, n_ratings: int = 4, n_boot: int = 500,
            seed: int = 0, include_meta_d: bool = True) -> Comparison:
    """Paired bootstrap comparison of two runs on the same items (same order).

    Confidence bins are fixed per model on the full data and reused inside the
    bootstrap, so resampling measures item noise, not binning noise.
    """
    ya, pa = np.asarray(correct_a, int), np.asarray(conf_a, float)
    yb, pb = np.asarray(correct_b, int), np.asarray(conf_b, float)
    if not (ya.shape == pa.shape == yb.shape == pb.shape):
        raise ValueError("all four arrays must have the same length (one entry per shared item)")
    if ya.size < 20:
        raise ValueError("need at least 20 paired items")
    ra, ea = bin_confidence(pa, n_ratings)
    rb, eb = bin_confidence(pb, n_ratings)
    ka, kb = len(ea) - 1, len(eb) - 1

    base_a = _scores(ya, pa, ra, ka, include_meta_d)
    base_b = _scores(yb, pb, rb, kb, include_meta_d)
    rng = np.random.default_rng(seed)
    boot = {m: [] for m in base_a}
    n = ya.size
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        sa = _scores(ya[i], pa[i], ra[i], ka, include_meta_d)
        sb = _scores(yb[i], pb[i], rb[i], kb, include_meta_d)
        for m in boot:
            d = sb[m] - sa[m]
            if np.isfinite(d):
                boot[m].append(d)

    diffs = []
    for m in base_a:
        v = np.asarray(boot[m])
        if v.size == 0:
            continue
        p = 2 * min((v <= 0).mean(), (v >= 0).mean())
        diffs.append(MetricDiff(m, base_a[m], base_b[m], base_b[m] - base_a[m],
                                float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)),
                                float(min(1.0, p)), _BETTER[m]))
    return Comparison(n=int(n), n_boot=n_boot, diffs=diffs)


def confidence_from_logprobs(option_logprobs: dict[str, float], chosen: str | None = None) -> tuple[str, float]:
    """Turn per-option log-probabilities into (answer, confidence).

    Normalizes over the listed options only (e.g. {"A": -0.1, "B": -2.5, ...}),
    which is the usual way to read multiple-choice confidence from token
    logprobs. If `chosen` is None the argmax option is returned.
    """
    if not option_logprobs:
        raise ValueError("no options given")
    keys = list(option_logprobs)
    lp = np.array([option_logprobs[k] for k in keys], dtype=float)
    probs = np.exp(lp - lp.max())
    probs /= probs.sum()
    idx = int(np.argmax(probs)) if chosen is None else keys.index(chosen)
    return keys[idx], float(probs[idx])
