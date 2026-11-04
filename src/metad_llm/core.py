"""High-level API: turn model outputs into rating counts and a full metacognition report."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np

from . import metrics
from .sdt import fit_meta_d

__all__ = [
    "bin_confidence",
    "counts_from_trials",
    "counts_from_probabilities",
    "counts_from_correctness",
    "MetacogReport",
    "assess",
    "assess_binary",
]


def bin_confidence(confidence, n_ratings: int = 4, edges=None):
    """Map continuous confidence to integer ratings 1..K.

    By default uses equal-count (quantile) bins, which keeps every rating level
    populated and makes meta-d' estimation stable. LLM confidences are often
    heavily tied (lots of 0.9 and 0.99); if quantile edges collapse, K is reduced
    and a warning is raised. Returns (ratings, edges).
    """
    c = np.asarray(confidence, dtype=float).ravel()
    if edges is None:
        qs = np.quantile(c, np.linspace(0, 1, n_ratings + 1))
        edges = np.unique(qs)
        if len(edges) - 1 < n_ratings:
            warnings.warn(
                f"confidence has too many ties for {n_ratings} quantile bins; "
                f"using {max(len(edges) - 1, 1)} levels",
                stacklevel=2,
            )
    edges = np.asarray(edges, dtype=float)
    k = len(edges) - 1
    if k < 2:
        raise ValueError("confidence takes a single value; meta-d' needs at least 2 rating levels")
    ratings = np.clip(np.searchsorted(edges[1:-1], c, side="right") + 1, 1, k)
    return ratings.astype(int), edges


def counts_from_trials(stimulus, response, rating, n_ratings: int):
    """Build nR_S1, nR_S2 from per-trial data.

    stimulus, response : 0 for S1, 1 for S2
    rating : integer confidence 1..n_ratings (1 = lowest)
    """
    s = np.asarray(stimulus).astype(int).ravel()
    r = np.asarray(response).astype(int).ravel()
    k = np.asarray(rating).astype(int).ravel()
    if not (s.shape == r.shape == k.shape):
        raise ValueError("stimulus, response and rating must have equal length")
    if k.min() < 1 or k.max() > n_ratings:
        raise ValueError(f"ratings must lie in 1..{n_ratings}")
    nR_S1 = np.zeros(2 * n_ratings)
    nR_S2 = np.zeros(2 * n_ratings)
    # S1 response with rating k -> index K - k ; S2 response -> index K + k - 1
    idx = np.where(r == 0, n_ratings - k, n_ratings + k - 1)
    np.add.at(nR_S1, idx[s == 0], 1)
    np.add.at(nR_S2, idx[s == 1], 1)
    return nR_S1, nR_S2


def counts_from_probabilities(stimulus, p_s2, n_ratings: int = 4, edges=None):
    """For binary tasks where the model outputs P(answer = S2), e.g. P("True").

    Response is argmax; confidence is max(p, 1 - p), binned into ratings.
    """
    p = np.asarray(p_s2, dtype=float).ravel()
    response = (p >= 0.5).astype(int)
    conf = np.maximum(p, 1 - p)
    ratings, edges = bin_confidence(conf, n_ratings, edges)
    k = len(edges) - 1
    return (*counts_from_trials(stimulus, response, ratings, k), edges)


def counts_from_correctness(correct, rating, n_ratings: int):
    """Counts for tasks that only have correct/incorrect plus confidence.

    Open-ended QA has no natural S1/S2 stimulus classes. Under the assumption of
    unbiased type-1 responding (c = 0) and symmetric confidence criteria, the
    type-2 data for S1 and S2 responses are exchangeable, so each
    (correctness, rating) cell can be split evenly across two virtual stimulus
    classes. Under that assumption d' = 2 * z(accuracy). This is the standard
    trick for applying meta-d' to non-binary tasks; report it as an assumption.
    """
    y = np.asarray(correct).astype(int).ravel()
    k = np.asarray(rating).astype(int).ravel()
    nR_S1 = np.zeros(2 * n_ratings)
    nR_S2 = np.zeros(2 * n_ratings)
    for lvl in range(1, n_ratings + 1):
        for ok in (0, 1):
            n = int(((k == lvl) & (y == ok)).sum())
            half_a, half_b = n // 2, n - n // 2
            if ok:  # correct: S1->resp S1, S2->resp S2
                nR_S1[n_ratings - lvl] += half_a
                nR_S2[n_ratings + lvl - 1] += half_b
            else:  # error: S1->resp S2, S2->resp S1
                nR_S1[n_ratings + lvl - 1] += half_a
                nR_S2[n_ratings - lvl] += half_b
    return nR_S1, nR_S2


@dataclass
class MetacogReport:
    n: int
    accuracy: float
    mean_confidence: float
    ece: float
    adaptive_ece: float
    brier: float
    auroc2: float
    d_prime: float
    meta_d: float
    m_ratio: float
    n_ratings: int
    m_ratio_ci: tuple[float, float] | None = None
    meta_d_ci: tuple[float, float] | None = None
    assumption: str = ""

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        lines = [
            f"n trials        {self.n}",
            f"accuracy        {self.accuracy:.3f}",
            f"mean confidence {self.mean_confidence:.3f}",
            f"ECE             {self.ece:.3f}",
            f"Brier           {self.brier:.3f}",
            f"type-2 AUROC    {self.auroc2:.3f}",
            f"d'              {self.d_prime:.3f}",
            f"meta-d'         {self.meta_d:.3f}" + (
                f"  95% CI [{self.meta_d_ci[0]:.2f}, {self.meta_d_ci[1]:.2f}]" if self.meta_d_ci else ""),
            f"M-ratio         {self.m_ratio:.3f}" + (
                f"  95% CI [{self.m_ratio_ci[0]:.2f}, {self.m_ratio_ci[1]:.2f}]" if self.m_ratio_ci else ""),
            f"rating levels   {self.n_ratings}",
        ]
        if self.assumption:
            lines.append(f"note            {self.assumption}")
        return "\n".join(lines)


def _fit(nR_S1, nR_S2):
    f = fit_meta_d(nR_S1, nR_S2)
    return f.d_prime, f.meta_d, f.m_ratio


def _bootstrap(fn, n, n_boot, seed):
    rng = np.random.default_rng(seed)
    md, mr = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            _, m, r = fn(idx)
        except ValueError:
            continue
        md.append(m)
        if np.isfinite(r):
            mr.append(r)
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else None  # noqa: E731
    return ci(md), ci(mr)


def assess(correct, confidence, n_ratings: int = 4, n_boot: int = 0, seed: int = 0) -> MetacogReport:
    """Full report for correctness + confidence data (open-ended QA, MCQ, etc.)."""
    y = np.asarray(correct).astype(int).ravel()
    p = np.asarray(confidence, dtype=float).ravel()
    ratings, edges = bin_confidence(p, n_ratings)
    k = len(edges) - 1
    d1, md, mr = _fit(*counts_from_correctness(y, ratings, k))
    md_ci = mr_ci = None
    if n_boot:
        md_ci, mr_ci = _bootstrap(
            lambda i: _fit(*counts_from_correctness(y[i], ratings[i], k)), y.size, n_boot, seed)
    return MetacogReport(
        n=int(y.size), accuracy=float(y.mean()), mean_confidence=float(p.mean()),
        ece=metrics.ece(y, p), adaptive_ece=metrics.adaptive_ece(y, p),
        brier=metrics.brier(y, p), auroc2=metrics.auroc2(y, p),
        d_prime=d1, meta_d=md, m_ratio=mr, n_ratings=k,
        meta_d_ci=md_ci, m_ratio_ci=mr_ci,
        assumption="correctness-only mode assumes unbiased type-1 responding (c = 0)",
    )


def assess_binary(stimulus, p_s2, n_ratings: int = 4, n_boot: int = 0, seed: int = 0) -> MetacogReport:
    """Full report for binary tasks (true/false, yes/no) with P(S2) per item.

    No symmetry assumption is needed here: response bias is estimated.
    """
    s = np.asarray(stimulus).astype(int).ravel()
    p = np.asarray(p_s2, dtype=float).ravel()
    resp = (p >= 0.5).astype(int)
    conf = np.maximum(p, 1 - p)
    y = (resp == s).astype(int)
    ratings, edges = bin_confidence(conf, n_ratings)
    k = len(edges) - 1
    d1, md, mr = _fit(*counts_from_trials(s, resp, ratings, k))
    md_ci = mr_ci = None
    if n_boot:
        md_ci, mr_ci = _bootstrap(
            lambda i: _fit(*counts_from_trials(s[i], resp[i], ratings[i], k)), s.size, n_boot, seed)
    return MetacogReport(
        n=int(s.size), accuracy=float(y.mean()), mean_confidence=float(conf.mean()),
        ece=metrics.ece(y, conf), adaptive_ece=metrics.adaptive_ece(y, conf),
        brier=metrics.brier(y, conf), auroc2=metrics.auroc2(y, conf),
        d_prime=d1, meta_d=md, m_ratio=mr, n_ratings=k, meta_d_ci=md_ci, m_ratio_ci=mr_ci,
    )
