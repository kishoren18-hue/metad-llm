"""Signal detection theory core: type-1 d', and meta-d' by maximum likelihood.

The meta-d' model follows Maniscalco & Lau (2012), "A signal detection theoretic
approach for estimating metacognitive sensitivity from confidence ratings",
Consciousness and Cognition 21(1), 422-430.

Count convention (same as the original MATLAB/Python toolboxes):

    nR_S1[i] = number of trials where stimulus was S1 and the subject gave
               response/confidence category i
    nR_S2[i] = same for S2 stimuli

with categories ordered from "S1, highest confidence" ... "S1, lowest confidence",
"S2, lowest confidence" ... "S2, highest confidence". With K confidence levels
each array has length 2K.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

__all__ = ["Type1Fit", "MetaDFit", "type1_sdt", "fit_meta_d", "pad_counts"]


@dataclass(frozen=True)
class Type1Fit:
    d_prime: float
    criterion: float
    hit_rate: float
    false_alarm_rate: float


@dataclass(frozen=True)
class MetaDFit:
    meta_d: float
    d_prime: float
    m_ratio: float
    m_diff: float
    criterion: float
    meta_criterion: float
    t2_criteria_s1: np.ndarray = field(repr=False)
    t2_criteria_s2: np.ndarray = field(repr=False)
    log_likelihood: float = float("nan")
    converged: bool = True
    n_ratings: int = 0

    def as_dict(self) -> dict:
        return {
            "meta_d": self.meta_d,
            "d_prime": self.d_prime,
            "m_ratio": self.m_ratio,
            "m_diff": self.m_diff,
            "criterion": self.criterion,
            "meta_criterion": self.meta_criterion,
            "log_likelihood": self.log_likelihood,
            "converged": self.converged,
            "n_ratings": self.n_ratings,
        }


def _validate(nR_S1, nR_S2) -> tuple[np.ndarray, np.ndarray, int]:
    a = np.asarray(nR_S1, dtype=float)
    b = np.asarray(nR_S2, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("nR_S1 and nR_S2 must be 1-D arrays of equal length")
    if a.size % 2 or a.size < 4:
        raise ValueError("count arrays must have even length >= 4 (2 x n_ratings, n_ratings >= 2)")
    if (a < 0).any() or (b < 0).any():
        raise ValueError("counts must be non-negative")
    if a.sum() == 0 or b.sum() == 0:
        raise ValueError("need at least one trial of each stimulus class")
    return a, b, a.size // 2


def pad_counts(nR_S1, nR_S2, pad: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Add a small constant to every cell so no response rate is exactly 0 or 1.

    Default pad is 1 / (2K) per cell (Hautus, 1995 style), as in the reference
    toolbox. Pass pad=0 to disable.
    """
    a, b, k = _validate(nR_S1, nR_S2)
    if pad is None:
        pad = 1.0 / (2 * k)
    return a + pad, b + pad


def type1_sdt(nR_S1, nR_S2, pad: float | None = None) -> Type1Fit:
    """Equal-variance type-1 d' and criterion c from rating counts."""
    a, b = pad_counts(nR_S1, nR_S2, pad)
    k = a.size // 2
    hr = b[k:].sum() / b.sum()  # P(resp S2 | S2)
    far = a[k:].sum() / a.sum()  # P(resp S2 | S1)
    zh, zf = norm.ppf(hr), norm.ppf(far)
    return Type1Fit(d_prime=float(zh - zf), criterion=float(-0.5 * (zh + zf)),
                    hit_rate=float(hr), false_alarm_rate=float(far))


def _unpack(theta: np.ndarray, k: int, c_prime: float):
    meta_d = theta[0]
    meta_c = c_prime * meta_d
    inc_s1 = np.exp(theta[1:k])
    inc_s2 = np.exp(theta[k:2 * k - 1])
    t2_s1 = meta_c - np.cumsum(inc_s1)  # decreasing, for S1 responses
    t2_s2 = meta_c + np.cumsum(inc_s2)  # increasing, for S2 responses
    return meta_d, meta_c, t2_s1, t2_s2


def _cell_probs(meta_d, meta_c, t2_s1, t2_s2):
    """Return P(conf category | stimulus, response) for both stimuli.

    Output arrays follow the nR ordering (S1 high conf ... S2 high conf).
    Probabilities are conditional on the type-1 response, which is what makes
    meta-d' independent of type-1 performance.
    """
    edges_s1 = np.concatenate(([-np.inf], t2_s1[::-1], [meta_c]))  # ascending
    edges_s2 = np.concatenate(([meta_c], t2_s2, [np.inf]))
    out = []
    for mu in (-meta_d / 2.0, meta_d / 2.0):
        cdf1 = norm.cdf(edges_s1, loc=mu)
        cdf2 = norm.cdf(edges_s2, loc=mu)
        p_r1 = cdf1[-1]
        p_r2 = 1.0 - cdf2[0]
        # S1 responses: bins from -inf upward = highest confidence first
        s1 = np.diff(cdf1) / max(p_r1, 1e-300)
        s2 = np.diff(cdf2) / max(p_r2, 1e-300)
        out.append(np.concatenate((s1, s2)))
    return out[0], out[1]


def _initial_theta(a, b, k, d1, c_prime):
    """Start type-2 criteria near the type-1 z-scored cumulative rates."""
    ratio_hr, ratio_far = [], []
    for i in range(1, 2 * k):
        ratio_hr.append(b[i:].sum() / b.sum())
        ratio_far.append(a[i:].sum() / a.sum())
    t1 = -0.5 * (norm.ppf(ratio_hr) + norm.ppf(ratio_far))
    t1 = t1 - t1[k - 1]  # centre on the type-1 criterion
    meta_d0 = max(d1, 0.1)
    meta_c0 = c_prime * meta_d0
    crit = t1 + meta_c0
    s1 = crit[:k - 1][::-1]  # below meta_c, closest first
    s2 = crit[k:]
    inc1 = np.maximum(np.diff(np.concatenate(([meta_c0], s1))) * -1, 0.05)
    inc2 = np.maximum(np.diff(np.concatenate(([meta_c0], s2))), 0.05)
    return np.concatenate(([meta_d0], np.log(inc1), np.log(inc2)))


def fit_meta_d(nR_S1, nR_S2, pad: float | None = None, s: float = 1.0) -> MetaDFit:
    """Fit meta-d' by maximum likelihood (equal-variance model, s=1).

    Parameters
    ----------
    nR_S1, nR_S2 : array-like of length 2K
        Rating counts (see module docstring for ordering).
    pad : float, optional
        Per-cell padding. Default 1/(2K). Use 0 for none.
    s : float
        Ratio of S1/S2 standard deviations. Only s=1 is supported.
    """
    if s != 1.0:
        raise NotImplementedError("only the equal-variance model (s=1) is implemented")
    a, b = pad_counts(nR_S1, nR_S2, pad)
    k = a.size // 2
    t1 = type1_sdt(nR_S1, nR_S2, pad)
    d1, c1 = t1.d_prime, t1.criterion
    # relative criterion; guards against d' ~ 0
    c_prime = c1 / d1 if abs(d1) > 1e-6 else 0.0

    def nll(theta):
        meta_d, meta_c, t2_s1, t2_s2 = _unpack(theta, k, c_prime)
        p1, p2 = _cell_probs(meta_d, meta_c, t2_s1, t2_s2)
        p1 = np.clip(p1, 1e-12, 1.0)
        p2 = np.clip(p2, 1e-12, 1.0)
        return -(a @ np.log(p1) + b @ np.log(p2))

    theta0 = _initial_theta(a, b, k, d1, c_prime)
    best = None
    # a few starting points for meta-d' make the fit robust when d' is small
    for md0 in (theta0[0], 0.5 * theta0[0] + 0.1, 1.5 * theta0[0] + 0.1):
        start = theta0.copy()
        start[0] = md0
        res = minimize(nll, start, method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    meta_d, meta_c, t2_s1, t2_s2 = _unpack(best.x, k, c_prime)
    m_ratio = meta_d / d1 if abs(d1) > 1e-6 else float("nan")
    return MetaDFit(
        meta_d=float(meta_d), d_prime=float(d1), m_ratio=float(m_ratio),
        m_diff=float(meta_d - d1), criterion=float(c1), meta_criterion=float(meta_c),
        t2_criteria_s1=t2_s1, t2_criteria_s2=t2_s2, log_likelihood=float(-best.fun),
        converged=bool(best.success), n_ratings=k,
    )
