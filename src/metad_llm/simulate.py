"""Simulate observers with known d' and meta-d' (for testing and demos)."""

from __future__ import annotations

import numpy as np

from .core import counts_from_trials

__all__ = ["simulate_observer"]


def simulate_observer(n_trials: int, d_prime: float, meta_noise: float = 0.0,
                      criterion: float = 0.0, n_ratings: int = 4, seed: int | None = None):
    """Simulate an SDT observer that rates confidence from a noisy copy of its evidence.

    meta_noise = 0 gives an ideal metacognitive observer (meta-d' ~= d', M-ratio ~= 1).
    Larger meta_noise corrupts the evidence used for confidence only, so meta-d'
    drops while d' is unchanged.

    Returns (nR_S1, nR_S2, trials) where trials is a dict of per-trial arrays.
    """
    rng = np.random.default_rng(seed)
    stim = rng.integers(0, 2, n_trials)
    x = rng.normal(np.where(stim == 1, d_prime / 2, -d_prime / 2), 1.0)
    resp = (x > criterion).astype(int)
    x_conf = x + rng.normal(0.0, meta_noise, n_trials) if meta_noise > 0 else x
    # signed distance from the criterion in the direction of the chosen response
    dist = np.where(resp == 1, x_conf - criterion, criterion - x_conf)
    # confidence criteria at fixed evidence distances, scaled to d'
    cuts = np.linspace(0, 1.5, n_ratings)[1:] * max(d_prime, 0.5)
    rating = 1 + np.searchsorted(cuts, dist, side="right")
    rating = np.clip(rating, 1, n_ratings)
    nR_S1, nR_S2 = counts_from_trials(stim, resp, rating, n_ratings)
    return nR_S1, nR_S2, {"stimulus": stim, "response": resp, "rating": rating,
                          "correct": (stim == resp).astype(int)}
