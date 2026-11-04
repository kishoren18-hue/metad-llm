import numpy as np
import pytest

from metad_llm import (
    assess,
    assess_binary,
    auroc2,
    bin_confidence,
    brier,
    counts_from_correctness,
    counts_from_trials,
    ece,
    fit_meta_d,
    simulate_observer,
    type1_sdt,
)


def test_type1_recovers_d_prime():
    a, b, _ = simulate_observer(40000, d_prime=1.8, criterion=0.3, seed=1)
    t = type1_sdt(a, b)
    assert t.d_prime == pytest.approx(1.8, abs=0.06)
    assert t.criterion == pytest.approx(0.3, abs=0.05)


@pytest.mark.parametrize("d", [1.0, 1.5, 2.2])
def test_ideal_observer_has_m_ratio_near_one(d):
    a, b, _ = simulate_observer(30000, d_prime=d, meta_noise=0.0, criterion=0.1, seed=2)
    f = fit_meta_d(a, b)
    assert f.converged
    assert f.m_ratio == pytest.approx(1.0, abs=0.08)


def test_metacognitive_noise_lowers_meta_d_but_not_d():
    clean = fit_meta_d(*simulate_observer(30000, 1.5, 0.0, seed=3)[:2])
    noisy = fit_meta_d(*simulate_observer(30000, 1.5, 1.5, seed=3)[:2])
    assert noisy.d_prime == pytest.approx(clean.d_prime, abs=0.05)
    assert noisy.meta_d < clean.meta_d - 0.4
    assert noisy.m_ratio < 0.7


def test_counts_layout():
    # one S1 trial answered S1 at top confidence, one S2 trial answered S2 at top confidence
    a, b = counts_from_trials([0, 1], [0, 1], [3, 3], n_ratings=3)
    assert a.tolist() == [1, 0, 0, 0, 0, 0]
    assert b.tolist() == [0, 0, 0, 0, 0, 1]


def test_correctness_mode_matches_binary_fit_for_unbiased_observer():
    a, b, tr = simulate_observer(40000, 1.6, 0.8, criterion=0.0, n_ratings=4, seed=4)
    binary = fit_meta_d(a, b)
    a2, b2 = counts_from_correctness(tr["correct"], tr["rating"], 4)
    collapsed = fit_meta_d(a2, b2)
    assert collapsed.d_prime == pytest.approx(binary.d_prime, abs=0.05)
    assert collapsed.meta_d == pytest.approx(binary.meta_d, abs=0.08)


def test_binning_handles_ties():
    conf = [0.9] * 50 + [0.99] * 50 + [0.6] * 10
    with pytest.warns(UserWarning):
        ratings, edges = bin_confidence(conf, n_ratings=4)
    assert ratings.min() == 1 and ratings.max() == len(edges) - 1


def test_binning_rejects_constant_confidence():
    with pytest.raises(ValueError):
        bin_confidence([0.8] * 20, 4)


def test_calibration_metrics_basic():
    y = np.array([1, 1, 0, 0])
    p = np.array([1.0, 1.0, 0.0, 0.0])
    assert ece(y, p) == pytest.approx(0.0)
    assert brier(y, p) == pytest.approx(0.0)
    assert auroc2(y, p) == pytest.approx(1.0)
    assert auroc2(y, 1 - p) == pytest.approx(0.0)


def test_assess_end_to_end_with_bootstrap():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.3, 1.0, 3000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    rep = assess(y, p, n_ratings=4, n_boot=50)
    assert rep.ece < 0.05
    assert 0.6 < rep.auroc2 < 0.8
    assert rep.m_ratio_ci[0] < rep.m_ratio < rep.m_ratio_ci[1]
    assert "unbiased" in rep.assumption


def test_assess_binary_runs():
    rng = np.random.default_rng(1)
    label = rng.integers(0, 2, 2000)
    logit = np.where(label == 1, 1.0, -1.0) + rng.normal(0, 1.2, label.size)
    p = 1 / (1 + np.exp(-logit))
    rep = assess_binary(label, p)
    assert rep.d_prime > 0.8
    assert rep.meta_d > 0.3


def test_bad_inputs():
    with pytest.raises(ValueError):
        fit_meta_d([1, 2, 3], [1, 2, 3])
    with pytest.raises(ValueError):
        ece([1, 0], [1.2, 0.3])
