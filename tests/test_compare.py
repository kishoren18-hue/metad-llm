import json

import numpy as np
import pytest

from metad_llm import compare, confidence_from_logprobs
from metad_llm.cli import main


def _model(signal_noise, seed, n=1500):
    rng = np.random.default_rng(seed)
    ability = np.random.default_rng(99).normal(0.6, 1.3, n)  # same items for every model
    correct = (rng.uniform(size=n) < 1 / (1 + np.exp(-ability))).astype(int)
    signal = ability + rng.normal(0, signal_noise, n)
    conf = 1 / (1 + np.exp(-signal))
    return correct, np.clip(conf, 0.01, 0.99)


def test_detects_better_metacognition():
    ya, pa = _model(2.5, seed=1)   # noisy confidence
    yb, pb = _model(0.3, seed=2)   # sharp confidence
    res = compare(ya, pa, yb, pb, n_boot=150)
    d = {m.name: m for m in res.diffs}
    assert d["auroc2"].diff > 0.05
    assert d["auroc2"].winner == "B"
    assert d["auroc2"].p_value < 0.05
    assert d["meta_d"].winner == "B"


def test_identical_runs_show_no_difference():
    y, p = _model(1.0, seed=3)
    res = compare(y, p, y, p, n_boot=100, include_meta_d=False)
    for m in res.diffs:
        assert m.diff == pytest.approx(0.0)
        assert m.winner == "no clear difference"


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        compare([1] * 30, [0.9] * 30, [1] * 29, [0.9] * 29)


def test_confidence_from_logprobs():
    ans, conf = confidence_from_logprobs({"A": np.log(0.7), "B": np.log(0.2), "C": np.log(0.1)})
    assert ans == "A" and conf == pytest.approx(0.7)
    ans, conf = confidence_from_logprobs({"True": -0.1, "False": -2.4}, chosen="False")
    assert ans == "False" and 0 < conf < 0.5


def test_cli_compare_joins_on_id(tmp_path, capsys):
    ya, pa = _model(2.5, seed=4, n=300)
    yb, pb = _model(0.3, seed=5, n=300)
    fa, fb = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    fa.write_text("\n".join(json.dumps({"id": i, "correct": int(y), "confidence": float(p)})
                            for i, (y, p) in enumerate(zip(ya, pa))))
    # B written in reverse order to prove the join uses ids
    fb.write_text("\n".join(json.dumps({"id": i, "correct": int(y), "confidence": float(p)})
                            for i, (y, p) in reversed(list(enumerate(zip(yb, pb))))))
    assert main(["compare", str(fa), str(fb), "--boot", "60", "--no-meta-d", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    auroc = next(m for m in out["metrics"] if m["name"] == "auroc2")
    assert auroc["winner"] == "B" and out["n"] == 300
