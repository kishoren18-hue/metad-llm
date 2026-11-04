"""Command line: metad-llm predictions.jsonl [--binary] [--json] [--plot out.png]"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from .core import assess, assess_binary


def load_records(path: str) -> list[dict]:
    p = Path(path)
    text = sys.stdin.read() if path == "-" else p.read_text()
    if p.suffix == ".csv":
        return list(csv.DictReader(text.splitlines()))
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _col(records, name):
    try:
        return np.array([float(r[name]) for r in records])
    except KeyError as e:
        raise SystemExit(f"missing field {e} in input; see --help for the expected columns") from None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="metad-llm",
        description="Metacognitive sensitivity (meta-d', M-ratio) and calibration for model confidence.",
        epilog="Default input fields: correct (0/1), confidence (0-1). "
               "With --binary: label (0/1) and p (model probability that label is 1).",
    )
    ap.add_argument("path", help="JSONL or CSV file, or - for stdin")
    ap.add_argument("--binary", action="store_true", help="binary task with label + p columns")
    ap.add_argument("--ratings", type=int, default=4, help="number of confidence levels (default 4)")
    ap.add_argument("--boot", type=int, default=0, help="bootstrap resamples for CIs")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    ap.add_argument("--plot", help="save reliability diagram + type-2 ROC to this image path")
    a = ap.parse_args(argv)

    recs = load_records(a.path)
    if a.binary:
        label, p = _col(recs, "label"), _col(recs, "p")
        rep = assess_binary(label, p, a.ratings, a.boot)
        correct = ((p >= 0.5).astype(int) == label).astype(int)
        conf = np.maximum(p, 1 - p)
    else:
        correct, conf = _col(recs, "correct"), _col(recs, "confidence")
        rep = assess(correct, conf, a.ratings, a.boot)
    print(json.dumps(rep.as_dict(), indent=2) if a.json else rep)
    if a.plot:
        from .plot import report_figure
        report_figure({Path(a.path).stem: (correct, conf)}).savefig(a.plot, dpi=150, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
