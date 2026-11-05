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


def _paired(path_a: str, path_b: str):
    """Join two runs on `id` (falls back to line order when there are no ids)."""
    ra, rb = load_records(path_a), load_records(path_b)
    if all("id" in r for r in ra + rb):
        ib = {str(r["id"]): r for r in rb}
        shared = [r for r in ra if str(r["id"]) in ib]
        rb = [ib[str(r["id"])] for r in shared]
        ra = shared
    elif len(ra) != len(rb):
        raise SystemExit("runs have different lengths and no 'id' field to join on")
    return ra, rb


def compare_main(argv) -> int:
    from .compare import compare

    ap = argparse.ArgumentParser(prog="metad-llm compare",
                                 description="Paired bootstrap comparison of two runs on the same items.")
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--ratings", type=int, default=4)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--no-meta-d", action="store_true", help="skip meta-d' (much faster)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    ra, rb = _paired(a.run_a, a.run_b)
    res = compare(_col(ra, "correct"), _col(ra, "confidence"), _col(rb, "correct"), _col(rb, "confidence"),
                  a.ratings, a.boot, include_meta_d=not a.no_meta_d)
    print(json.dumps(res.as_dict(), indent=2) if a.json else f"A = {a.run_a}\nB = {a.run_b}\n{res}")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "compare":
        return compare_main(argv[1:])
    ap = argparse.ArgumentParser(
        prog="metad-llm",
        description="Metacognitive sensitivity (meta-d', M-ratio) and calibration for model confidence.",
        epilog="Default input fields: correct (0/1), confidence (0-1). "
               "With --binary: label (0/1) and p (model probability that label is 1). "
               "To compare two runs: metad-llm compare run_a.jsonl run_b.jsonl",
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
