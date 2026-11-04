"""metad-llm: cognitive-science metacognition metrics for model confidence."""

from .core import (
    MetacogReport,
    assess,
    assess_binary,
    bin_confidence,
    counts_from_correctness,
    counts_from_probabilities,
    counts_from_trials,
)
from .metrics import adaptive_ece, auroc2, brier, ece, overconfidence, reliability_table
from .sdt import MetaDFit, Type1Fit, fit_meta_d, type1_sdt
from .simulate import simulate_observer

__version__ = "0.1.0"

__all__ = [
    "MetacogReport", "assess", "assess_binary", "bin_confidence",
    "counts_from_correctness", "counts_from_probabilities", "counts_from_trials",
    "adaptive_ece", "auroc2", "brier", "ece", "overconfidence", "reliability_table",
    "MetaDFit", "Type1Fit", "fit_meta_d", "type1_sdt", "simulate_observer",
]
