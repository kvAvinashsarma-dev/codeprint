"""Unified feature extraction: style + AST structure + predictability -> one vector."""

from __future__ import annotations

import numpy as np

from .ast_struct import AST_FEATURE_NAMES, extract_ast_features
from .predictability import (
    PREDICTABILITY_FEATURE_NAMES,
    CharTrigramLM,
    extract_predictability_features,
)
from .style import STYLE_FEATURE_NAMES, STYLE_FEATURES, extract_style_features

ALL_FEATURE_NAMES = STYLE_FEATURE_NAMES + AST_FEATURE_NAMES + PREDICTABILITY_FEATURE_NAMES

# Human-readable descriptions used in evidence reports.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    **{name: desc for name, (desc, _) in STYLE_FEATURES.items()},
    "max_depth": "Maximum nesting depth of the syntax tree",
    "mean_depth": "Average nesting depth of the syntax tree",
    "mean_branching": "Average statements per block",
    "func_count": "Number of functions defined",
    "mean_func_size": "Average function size (AST nodes)",
    "func_size_cv": "Variability of function sizes (LLMs produce uniform functions)",
    "mean_logprob": "Overall predictability under the course corpus language model",
    "line_logprob_std": "Burstiness: variance of per-line predictability (humans are bursty)",
    "min_line_logprob": "Predictability of the most surprising line",
    "repeated_4gram_ratio": "Amount of internally repeated phrasing",
    "char_entropy": "Character-level entropy of the file",
}


class FeatureExtractor:
    """Turns raw source into the fixed-length vector the network consumes.

    The predictability block needs a cohort language model; fit it once on the
    training corpus (or the class's own submissions) via ``fit_lm``.
    """

    def __init__(self, lm: CharTrigramLM | None = None) -> None:
        self.lm = lm or CharTrigramLM()

    def fit_lm(self, corpus: list[str]) -> "FeatureExtractor":
        self.lm = CharTrigramLM().fit(corpus)
        return self

    @property
    def feature_names(self) -> list[str]:
        return list(ALL_FEATURE_NAMES)

    @property
    def n_features(self) -> int:
        return len(ALL_FEATURE_NAMES)

    def extract(self, source: str) -> np.ndarray:
        style = extract_style_features(source)
        vec = (
            [style[name] for name in STYLE_FEATURE_NAMES]
            + extract_ast_features(source)
            + extract_predictability_features(source, self.lm)
        )
        return np.asarray(vec, dtype=np.float32)

    def extract_batch(self, sources: list[str]) -> np.ndarray:
        return np.stack([self.extract(s) for s in sources])

    def style_vector(self, source: str) -> np.ndarray:
        """Style-only sub-vector, used for per-student drift profiles."""
        style = extract_style_features(source)
        return np.asarray([style[name] for name in STYLE_FEATURE_NAMES], dtype=np.float32)
