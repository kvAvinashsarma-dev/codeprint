"""Predictability / burstiness features from a cohort-trained character LM.

LLM output is statistically *smooth*: every line is about equally predictable.
Human student code is bursty — clean textbook lines next to chaotic debugging
lines. We fit a small character trigram model on the course's own corpus (no
external model, no internet, no student code leaving the machine) and measure
how uniformly predictable each submission is.
"""

from __future__ import annotations

import math
from collections import Counter

PREDICTABILITY_FEATURE_NAMES = [
    "mean_logprob",
    "line_logprob_std",   # burstiness: low == suspiciously uniform
    "min_line_logprob",
    "repeated_4gram_ratio",
    "char_entropy",
]


class CharTrigramLM:
    """Laplace-smoothed character trigram model."""

    def __init__(self) -> None:
        self.trigrams: Counter = Counter()
        self.bigrams: Counter = Counter()
        self.vocab: set[str] = set()

    def fit(self, texts: list[str]) -> "CharTrigramLM":
        for text in texts:
            padded = "\x02\x02" + text
            self.vocab.update(padded)
            for i in range(len(padded) - 2):
                self.trigrams[padded[i:i + 3]] += 1
                self.bigrams[padded[i:i + 2]] += 1
        return self

    def _char_logprob(self, context: str, char: str) -> float:
        v = max(len(self.vocab), 2)
        num = self.trigrams[context + char] + 1
        den = self.bigrams[context] + v
        return math.log2(num / den)

    def mean_logprob(self, text: str) -> float:
        if not text:
            return 0.0
        padded = "\x02\x02" + text
        total = sum(self._char_logprob(padded[i:i + 2], padded[i + 2])
                    for i in range(len(padded) - 2))
        return total / max(len(text), 1)


def _char_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def extract_predictability_features(source: str, lm: CharTrigramLM) -> list[float]:
    lines = [ln for ln in source.splitlines() if ln.strip()]
    line_lps = [lm.mean_logprob(ln) for ln in lines] or [0.0]
    mean_lp = sum(line_lps) / len(line_lps)
    std_lp = math.sqrt(sum((lp - mean_lp) ** 2 for lp in line_lps) / len(line_lps))

    tokens = source.split()
    grams = [tuple(tokens[i:i + 4]) for i in range(len(tokens) - 3)]
    repeated = 0.0
    if grams:
        counts = Counter(grams)
        repeated = sum(c for c in counts.values() if c > 1) / len(grams)

    return [
        lm.mean_logprob(source),
        std_lp,
        min(line_lps),
        repeated,
        _char_entropy(source),
    ]
