"""Per-student longitudinal style baselines ("who wrote this?" not just "is it AI?").

Every student accumulates a stylometric fingerprint from submissions written
before LLM access mattered (week-1 labs, in-class exercises, past homework).
A new submission is compared against *that student's own* fingerprint: a
sudden jump in style — regardless of absolute AI-likeness — is flagged as
authorship drift. This catches AI use that absolute classifiers miss (e.g. a
strong student whose code already looks polished) and protects students whose
natural style just happens to resemble LLM output.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features.extract import FeatureExtractor
from .features.style import STYLE_FEATURE_NAMES

MIN_HISTORY = 3


class StudentProfile:
    def __init__(self, student_id: str, mean: np.ndarray, std: np.ndarray, n: int):
        self.student_id = student_id
        self.mean = mean
        self.std = std
        self.n = n

    @classmethod
    def build(cls, student_id: str, codes: list[str],
              extractor: FeatureExtractor) -> "StudentProfile":
        if len(codes) < MIN_HISTORY:
            raise ValueError(
                f"Need at least {MIN_HISTORY} known-authentic submissions to "
                f"build a profile, got {len(codes)}")
        vecs = np.stack([extractor.style_vector(c) for c in codes])
        return cls(student_id, vecs.mean(axis=0), vecs.std(axis=0) + 0.05, len(codes))

    def drift(self, code: str, extractor: FeatureExtractor) -> dict:
        """Mean absolute z-distance of the new submission from this student's
        own history, plus the most-shifted features for the evidence report."""
        vec = extractor.style_vector(code)
        z = (vec - self.mean) / self.std
        order = np.argsort(-np.abs(z))
        top = [
            {"feature": STYLE_FEATURE_NAMES[i], "z": float(z[i]),
             "historical": float(self.mean[i]), "observed": float(vec[i])}
            for i in order[:5] if abs(z[i]) > 2.0
        ]
        score = float(np.abs(z).mean())
        return {
            "drift_score": score,
            "level": "high" if score > 2.0 else "moderate" if score > 1.2 else "normal",
            "history_size": self.n,
            "shifted_features": top,
        }

    def save(self, profiles_dir: str | Path) -> Path:
        path = Path(profiles_dir) / f"{self.student_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "student_id": self.student_id, "n": self.n,
            "mean": self.mean.tolist(), "std": self.std.tolist(),
        }), encoding="utf-8")
        return path

    @classmethod
    def load(cls, profiles_dir: str | Path, student_id: str) -> "StudentProfile | None":
        path = Path(profiles_dir) / f"{student_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(student_id, np.array(data["mean"]), np.array(data["std"]), data["n"])
