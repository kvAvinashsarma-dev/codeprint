"""Assignment-cohort normalization.

A constrained assignment ("implement bubble sort using this skeleton") makes
everyone's code look alike — absolute AI-probabilities drift upward on such
assignments and a naive detector over-flags the whole class. Scoring each
submission relative to the cohort's own distribution (robust z via median/MAD)
separates "this assignment naturally looks generated" from "this submission
is an outlier within its own class".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .features.extract import FeatureExtractor
from .model.network import DetectorNet, mc_predict


def score_cohort(
    files: list[Path],
    model: DetectorNet,
    extractor: FeatureExtractor,
    stats: dict,
) -> list[dict]:
    codes = [f.read_text(encoding="utf-8", errors="replace") for f in files]
    X = extractor.extract_batch(codes)
    mean = np.array(stats["scaler_mean"])
    std = np.array(stats["scaler_std"])
    Xs = torch.tensor((X - mean) / std, dtype=torch.float32)
    p_mean, p_std = mc_predict(model, Xs, temperature=stats["temperature"])
    p = p_mean.numpy()

    median = float(np.median(p))
    mad = float(np.median(np.abs(p - median))) or 1e-6
    robust_z = np.clip(0.6745 * (p - median) / mad, -50, 50)

    results = []
    for f, pi, si, zi in zip(files, p, p_std.numpy(), robust_z):
        results.append({
            "file": str(f),
            "p_ai": float(pi),
            "uncertainty": float(si),
            "cohort_z": float(zi),
            "cohort_outlier": bool(zi > 2.5 and pi > 0.5),
        })
    return sorted(results, key=lambda r: -r["cohort_z"])
