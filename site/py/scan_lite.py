"""Browser-side CodePrint scanner (Pyodide + numpy, no torch).

Mirrors codeprint.report.scan_submission: same features, same standardization,
an equivalent numpy implementation of the MC-dropout MLP forward pass, the
same calibrated thresholds, evidence extraction, and authorship drift.
"""

from __future__ import annotations

import json
import math
from collections import Counter

import numpy as np

from ast_struct import AST_FEATURE_NAMES, extract_ast_features
from predictability import (
    PREDICTABILITY_FEATURE_NAMES,
    CharTrigramLM,
    extract_predictability_features,
)
from style import STYLE_FEATURE_NAMES, STYLE_FEATURES, extract_style_features

ALL_FEATURE_NAMES = STYLE_FEATURE_NAMES + AST_FEATURE_NAMES + PREDICTABILITY_FEATURE_NAMES

FEATURE_DESCRIPTIONS = {
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

_ART: dict | None = None
_LM: CharTrigramLM | None = None
_RNG = np.random.default_rng(7)


def init(artifacts_json: str) -> None:
    """Load the exported artifacts (called once from JS)."""
    global _ART, _LM
    _ART = json.loads(artifacts_json)
    lm = CharTrigramLM()
    lm.trigrams = Counter(_ART["lm"]["trigrams"])
    lm.bigrams = Counter(_ART["lm"]["bigrams"])
    lm.vocab = set(_ART["lm"]["vocab"])
    _LM = lm


def _extract(source: str) -> np.ndarray:
    style = extract_style_features(source)
    vec = (
        [style[name] for name in STYLE_FEATURE_NAMES]
        + extract_ast_features(source)
        + extract_predictability_features(source, _LM)
    )
    return np.asarray(vec, dtype=np.float32)


def _mc_forward(x: np.ndarray, passes: int = 30) -> tuple[float, float]:
    """MC-dropout forward pass; numpy port of model.network.mc_predict."""
    layers = _ART["layers"]
    p = _ART["dropout"]
    temperature = _ART["stats"]["temperature"]
    Ws = [np.asarray(l["W"], dtype=np.float64) for l in layers]
    bs = [np.asarray(l["b"], dtype=np.float64) for l in layers]

    probs = []
    for _ in range(passes):
        h = x
        for i, (W, b) in enumerate(zip(Ws, bs)):
            h = h @ W.T + b
            if i < len(Ws) - 1:  # hidden layers: ReLU + dropout
                h = np.maximum(h, 0.0)
                mask = _RNG.random(h.shape) >= p
                h = h * mask / (1.0 - p)
        logits = h / temperature
        e = np.exp(logits - logits.max())
        probs.append(float(e[1] / e.sum()))
    arr = np.asarray(probs)
    # ddof=1 matches torch.std's default (Bessel-corrected)
    return float(arr.mean()), float(arr.std(ddof=1))


def _evidence(vec: np.ndarray, top_k: int = 8) -> list[dict]:
    stats = _ART["stats"]
    names = stats["feature_names"]
    h_mean = np.array(stats["human_mean"])
    h_std = np.maximum(np.array(stats["human_std"]), 0.02)
    a_mean = np.array(stats["ai_mean"])

    z_human = np.clip((vec - h_mean) / h_std, -25, 25)
    toward_ai = np.sign(a_mean - h_mean) == np.sign(vec - h_mean)
    strength = np.abs(z_human) * toward_ai

    items = []
    for i in np.argsort(-strength):
        if strength[i] < 1.5 or len(items) >= top_k:
            break
        name = names[i]
        if name.startswith(("astpath_", "node_")):
            continue
        items.append({
            "feature": name,
            "description": FEATURE_DESCRIPTIONS.get(name, name),
            "observed": round(float(vec[i]), 3),
            "typical_human": round(float(h_mean[i]), 3),
            "typical_ai": round(float(a_mean[i]), 3),
            "z_vs_humans": round(float(z_human[i]), 2),
        })
    return items


def _drift(code: str, profile: dict) -> dict:
    style = extract_style_features(code)
    vec = np.asarray([style[n] for n in STYLE_FEATURE_NAMES], dtype=np.float32)
    mean = np.array(profile["mean"])
    std = np.array(profile["std"])
    z = (vec - mean) / std
    order = np.argsort(-np.abs(z))
    top = [
        {"feature": STYLE_FEATURE_NAMES[i], "z": float(z[i]),
         "historical": float(mean[i]), "observed": float(vec[i])}
        for i in order[:5] if abs(z[i]) > 2.0
    ]
    score = float(np.abs(z).mean())
    return {
        "drift_score": score,
        "level": "high" if score > 2.0 else "moderate" if score > 1.2 else "normal",
        "history_size": profile["n"],
        "shifted_features": top,
    }


def scan(code: str, student_id: str | None = None) -> str:
    """Scan a submission; returns the report as a JSON string for JS."""
    stats = _ART["stats"]
    vec = _extract(code)
    mean = np.array(stats["scaler_mean"])
    std = np.array(stats["scaler_std"])
    x = (vec - mean) / std
    p_ai, unc = _mc_forward(x)

    th = stats["thresholds"]
    if p_ai >= th["ai"] and unc <= th["max_std"]:
        verdict = "AI-GENERATED"
    elif p_ai <= th["human"] and unc <= th["max_std"]:
        verdict = "HUMAN-WRITTEN"
    else:
        verdict = "INCONCLUSIVE"

    report = {
        "verdict": verdict,
        "p_ai": round(p_ai, 4),
        "uncertainty": round(unc, 4),
        "evidence": _evidence(vec),
        "drift": None,
        "note": (
            "Verdict is calibrated; INCONCLUSIVE means the model declines to "
            "accuse rather than guess. Use evidence + drift as discussion "
            "points with the student, not as standalone proof."
        ),
    }
    profile = (_ART.get("profiles") or {}).get(student_id or "")
    if profile:
        report["drift"] = _drift(code, profile)
    return json.dumps(report)
