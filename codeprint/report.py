"""Evidence reports: a verdict an instructor can defend in an integrity hearing.

Instead of a bare percentage, every scan produces:
  - a calibrated three-way verdict (AI-GENERATED / HUMAN-WRITTEN / INCONCLUSIVE)
  - the concrete stylometric evidence behind it, in plain English
  - optional authorship-drift analysis against the student's own history
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import torch

from .baseline import StudentProfile
from .features.extract import FEATURE_DESCRIPTIONS, FeatureExtractor
from .model.network import DetectorNet, mc_predict


def _evidence(vec: np.ndarray, stats: dict, top_k: int = 8) -> list[dict]:
    """Features where this submission sits far from the human population and
    on the AI side of the divide."""
    names = stats["feature_names"]
    h_mean = np.array(stats["human_mean"])
    # floor the std so zero-variance features can't produce absurd z-scores
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
        if name.startswith(("astpath_", "node_")):  # hashed dims aren't human-readable
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


def scan_submission(
    code: str,
    model: DetectorNet,
    extractor: FeatureExtractor,
    stats: dict,
    student_profile: StudentProfile | None = None,
    filename: str = "<submission>",
) -> dict:
    vec = extractor.extract(code)
    mean = np.array(stats["scaler_mean"])
    std = np.array(stats["scaler_std"])
    x = torch.tensor(((vec - mean) / std)[None, :], dtype=torch.float32)
    p_mean, p_std = mc_predict(model, x, temperature=stats["temperature"])
    p_ai, unc = float(p_mean[0]), float(p_std[0])

    th = stats["thresholds"]
    if p_ai >= th["ai"] and unc <= th["max_std"]:
        verdict = "AI-GENERATED"
    elif p_ai <= th["human"] and unc <= th["max_std"]:
        verdict = "HUMAN-WRITTEN"
    else:
        verdict = "INCONCLUSIVE"

    report = {
        "file": filename,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verdict": verdict,
        "p_ai": round(p_ai, 4),
        "uncertainty": round(unc, 4),
        "evidence": _evidence(vec, stats),
        "drift": None,
        "note": (
            "Verdict is calibrated; INCONCLUSIVE means the model declines to "
            "accuse rather than guess. Use evidence + drift as discussion "
            "points with the student, not as standalone proof."
        ),
    }
    if student_profile is not None:
        report["drift"] = student_profile.drift(code, extractor)
    return report


def render_markdown(report: dict) -> str:
    icon = {"AI-GENERATED": "🚨", "HUMAN-WRITTEN": "✅", "INCONCLUSIVE": "⚖️"}
    lines = [
        f"# CodePrint report — `{report['file']}`",
        "",
        f"**Verdict:** {icon[report['verdict']]} **{report['verdict']}**  ",
        f"**P(AI-generated):** {report['p_ai']:.1%} ± {report['uncertainty']:.1%} "
        f"(Monte-Carlo uncertainty)",
        "",
    ]
    if report["evidence"]:
        lines += ["## Evidence", "",
                  "| Signal | Observed | Typical human | Typical AI | σ from humans |",
                  "|---|---|---|---|---|"]
        for e in report["evidence"]:
            lines.append(
                f"| {e['description']} | {e['observed']} | {e['typical_human']} "
                f"| {e['typical_ai']} | {e['z_vs_humans']:+.1f} |")
        lines.append("")
    drift = report.get("drift")
    if drift:
        lines += [
            "## Authorship drift vs. this student's own history",
            "",
            f"Drift score **{drift['drift_score']:.2f}** "
            f"({drift['level']}; baseline of {drift['history_size']} prior submissions)",
            "",
        ]
        for s in drift["shifted_features"]:
            lines.append(
                f"- **{FEATURE_DESCRIPTIONS.get(s['feature'], s['feature'])}**: "
                f"historically {s['historical']:.2f}, now {s['observed']:.2f} "
                f"({s['z']:+.1f}σ)")
        lines.append("")
    lines += ["---", f"_{report['note']}_"]
    return "\n".join(lines)


def render_json(report: dict) -> str:
    return json.dumps(report, indent=2)
