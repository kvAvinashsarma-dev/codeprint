"""Export trained artifacts to site/assets/artifacts.json for the in-browser
(Pyodide) version of the detector: MLP weights, feature stats, the cohort
trigram LM, demo examples, and the demo student profile."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "site" / "assets"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ckpt = torch.load(ROOT / "models" / "model.pt", weights_only=False)
    sd = ckpt["state_dict"]
    layers = []
    # nn.Sequential: Linear layers live at net.0, net.3, net.6
    for key in ("net.0", "net.3", "net.6"):
        layers.append({
            "W": [[round(float(v), 7) for v in row] for row in sd[f"{key}.weight"]],
            "b": [round(float(v), 7) for v in sd[f"{key}.bias"]],
        })

    stats = json.loads((ROOT / "models" / "stats.json").read_text(encoding="utf-8"))

    with open(ROOT / "models" / "lm.pkl", "rb") as fh:
        lm = pickle.load(fh)

    examples = {}
    for key, name in (("ai", "suspicious_submission.py"),
                      ("human", "typical_student_submission.py")):
        examples[key] = (ROOT / "examples" / name).read_text(encoding="utf-8")

    profiles = {}
    for p in (ROOT / "profiles").glob("*.json"):
        profiles[p.stem] = json.loads(p.read_text(encoding="utf-8"))

    artifacts = {
        "in_dim": ckpt["in_dim"],
        "dropout": 0.3,
        "layers": layers,
        "stats": stats,
        "lm": {
            "trigrams": dict(lm.trigrams),
            "bigrams": dict(lm.bigrams),
            "vocab": sorted(lm.vocab),
        },
        "examples": examples,
        "profiles": profiles,
    }
    out_path = OUT / "artifacts.json"
    out_path.write_text(json.dumps(artifacts), encoding="utf-8")
    print(f"wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
