"""Build demo folders from the generated dataset: one honest student's history
and one assignment cohort (10 human + 2 AI submissions)."""

import json
import pathlib
import random

root = pathlib.Path(__file__).resolve().parents[1]
recs = [json.loads(l) for l in open(root / "data/dataset.jsonl", encoding="utf-8")]

hist = root / "demo/history"
hist.mkdir(parents=True, exist_ok=True)
mine = [r for r in recs if r["student_id"] == "s001"][:4]
for i, r in enumerate(mine):
    (hist / f"week{i + 1}_{r['assignment']}.py").write_text(r["code"], encoding="utf-8")

coh = root / "demo/hw_grades"
coh.mkdir(parents=True, exist_ok=True)
g = [r for r in recs if r["assignment"] == "grades"]
rnd = random.Random(1)
pick = (rnd.sample([r for r in g if r["label"] == 0], 10)
        + rnd.sample([r for r in g if r["label"] == 1], 2))
for r in pick:
    label = "ai" if r["label"] else "human"
    (coh / f"{r['student_id']}_{label}.py").write_text(r["code"], encoding="utf-8")

print("history:", len(mine), "cohort:", len(pick))
