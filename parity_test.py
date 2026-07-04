"""Check that the browser (numpy) scanner matches the torch pipeline."""

import json
import sys

sys.path.insert(0, "site/py")
import scan_lite

from codeprint.model.train import load_artifacts
from codeprint.report import scan_submission

scan_lite.init(open("site/assets/artifacts.json", encoding="utf-8").read())
model, extractor, stats = load_artifacts("models")

for key, path in [("ai", "examples/suspicious_submission.py"),
                  ("human", "examples/typical_student_submission.py")]:
    code = open(path, encoding="utf-8").read()
    lite = json.loads(scan_lite.scan(code, "s001"))
    ref = scan_submission(code, model, extractor, stats)
    drift = lite["drift"]["level"] if lite["drift"] else None
    print(f"{key:6s} lite : {lite['verdict']:14s} p={lite['p_ai']:.3f} "
          f"unc={lite['uncertainty']:.3f} evidence={len(lite['evidence'])} drift={drift}")
    print(f"{'':6s} torch: {ref['verdict']:14s} p={ref['p_ai']:.3f} "
          f"unc={ref['uncertainty']:.3f} evidence={len(ref['evidence'])}")
    assert lite["verdict"] == ref["verdict"], "verdict mismatch"
    assert abs(lite["p_ai"] - ref["p_ai"]) < 0.05, "probability mismatch"

print("PARITY OK")
