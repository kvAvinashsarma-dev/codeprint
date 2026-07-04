"""Local web interface for CodePrint.

    python webapp.py          ->  http://127.0.0.1:5000

Everything runs on this machine: the model is loaded once from ./models and
no submission ever leaves localhost.
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request

from codeprint.baseline import StudentProfile
from codeprint.model.train import load_artifacts
from codeprint.report import scan_submission

ROOT = Path(__file__).resolve().parent
PROFILES = ROOT / "profiles"

app = Flask(__name__, static_folder=str(ROOT / "web"), static_url_path="")
model, extractor, stats = load_artifacts(ROOT / "models")


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/students")
def students():
    return jsonify(sorted(p.stem for p in PROFILES.glob("*.json")))


@app.get("/api/examples")
def examples():
    out = {}
    for key, name in (("ai", "suspicious_submission.py"),
                      ("human", "typical_student_submission.py")):
        path = ROOT / "examples" / name
        if path.exists():
            out[key] = path.read_text(encoding="utf-8")
    return jsonify(out)


@app.post("/api/scan")
def scan():
    data = request.get_json(force=True, silent=True) or {}
    code = data.get("code", "")
    if not code.strip():
        return jsonify({"error": "No code provided"}), 400
    profile = None
    if data.get("student_id"):
        profile = StudentProfile.load(PROFILES, data["student_id"])
    report = scan_submission(
        code, model, extractor, stats,
        student_profile=profile,
        filename=data.get("filename", "pasted code"),
    )
    return jsonify(report)


if __name__ == "__main__":
    print("CodePrint running at http://127.0.0.1:5000")
    # threaded=False: mc_predict toggles dropout mode on the shared model
    app.run(host="127.0.0.1", port=5000, threaded=False)
