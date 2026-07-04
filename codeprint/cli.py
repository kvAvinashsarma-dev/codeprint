"""Command-line interface.

    python -m codeprint generate-data --out data/dataset.jsonl
    python -m codeprint train --data data/dataset.jsonl --out models/
    python -m codeprint profile --student s042 --out profiles/ file1.py file2.py file3.py
    python -m codeprint scan submission.py --model models/ [--student s042 --profiles profiles/]
    python -m codeprint cohort submissions_dir/ --model models/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_generate_data(args: argparse.Namespace) -> None:
    from .data.generate_dataset import generate
    n = generate(args.out, n_students=args.students, seed=args.seed)
    print(f"Wrote {n} samples to {args.out}")


def cmd_train(args: argparse.Namespace) -> None:
    from .model.train import train
    metrics = train(args.data, args.out, epochs=args.epochs)
    print(json.dumps(metrics, indent=2))


def cmd_profile(args: argparse.Namespace) -> None:
    from .baseline import StudentProfile
    from .features.extract import FeatureExtractor
    codes = [Path(f).read_text(encoding="utf-8", errors="replace") for f in args.files]
    profile = StudentProfile.build(args.student, codes, FeatureExtractor())
    path = profile.save(args.out)
    print(f"Profile for {args.student} built from {len(codes)} submissions -> {path}")


def cmd_scan(args: argparse.Namespace) -> None:
    from .baseline import StudentProfile
    from .model.train import load_artifacts
    from .report import render_json, render_markdown, scan_submission

    model, extractor, stats = load_artifacts(args.model)
    profile = None
    if args.student and args.profiles:
        profile = StudentProfile.load(args.profiles, args.student)
        if profile is None:
            print(f"(no stored profile for {args.student}; drift analysis skipped)",
                  file=sys.stderr)

    for file in args.files:
        code = Path(file).read_text(encoding="utf-8", errors="replace")
        report = scan_submission(code, model, extractor, stats,
                                 student_profile=profile, filename=file)
        print(render_json(report) if args.format == "json" else render_markdown(report))
        print()


def cmd_cohort(args: argparse.Namespace) -> None:
    from .cohort import score_cohort
    from .model.train import load_artifacts

    model, extractor, stats = load_artifacts(args.model)
    files = sorted(Path(args.directory).glob("*.py"))
    if not files:
        sys.exit(f"No .py files in {args.directory}")
    results = score_cohort(files, model, extractor, stats)
    print(f"{'file':<40} {'P(AI)':>7} {'cohort z':>9}  flag")
    for r in results:
        flag = "OUTLIER" if r["cohort_outlier"] else ""
        print(f"{Path(r['file']).name:<40} {r['p_ai']:>7.1%} {r['cohort_z']:>9.2f}  {flag}")


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="codeprint",
                                     description="AI-generated code detection for programming assignments")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate-data", help="generate the synthetic demo dataset")
    p.add_argument("--out", default="data/dataset.jsonl")
    p.add_argument("--students", type=int, default=80)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_generate_data)

    p = sub.add_parser("train", help="train the detector")
    p.add_argument("--data", default="data/dataset.jsonl")
    p.add_argument("--out", default="models")
    p.add_argument("--epochs", type=int, default=60)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("profile", help="build a student's style baseline")
    p.add_argument("files", nargs="+")
    p.add_argument("--student", required=True)
    p.add_argument("--out", default="profiles")
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("scan", help="scan submissions and emit evidence reports")
    p.add_argument("files", nargs="+")
    p.add_argument("--model", default="models")
    p.add_argument("--student", default=None)
    p.add_argument("--profiles", default="profiles")
    p.add_argument("--format", choices=["md", "json"], default="md")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("cohort", help="score a whole assignment directory with cohort normalization")
    p.add_argument("directory")
    p.add_argument("--model", default="models")
    p.set_defaults(func=cmd_cohort)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
