# CodePrint 🔍

**Deep learning detection of AI-generated code in programming assignments — with evidence, not just a score.**

MOSS and JPlag catch students copying *from each other*. Text-based AI detectors (GPTZero, Turnitin) treat code as prose and fall apart on it. CodePrint models *how code is written* — syntax-tree structure, stylometry, and statistical burstiness — and answers the question those tools can't: **"did this student write this, or did an LLM?"**

## What makes CodePrint different (USPs)

1. **Code-native detection, not text detection.** Features come from the AST (nesting habits, branching shape, function-size uniformity), tokenizer-level stylometry (naming-convention entropy, quote mixing, spacing consistency, commented-out code), and a predictability model — not from prose-oriented perplexity over words.

2. **Per-student authorship drift — our signature feature.** CodePrint builds a stylometric fingerprint per student from their known-authentic early work, then flags submissions that *deviate from that student's own history*. This catches AI use by students whose work would otherwise pass an absolute check, and — just as importantly — protects students whose natural style simply resembles LLM output. No mainstream tool does longitudinal, per-author baselining for code.

3. **Calibrated three-way verdicts with abstention.** Temperature-scaled probabilities plus Monte-Carlo-dropout uncertainty produce `AI-GENERATED / HUMAN-WRITTEN / INCONCLUSIVE`. The model *declines to accuse* when unsure. False accusations are the failure mode that gets detectors banned from campuses; abstention is a first-class design goal, not an afterthought.

4. **Evidence reports built for integrity hearings.** Every flag ships with plain-English evidence ("every function has a docstring — 3.1σ above the human norm; spacing around `=` is perfectly consistent, historically this student's was not"). An instructor gets discussion points for a conversation with the student, not an unexplainable percentage.

5. **Assignment-cohort normalization.** Constrained assignments make everyone's code converge; absolute detectors over-flag entire classes. CodePrint scores each submission relative to its own cohort's distribution (robust median/MAD z-scores), separating "this assignment naturally looks generated" from "this submission is the outlier".

6. **Evasion-resistant by construction.** Renaming variables and re-formatting defeats fingerprint-style detectors; AST-structural features and burstiness statistics survive both. Conversely, the drift detector catches the *opposite* evasion — a student lightly "messying up" ChatGPT output still won't reproduce their own historical fingerprint.

7. **100% local and private.** No student code leaves the machine — no cloud API, no third-party retention of student IP. FERPA/GDPR-friendly by architecture. The predictability language model is trained on the course's own corpus.

8. **Honest evaluation methodology.** Train/val/test splits are grouped by student, so no author appears in two splits — the leakage that inflates most detector benchmarks is designed out.

### Competitor comparison

| Capability | MOSS / JPlag | GPTZero / Turnitin AI | Copyleaks Codeleaks | **CodePrint** |
|---|---|---|---|---|
| Student↔student copying | ✅ | ❌ | ✅ | ➖ (out of scope) |
| AI-generated code detection | ❌ | ⚠️ prose-based | ✅ score only | ✅ code-native |
| Per-student longitudinal baseline | ❌ | ❌ | ❌ | ✅ |
| Explains *why* it flagged | ⚠️ diff view | ❌ | ⚠️ limited | ✅ feature-level evidence |
| Abstains when uncertain | n/a | ❌ | ❌ | ✅ calibrated |
| Cohort-relative scoring | ✅ implicitly | ❌ | ❌ | ✅ explicit |
| Runs fully offline | ❌ (cloud) | ❌ (cloud) | ❌ (cloud) | ✅ |

## Architecture

```
submission.py
   ├─ Stylometry (22 features)        naming entropy, docstring/type-hint coverage,
   │                                  quote & spacing consistency, debug residue…
   ├─ AST structure (78 features)     node distribution, depth/branching, function-size
   │                                  uniformity, hashed parent→child path bigrams
   └─ Predictability (5 features)     cohort char-trigram LM: burstiness, repetition
            │
            ▼
   Standardize → DetectorNet (MLP, dropout) → temperature scaling
            │
            ├─ MC-dropout ⇒ uncertainty ⇒ AI / HUMAN / INCONCLUSIVE
            ├─ StudentProfile ⇒ authorship-drift score
            ├─ Cohort robust-z ⇒ class-relative outliers
            └─ Evidence report (markdown / JSON)
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. build the demo corpus (synthetic; see note below)
python -m codeprint generate-data --out data/dataset.jsonl

# 2. train + calibrate (CPU, ~1 min)
python -m codeprint train --data data/dataset.jsonl --out models

# 3. scan a submission -> evidence report
python -m codeprint scan examples/suspicious_submission.py --model models

# 4. per-student drift: build a baseline from known-authentic work, then scan
python -m codeprint profile week1.py week2.py week3.py --student s042 --out profiles
python -m codeprint scan hw4.py --model models --student s042 --profiles profiles

# 5. score a whole assignment directory with cohort normalization
python -m codeprint cohort submissions/hw4/ --model models
```

Run tests: `python tests/test_pipeline.py`

## Using real data

The bundled generator produces a *synthetic* corpus so the pipeline runs offline
end-to-end; its accuracy numbers demonstrate the pipeline, not real-world
performance. For deployment, retrain on:

- **Human class:** the course's own pre-LLM-era submissions (pre-2022 semesters
  are guaranteed human), or proctored in-class work.
- **AI class:** solutions to the same assignment prompts generated by the models
  students actually use (GPT-4/o-series, Claude, Gemini, Copilot), across several
  prompting styles including "write this like a student would".

Everything downstream — training, calibration, drift, cohort scoring, reports —
works unchanged on real data (`{"code", "label", "student_id", "assignment"}` JSONL).

## Ethics & intended use

CodePrint is a **triage and conversation tool, not a judge**. Verdicts are
calibrated and the model abstains when unsure, but no detector is proof.
Intended workflow: flag → review evidence → talk to the student. Never auto-penalize
from a score.

## Roadmap

- Multi-language support via tree-sitter (Java, C/C++, JS)
- GNN encoder over the raw AST replacing hashed path features
- Keystroke/commit-history process signals as a third evidence channel
- LMS plugins (Moodle/Canvas) around the JSON report API
