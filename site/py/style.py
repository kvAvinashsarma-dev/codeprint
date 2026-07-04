"""Stylometric features: how the author writes, independent of what the code does.

These are the features MOSS/JPlag never look at (they compare submissions to
each other) and text-based AI detectors can't compute (they treat code as prose).
Each feature is a cheap, interpretable scalar so verdicts can cite them as evidence.
"""

from __future__ import annotations

import ast
import io
import math
import re
import tokenize
from collections import Counter

# feature name -> (description, direction hint: +1 means higher value looks more AI-like)
STYLE_FEATURES: dict[str, tuple[str, int]] = {
    "avg_identifier_length": ("Average length of variable/function names", +1),
    "single_letter_var_ratio": ("Fraction of single-letter identifiers", -1),
    "naming_convention_entropy": ("Inconsistency across naming styles (snake/camel/etc.)", -1),
    "comment_density": ("Comment lines per code line", 0),
    "avg_comment_length": ("Average comment length in characters", +1),
    "commented_out_code_ratio": ("Fraction of comments that are disabled code", -1),
    "docstring_coverage": ("Fraction of functions/classes carrying a docstring", +1),
    "type_hint_coverage": ("Fraction of parameters/returns with type annotations", +1),
    "blank_line_ratio": ("Blank lines per total line", 0),
    "avg_line_length": ("Mean line length", 0),
    "line_length_std": ("Variability of line lengths", -1),
    "long_line_ratio": ("Fraction of lines over 79 chars", -1),
    "quote_mix_entropy": ("Mixing of single vs double quotes", -1),
    "fstring_ratio": ("Fraction of strings using f-string formatting", +1),
    "magic_number_density": ("Unnamed numeric literals per code line", -1),
    "print_density": ("print() calls per code line", -1),
    "operator_space_consistency": ("Consistency of spacing around '='", +1),
    "trailing_whitespace_ratio": ("Lines with trailing whitespace", -1),
    "todo_marker_density": ("TODO/FIXME/XXX markers per comment", -1),
    "comment_capitalized_ratio": ("Comments starting with a capital letter", +1),
    "dead_code_ratio": ("Unreachable / unused statement heuristic", -1),
    "exception_usage": ("try/except blocks per function", +1),
}

STYLE_FEATURE_NAMES = list(STYLE_FEATURES.keys())

_NAME_SNAKE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)+$")
_NAME_CAMEL = re.compile(r"^[a-z]+([A-Z][a-z0-9]*)+$")
_NAME_UPPER = re.compile(r"^[A-Z][A-Z0-9_]*$")
_NAME_PLAIN = re.compile(r"^[a-z][a-z0-9]*$")
_ASSIGN_LINE = re.compile(r"^\s*[\w.\[\]]+( ?)=( ?)[^=]")
_CODE_HINT = re.compile(r"[=()\[\]]|^\s*(print|return|if|for|while|import|def)\b")


def _entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _collect_identifiers(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.arg):
            names.append(node.arg)
    return [n for n in names if not (n.startswith("__") and n.endswith("__"))]


def _looks_like_code(comment: str) -> bool:
    text = comment.strip()
    if not text or not _CODE_HINT.search(text):
        return False
    try:
        parsed = ast.parse(text)
    except SyntaxError:
        return False
    return bool(parsed.body)


def extract_style_features(source: str) -> dict[str, float]:
    lines = source.splitlines()
    total_lines = max(len(lines), 1)
    stripped = [ln.strip() for ln in lines]
    code_lines = max(sum(1 for ln in stripped if ln and not ln.startswith("#")), 1)
    blank_lines = sum(1 for ln in stripped if not ln)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = ast.parse("")

    comments: list[str] = []
    strings: list[str] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                comments.append(tok.string.lstrip("#").strip())
            elif tok.type == tokenize.STRING:
                strings.append(tok.string)
    except (tokenize.TokenError, SyntaxError, IndentationError):
        pass

    identifiers = _collect_identifiers(tree)
    conv_counts = Counter()
    for name in identifiers:
        if len(name) == 1:
            conv_counts["single"] += 1
        elif _NAME_SNAKE.match(name):
            conv_counts["snake"] += 1
        elif _NAME_CAMEL.match(name):
            conv_counts["camel"] += 1
        elif _NAME_UPPER.match(name):
            conv_counts["upper"] += 1
        elif _NAME_PLAIN.match(name):
            conv_counts["plain"] += 1
        else:
            conv_counts["other"] += 1

    defs = [n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    with_doc = sum(1 for d in defs if ast.get_docstring(d))
    if ast.get_docstring(tree):
        with_doc += 1

    funcs = [d for d in defs if isinstance(d, (ast.FunctionDef, ast.AsyncFunctionDef))]
    total_args = annotated = 0
    for f in funcs:
        args = list(f.args.args) + list(f.args.kwonlyargs)
        args = [a for a in args if a.arg not in ("self", "cls")]
        total_args += len(args) + 1  # +1 for the return annotation slot
        annotated += sum(1 for a in args if a.annotation is not None)
        annotated += 1 if f.returns is not None else 0

    magic = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool) and node.value not in (0, 1, -1):
            magic += 1

    prints = sum(1 for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "print")
    tries = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Try))

    single_q = sum(1 for s in strings if s.lstrip("rbfuRBFU").startswith("'"))
    double_q = sum(1 for s in strings if s.lstrip("rbfuRBFU").startswith('"'))
    fstrings = sum(1 for s in strings if "f" in s[:2].lower())

    spaced = unspaced = 0
    for ln in lines:
        m = _ASSIGN_LINE.match(ln)
        if m:
            if m.group(1) == " " and m.group(2) == " ":
                spaced += 1
            elif m.group(1) == "" and m.group(2) == "":
                unspaced += 1
    assigns = spaced + unspaced

    line_lens = [len(ln) for ln in lines if ln.strip()]
    mean_len = _safe_div(sum(line_lens), len(line_lens))
    std_len = math.sqrt(_safe_div(sum((l - mean_len) ** 2 for l in line_lens),
                                  len(line_lens))) if line_lens else 0.0

    todo = sum(1 for c in comments if re.search(r"\b(todo|fixme|xxx|hack)\b", c, re.I))
    cap = sum(1 for c in comments if c[:1].isupper())
    dead = sum(1 for c in comments if _looks_like_code(c))

    return {
        "avg_identifier_length": _safe_div(sum(map(len, identifiers)), len(identifiers)),
        "single_letter_var_ratio": _safe_div(conv_counts["single"], len(identifiers)),
        "naming_convention_entropy": _entropy(list(conv_counts.values())),
        "comment_density": len(comments) / code_lines,
        "avg_comment_length": _safe_div(sum(map(len, comments)), len(comments)),
        "commented_out_code_ratio": _safe_div(dead, len(comments)),
        "docstring_coverage": _safe_div(with_doc, len(defs) + 1),
        "type_hint_coverage": _safe_div(annotated, total_args),
        "blank_line_ratio": blank_lines / total_lines,
        "avg_line_length": mean_len,
        "line_length_std": std_len,
        "long_line_ratio": _safe_div(sum(1 for l in line_lens if l > 79), len(line_lens)),
        "quote_mix_entropy": _entropy([single_q, double_q]),
        "fstring_ratio": _safe_div(fstrings, len(strings)),
        "magic_number_density": magic / code_lines,
        "print_density": prints / code_lines,
        "operator_space_consistency": _safe_div(max(spaced, unspaced), assigns) if assigns else 1.0,
        "trailing_whitespace_ratio": sum(1 for ln in lines if ln != ln.rstrip()) / total_lines,
        "todo_marker_density": _safe_div(todo, len(comments)),
        "comment_capitalized_ratio": _safe_div(cap, len(comments)),
        "dead_code_ratio": _safe_div(dead, code_lines),
        "exception_usage": _safe_div(tries, max(len(funcs), 1)),
    }
