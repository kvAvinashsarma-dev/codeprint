"""Structural features from the abstract syntax tree.

Rename-proof and reformat-proof: a student can rename every variable and rerun
a formatter, but the *shape* of LLM-generated code (branching habits, guard
clauses, comprehension usage, uniform function sizes) survives those evasions.
"""

from __future__ import annotations

import ast
import hashlib
import math

NODE_VOCAB = [
    "FunctionDef", "AsyncFunctionDef", "ClassDef", "Return", "Assign", "AugAssign",
    "AnnAssign", "For", "While", "If", "With", "Raise", "Try", "ExceptHandler",
    "Import", "ImportFrom", "Expr", "Call", "Name", "Attribute", "Constant",
    "BinOp", "UnaryOp", "BoolOp", "Compare", "IfExp", "ListComp", "SetComp",
    "DictComp", "GeneratorExp", "Lambda", "Subscript", "List", "Tuple", "Dict",
    "Set", "Starred", "keyword", "arg", "alias",
]
N_BIGRAM_BUCKETS = 32

AST_FEATURE_NAMES = (
    [f"node_{n}" for n in NODE_VOCAB]
    + ["max_depth", "mean_depth", "mean_branching",
       "func_count", "mean_func_size", "func_size_cv"]
    + [f"astpath_{i}" for i in range(N_BIGRAM_BUCKETS)]
)


def _bucket(parent: str, child: str) -> int:
    digest = hashlib.md5(f"{parent}>{child}".encode()).digest()
    return digest[0] % N_BIGRAM_BUCKETS


def extract_ast_features(source: str) -> list[float]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = ast.parse("")

    node_counts = {name: 0 for name in NODE_VOCAB}
    bigrams = [0] * N_BIGRAM_BUCKETS
    depths: list[int] = []
    children_counts: list[int] = []
    total_nodes = 0

    def visit(node: ast.AST, depth: int) -> None:
        nonlocal total_nodes
        total_nodes += 1
        depths.append(depth)
        name = type(node).__name__
        if name in node_counts:
            node_counts[name] += 1
        kids = list(ast.iter_child_nodes(node))
        children_counts.append(len(kids))
        for kid in kids:
            bigrams[_bucket(name, type(kid).__name__)] += 1
            visit(kid, depth + 1)

    visit(tree, 0)
    total = max(total_nodes, 1)

    funcs = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    func_sizes = [sum(1 for _ in ast.walk(f)) for f in funcs]
    mean_size = sum(func_sizes) / len(func_sizes) if func_sizes else 0.0
    if len(func_sizes) > 1 and mean_size:
        var = sum((s - mean_size) ** 2 for s in func_sizes) / len(func_sizes)
        size_cv = math.sqrt(var) / mean_size
    else:
        size_cv = 0.0

    total_bigrams = max(sum(bigrams), 1)
    return (
        [node_counts[n] / total for n in NODE_VOCAB]
        + [float(max(depths)), sum(depths) / len(depths),
           sum(children_counts) / len(children_counts),
           float(len(funcs)), mean_size, size_cv]
        + [b / total_bigrams for b in bigrams]
    )
