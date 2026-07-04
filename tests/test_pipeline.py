"""Smoke tests: feature extraction shape/stability and generator validity."""

import ast
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codeprint.data.generate_dataset import TASKS, _make_sample
from codeprint.features.extract import ALL_FEATURE_NAMES, FeatureExtractor

SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "suspicious_submission.py"


def test_feature_vector_shape_and_determinism():
    code = SAMPLE.read_text(encoding="utf-8")
    ex = FeatureExtractor().fit_lm([code])
    v1, v2 = ex.extract(code), ex.extract(code)
    assert v1.shape == (len(ALL_FEATURE_NAMES),)
    assert (v1 == v2).all()
    assert not any(map(lambda x: x != x, v1))  # no NaNs


def test_handles_broken_code():
    ex = FeatureExtractor().fit_lm(["print('x')"])
    vec = ex.extract("def broken(:\n    pass")
    assert vec.shape == (len(ALL_FEATURE_NAMES),)


def test_generated_samples_parse():
    rng = random.Random(0)
    for task in TASKS:
        for label in (0, 1):
            for _ in range(5):
                ast.parse(_make_sample(rng, task, label))


if __name__ == "__main__":
    test_feature_vector_shape_and_determinism()
    test_handles_broken_code()
    test_generated_samples_parse()
    print("all tests passed")
