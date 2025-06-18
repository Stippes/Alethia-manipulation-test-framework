import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import py_compile
import math
import pytest

from inter_rater import compute_cohens_kappa, compute_cronbach_alpha


def test_module_compiles():
    py_compile.compile('inter_rater.py')


def test_cohens_kappa_basic():
    r1 = [1, 1, 0, 1]
    r2 = [1, 0, 0, 1]
    kappa = compute_cohens_kappa(r1, r2)
    assert -1.0 <= kappa <= 1.0


def test_cohens_kappa_length_mismatch():
    with pytest.raises(ValueError):
        compute_cohens_kappa([1, 2], [1])


def test_cronbach_alpha_basic():
    ratings = [
        [1, 2, 3, 4],
        [1, 2, 4, 4],
        [2, 2, 3, 5],
    ]
    alpha = compute_cronbach_alpha(ratings)
    assert not math.isnan(alpha)
    assert -1.0 <= alpha <= 1.0


def test_cronbach_alpha_invalid_length():
    with pytest.raises(ValueError):
        compute_cronbach_alpha([[1, 2], [1]])
