import numpy as np
import pytest

from llmtm.agreement import bootstrap_ci, cohen_kappa, krippendorff_alpha_nominal, precision_recall_f1

N = None


def test_krippendorff_alpha_reproduces_the_worked_example():
    """Krippendorff (2011), 'Computing Krippendorff's alpha-reliability', nominal example:
    4 observers, 12 units, missing values; alpha = 0.743."""
    observers = [
        [1, 2, 3, 3, 2, 1, 4, 1, 2, N, N, N],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, N, 3],
        [N, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, N],
        [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, N],
    ]
    units = list(zip(*observers, strict=True))
    assert krippendorff_alpha_nominal(units) == pytest.approx(0.743, abs=0.0005)


def test_alpha_is_one_for_perfect_agreement_and_near_zero_for_chance():
    rng = np.random.default_rng(0)
    x = rng.integers(3, size=2000)
    assert krippendorff_alpha_nominal(list(zip(x, x, strict=True))) == 1
    y = rng.integers(3, size=2000)
    assert abs(krippendorff_alpha_nominal(list(zip(x, y, strict=True)))) < 0.05


def test_kappa_and_f1_on_a_small_table():
    a = [1, 1, 1, 0, 0, 0, 1, 0]
    b = [1, 1, 0, 0, 0, 1, 1, 0]
    assert cohen_kappa(a, b) == pytest.approx(0.5)
    p, r, f = precision_recall_f1(a, b)
    assert (p, r, f) == pytest.approx((0.75, 0.75, 0.75))


def test_bootstrap_interval_covers_the_mean():
    x = np.random.default_rng(1).normal(size=500)
    lo, hi = bootstrap_ci(lambda idx: x[idx].mean(), len(x), n_boot=500)
    assert lo < x.mean() < hi and hi - lo < 0.25
