"""Agreement and accuracy statistics, with bootstrap intervals over units."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def cohen_kappa(a, b) -> float:
    """Cohen's kappa for two raters on the same units (any hashable categories)."""
    a, b = np.asarray(a), np.asarray(b)
    cats = np.unique(np.concatenate([a, b]))
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def krippendorff_alpha_nominal(matrix) -> float:
    """Krippendorff's alpha for nominal data.

    `matrix` is units x coders; missing values are None or NaN. Units coded by fewer than two
    coders are ignored. alpha = 1 - (n - 1) * sum_{c != k} o_ck / sum_{c != k} n_c n_k, where o is
    the coincidence matrix (Krippendorff, 2011).
    """
    rows = []
    for row in matrix:
        vals = [v for v in row if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if len(vals) >= 2:
            rows.append(vals)
    cats = sorted({v for r in rows for v in r}, key=str)
    idx = {c: i for i, c in enumerate(cats)}
    o = np.zeros((len(cats), len(cats)))
    for vals in rows:
        m = len(vals)
        counts = np.zeros(len(cats))
        for v in vals:
            counts[idx[v]] += 1
        pairs = np.outer(counts, counts) - np.diag(counts)
        o += pairs / (m - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    disagree = o.sum() - np.trace(o)
    expected = n_c.sum() ** 2 - (n_c**2).sum()
    if expected == 0:
        return 1.0
    return float(1 - (n - 1) * disagree / expected)


def precision_recall_f1(truth, pred) -> tuple[float, float, float]:
    truth, pred = np.asarray(truth, bool), np.asarray(pred, bool)
    tp = np.sum(truth & pred)
    precision = tp / pred.sum() if pred.sum() else np.nan
    recall = tp / truth.sum() if truth.sum() else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else np.nan
    return float(precision), float(recall), float(f1)


def bootstrap_ci(
    statistic: Callable[[np.ndarray], float], n_units: int, n_boot: int = 1000, level: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile interval of `statistic(indices)` over resamples of the units."""
    rng = np.random.default_rng(seed)
    draws = np.array([statistic(rng.integers(n_units, size=n_units)) for _ in range(n_boot)])
    draws = draws[~np.isnan(draws)]
    lo, hi = np.quantile(draws, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)
