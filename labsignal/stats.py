import math

import numpy as np
from scipy.stats import binomtest, fisher_exact, norm, t, ttest_ind


def holm(values):
    ordered = sorted(range(len(values)), key=lambda i: values[i])
    adjusted, previous = [1.0] * len(values), 0.0
    for rank, index in enumerate(ordered):
        previous = max(previous, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, previous)
    return adjusted


def analyze(a, b, theta=0.0, alpha=0.05, counts=None):
    if min(len(a), len(b)) < 2:
        raise ValueError("Нужны хотя бы два зрелых показа в каждой группе")
    counts = counts or (len(a), len(b))
    srm = float(binomtest(counts[0], sum(counts), 0.5).pvalue)
    ca, cb = sum(row["converted"] for row in a), sum(row["converted"] for row in b)
    conversion_p = float(fisher_exact([[ca, len(a) - ca], [cb, len(b) - cb]]).pvalue)
    xa = np.array([row["revenue"] - theta * row["pre_value"] for row in a])
    xb = np.array([row["revenue"] - theta * row["pre_value"] for row in b])
    if np.var(xa) == 0 and np.var(xb) == 0:
        revenue_p = 1.0
    else:
        revenue_p = float(ttest_ind(xa, xb, equal_var=False).pvalue)
        if not math.isfinite(revenue_p):
            revenue_p = 1.0
    corrected = holm([conversion_p, revenue_p])
    # Интервалы консервативно учитывают две метрики; для planned alpha уже делится на число просмотров.
    z = float(norm.ppf(1 - alpha / 8))

    def wilson(success, n):
        p = success / n
        center = (p + z * z / (2 * n)) / (1 + z * z / n)
        margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return center - margin, center + margin

    la, ua = wilson(ca, len(a))
    lb, ub = wilson(cb, len(b))
    va, vb = float(np.var(xa, ddof=1) / len(a)), float(np.var(xb, ddof=1) / len(b))
    revenue_ci = None
    if va + vb > 0:
        degrees = (va + vb) ** 2 / (va * va / (len(a) - 1) + vb * vb / (len(b) - 1))
        margin = float(t.ppf(1 - alpha / 4, degrees)) * math.sqrt(va + vb)
        effect = float(xb.mean() - xa.mean())
        revenue_ci = [effect - margin, effect + margin]

    return {
        "n_a": len(a),
        "n_b": len(b),
        "srm_p": srm,
        "srm_failed": srm < 0.001,
        "alpha": alpha,
        "conversion": {
            "effect": cb / len(b) - ca / len(a),
            "interval": [lb - ua, ub - la],
            "p": conversion_p,
            "adjusted_p": corrected[0],
            "significant": corrected[0] <= alpha and srm >= 0.001,
        },
        "revenue_cuped": {
            "effect": float(xb.mean() - xa.mean()),
            "interval": revenue_ci,
            "variance_supported": revenue_ci is not None,
            "p": revenue_p,
            "adjusted_p": corrected[1],
            "significant": corrected[1] <= alpha and srm >= 0.001,
        },
        "theta": theta,
        "interval_method": "Консервативные интервалы Wilson для разности долей и Welch для CUPED; Bonferroni по двум метрикам",
    }
