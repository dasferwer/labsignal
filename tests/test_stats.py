import numpy as np

from labsignal.stats import analyze, holm


def row(converted, revenue=0, pre=0):
    return {"converted": converted, "revenue": revenue, "pre_value": pre}


def test_holm_and_no_effect():
    assert holm([0.04, 0.01]) == [0.04, 0.02]
    report = analyze([row(False)] * 20, [row(False)] * 20)
    assert report["conversion"]["p"] == 1
    assert not report["revenue_cuped"]["significant"]


def test_known_effect_and_srm_gate():
    a, b = [row(False)] * 100, [row(True, 10)] * 100
    report = analyze(a, b)
    assert report["conversion"]["significant"]
    assert report["conversion"]["effect"] == 1
    skew = analyze(a, b, counts=(190, 10))
    assert skew["srm_failed"]
    assert not skew["conversion"]["significant"]


def test_fixed_cuped_coefficient_reduces_noise():
    rng = np.random.default_rng(42)
    pre_a, pre_b = rng.normal(100, 20, 500), rng.normal(100, 20, 500)
    a = [row(False, p + e, p) for p, e in zip(pre_a, rng.normal(0, 1, 500))]
    b = [row(False, p + e + 1, p) for p, e in zip(pre_b, rng.normal(0, 1, 500))]
    adjusted = analyze(a, b, theta=1)
    assert abs(adjusted["revenue_cuped"]["effect"] - 1) < 0.2
    assert adjusted["revenue_cuped"]["significant"]
