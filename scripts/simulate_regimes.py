"""Заранее заданные null-сценарии; production analyze не изменяется."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from labsignal.stats import analyze

PROTOCOL = Path(__file__).resolve().parents[1] / "docs/simulation-regimes.json"


def interval(success, total):
    bounds = binomtest(success, total).proportion_ci(confidence_level=0.95, method="exact")
    return {
        "count": success,
        "trials": total,
        "rate": success / total,
        "interval_95": [float(bounds.low), float(bounds.high)],
    }


def generate(rng, regime, n):
    sizes = (900, 300) if regime == "allocation_75_25" else (n, n)
    groups = []
    for index, size in enumerate(sizes):
        converted = rng.binomial(1, 0.15, size)
        pre = rng.gamma(2, 10, size)
        noise = (
            rng.lognormal(2, 2.2, size)
            if regime == "heavy_tail_lognormal"
            else rng.gamma(2, 20, size)
        )
        revenue = converted * noise + 0.8 * pre
        rows = [
            {"converted": bool(c), "revenue": float(y), "pre_value": float(x)}
            for c, y, x in zip(converted, revenue, pre)
        ]
        if regime == "dropout_mcar_25pct":
            rows = [row for row, keep in zip(rows, rng.random(size) >= 0.25) if keep]
        elif regime == "dropout_outcome_selected_balanced":
            # Ровно одинаковое число зрелых показов скрывает outcome-dependent selection от SRM.
            order = np.argsort(revenue)
            kept = order[:360] if index == 0 else order[-360:]
            rows = [rows[i] for i in sorted(kept)]
        groups.append(rows)
    # Независимый пилот с противоположной связью; данные эксперимента не используются.
    pilot_x = rng.gamma(2, 10, 1000)
    sign = -0.8 if regime == "wrong_independent_pilot" else 0.8
    pilot_y = sign * pilot_x + rng.normal(0, 3, 1000)
    # Пилот соответствует API: денежные значения неотрицательны; сдвиг не меняет covariance.
    pilot_y = pilot_y - min(pilot_y) + 1
    theta = float(np.cov(pilot_x, pilot_y, ddof=0)[0, 1] / np.var(pilot_x))
    return groups, theta


def discoveries(report):
    return any(report[metric]["significant"] for metric in ("conversion", "revenue_cuped"))


def simulate(protocol):
    results = []
    for regime_index, regime in enumerate(protocol["regimes"]):
        rng = np.random.default_rng(np.random.SeedSequence([protocol["seed"], regime_index]))
        fixed_false = planned_false = fixed_srm = planned_srm = 0
        unavailable = 0
        theta_sum = 0.0
        sample_sizes = []
        for _ in range(protocol["trials"]):
            groups, theta = generate(rng, regime, protocol["assigned_per_arm"])
            a, b = groups
            theta_sum += theta
            sample_sizes.append([len(a), len(b)])
            fixed = analyze(a, b, theta=theta)
            fixed_false += discoveries(fixed)
            fixed_srm += fixed["srm_failed"]
            reports = []
            for size in protocol["planned_looks"]:
                if min(len(a), len(b)) < size:
                    unavailable += 1
                    continue
                # При этой временной проверке наблюдаем пропорциональное число зрелых показов.
                progress = size / min(len(a), len(b))
                counts = (max(size, round(len(a) * progress)), max(size, round(len(b) * progress)))
                reports.append(
                    analyze(
                        a[:size],
                        b[:size],
                        theta=theta,
                        alpha=0.05 / len(protocol["planned_looks"]),
                        counts=counts,
                    )
                )
            planned_false += any(discoveries(report) for report in reports)
            planned_srm += any(report["srm_failed"] for report in reports)
        results.append(
            {
                "regime": regime,
                "fixed": {
                    "false_conclusions": interval(fixed_false, protocol["trials"]),
                    "srm_rejections": interval(fixed_srm, protocol["trials"]),
                },
                "planned": {
                    "false_conclusions": interval(planned_false, protocol["trials"]),
                    "srm_rejections": interval(planned_srm, protocol["trials"]),
                    "unavailable_looks": unavailable,
                },
                "mean_theta": theta_sum / protocol["trials"],
                "observed_n_min": np.min(sample_sizes, axis=0).tolist(),
                "observed_n_max": np.max(sample_sizes, axis=0).tolist(),
                "population_effect": 0,
            }
        )
    return results


def main():
    source = PROTOCOL.read_bytes()
    protocol = json.loads(source)
    started = time.perf_counter()
    results = simulate(protocol)
    print(
        json.dumps(
            {
                "protocol": protocol,
                "protocol_sha256": hashlib.sha256(source).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "results": results,
                "seconds": time.perf_counter() - started,
                "numpy": np.__version__,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
