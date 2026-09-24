import json

import numpy as np
from scipy.stats import binomtest

from labsignal.stats import analyze

rng = np.random.default_rng(42)
trials = 400
results = {}
for effect in [0.0, 0.10]:
    discoveries = 0
    sequential_discoveries = 0
    for _ in range(trials):
        groups = []
        for probability in [0.15, 0.15 + effect]:
            converted = rng.binomial(1, probability, 600)
            revenue = converted * rng.gamma(2, 20, 600)
            groups.append(
                [
                    {"converted": bool(c), "revenue": float(r), "pre_value": 0}
                    for c, r in zip(converted, revenue)
                ]
            )
        fixed = analyze(groups[0], groups[1])
        discoveries += any(
            fixed[metric]["significant"] for metric in ["conversion", "revenue_cuped"]
        )
        sequential = [
            analyze(groups[0][:size], groups[1][:size], alpha=0.05 / 3) for size in [200, 400, 600]
        ]
        sequential_discoveries += any(
            report[metric]["significant"]
            for report in sequential
            for metric in ["conversion", "revenue_cuped"]
        )
    interval = binomtest(discoveries, trials).proportion_ci()
    results[str(effect)] = {
        "fixed_discovery_rate": discoveries / trials,
        "fixed_rate_95_interval": [interval.low, interval.high],
        "planned_discovery_rate": sequential_discoveries / trials,
    }
print(
    json.dumps(
        {"seed": 42, "trials_per_effect": trials, "per_group": 600, "results": results}, indent=2
    )
)
assert results["0.0"]["fixed_discovery_rate"] < 0.1
assert results["0.0"]["planned_discovery_rate"] < 0.1
assert results["0.1"]["fixed_discovery_rate"] > 0.8
