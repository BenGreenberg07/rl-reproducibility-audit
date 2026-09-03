"""Statistical power analysis: the theoretical companion to smalln_reliability.py.

Rather than just showing empirically that small-N conclusions are unreliable, this
computes, for the effect sizes we actually observed, how much statistical power a
3/5/10-seed study has to detect them, and what sample size would be needed for
conventionally adequate (80%) power. This explains *why* small-N fails, not just that
it does.
"""
import itertools
import numpy as np
import pandas as pd
from statsmodels.stats.power import TTestIndPower

import config as C
from analyze import load_results

_power = TTestIndPower()


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return (x.mean() - y.mean()) / pooled_std if pooled_std > 0 else 0.0


def _safe_power(d, n):
    """statsmodels' noncentral-t power computation can return NaN with no exception at
    extreme (effect_size, n) combinations (e.g. n=2 with a very large effect size hits a
    numerical edge case), so check finiteness directly rather than relying on try/except."""
    try:
        val = float(_power.power(effect_size=d, nobs1=n, alpha=0.05, ratio=1.0))
    except Exception:
        return float("nan")
    return val if np.isfinite(val) else float("nan")


def _n_needed_for_power(d, target=0.8, grid=range(2, 201)):
    """Smallest n with power >= target, found by direct grid search (robust to the
    solver's non-convergence at extreme effect sizes) rather than trusting solve_power
    alone."""
    for n in grid:
        pw = _safe_power(d, n)
        if np.isfinite(pw) and pw >= target:
            return float(n)
    try:
        val = float(_power.solve_power(effect_size=d, alpha=0.05, power=target, ratio=1.0))
        return val if np.isfinite(val) else float("nan")
    except Exception:
        return float("nan")


def power_table(preset_name, sample_sizes=(3, 5, 8, 10, 15, 20, 30, 50)):
    p = C.PRESETS[preset_name]
    scalar, _ = load_results(p["envs"], p["algos"], p["seeds"])

    rows = []
    for env_id in p["envs"]:
        algos_here = sorted({a for (e, a) in scalar if e == env_id})
        for a1, a2 in itertools.combinations(algos_here, 2):
            x, y = scalar[(env_id, a1)], scalar[(env_id, a2)]
            if len(x) < 2 or len(y) < 2:
                continue
            d = abs(cohens_d(x, y))
            row = dict(env=env_id, algo_a=a1, algo_b=a2, observed_cohens_d=d)
            for n in sample_sizes:
                row[f"power_n{n}"] = _safe_power(d, n)
            row["n_seeds_needed_for_80pct_power"] = _n_needed_for_power(d)
            rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(C.RESULTS / f"{preset_name}_power_analysis.csv", index=False)
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=list(C.PRESETS))
    args = ap.parse_args()
    df = power_table(args.preset)
    if len(df) == 0:
        print("No algorithm pairs available yet.")
    else:
        cols = ["env", "algo_a", "algo_b", "observed_cohens_d",
                "power_n3", "power_n5", "power_n10", "n_seeds_needed_for_80pct_power"]
        print(df[cols].to_string(index=False))
        print(f"\n[saved] {args.preset}_power_analysis.csv")
