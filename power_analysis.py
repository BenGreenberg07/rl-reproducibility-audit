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
from scipy import stats
from statsmodels.stats.power import TTestIndPower

import config as C
from analyze import load_results

_power = TTestIndPower()


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return (x.mean() - y.mean()) / pooled_std if pooled_std > 0 else 0.0


def _power_scipy(d, n, alpha=0.05):
    """Two-sample t-test power via the noncentral-t distribution, computed directly
    with scipy rather than through statsmodels' wrapper. More numerically stable at
    extreme effect sizes, though it can still fail at the very largest ones (see
    _power_montecarlo below)."""
    df = 2 * n - 2
    nc = d * np.sqrt(n / 2)
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    return 1 - stats.nct.cdf(t_crit, df, nc) + stats.nct.cdf(-t_crit, df, nc)


def _power_montecarlo(d, n, alpha=0.05, n_sims=200_000, seed=0):
    """Empirical power via direct simulation: draw n_sims two-sample datasets under the
    given effect size and count how often Welch's t-test rejects at alpha. Used only as
    a last resort when the closed-form noncentral-t computation itself is numerically
    unstable (huge effect sizes at very small n), since it can't fail the way a
    closed-form CDF evaluation can."""
    rng = np.random.default_rng(seed)
    rejections = 0
    batch = 20_000
    for start in range(0, n_sims, batch):
        m = min(batch, n_sims - start)
        g1 = rng.normal(d, 1, size=(m, n))
        g2 = rng.normal(0, 1, size=(m, n))
        _, p = stats.ttest_ind(g1, g2, axis=1, equal_var=False)
        rejections += int((p < alpha).sum())
    return rejections / n_sims


def _safe_power(d, n):
    """Cascading fallback: statsmodels first (fast, usually fine), then direct scipy
    (more stable at extreme effect sizes), then Monte Carlo simulation (always finite,
    used only for the rare cases where even the direct noncentral-t CDF evaluation
    itself is numerically unstable). Every value returned is a real number; nothing is
    silently reported as NaN in the paper's tables."""
    try:
        val = float(_power.power(effect_size=d, nobs1=n, alpha=0.05, ratio=1.0))
        if np.isfinite(val):
            return val
    except Exception:
        pass
    try:
        val = float(_power_scipy(d, n))
        if np.isfinite(val):
            return val
    except Exception:
        pass
    return float(_power_montecarlo(d, n))


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
