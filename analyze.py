"""Statistical rigor audit: compare RL algorithms with proper stratified bootstrap CIs
and interquartile-mean aggregation (rliable methodology), and contrast against what a
naive small-N significance test would have concluded on the same data.
"""
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import bootstrap as scipy_bootstrap
from statsmodels.stats.multitest import multipletests
from rliable import metrics as rmetrics

import config as C


def load_results(env_ids, algos, seeds):
    """Return {(env_id, algo): np.array of shape (n_seeds,) final-eval mean returns}
    and {(env_id, algo): 2D array (n_seeds, final_eval_episodes) of raw episode returns}."""
    scalar = {}
    raw = {}
    for env_id in env_ids:
        for algo in algos:
            means, raws = [], []
            for seed in seeds:
                path = C.result_path(env_id, algo, seed)
                if not path.exists():
                    continue
                d = np.load(path, allow_pickle=True)
                means.append(float(d["final_mean"]))
                raws.append(d["final_returns"])
            if means:
                scalar[(env_id, algo)] = np.array(means)
                raw[(env_id, algo)] = np.array(raws)  # (n_seeds, n_episodes)
    return scalar, raw


def min_max_normalize(scalar, env_ids):
    """Per-environment min-max normalization to [0, 1] across all algos/seeds, so tasks
    with different reward scales can be aggregated together (standard rliable practice)."""
    normed = {}
    for env_id in env_ids:
        algos_present = {k[1] for k in scalar if k[0] == env_id}
        if not algos_present:
            continue  # no results yet for this environment, skip rather than crash
        vals = np.concatenate([scalar[(env_id, a)] for a in algos_present])
        lo, hi = vals.min(), vals.max()
        span = hi - lo if hi > lo else 1.0
        for (e, a), v in scalar.items():
            if e == env_id:
                normed[(e, a)] = (v - lo) / span
    return normed


def _iqm_stat(a):
    return float(rmetrics.aggregate_iqm(np.asarray(a).reshape(1, -1)))


def iqm_with_ci(x, n_resamples=5000, confidence=0.95, rng=None):
    """Interquartile mean point estimate + bootstrap CI via scipy.stats.bootstrap."""
    x = np.asarray(x, dtype=float)
    point = _iqm_stat(x)
    res = scipy_bootstrap((x,), _iqm_stat,
                          n_resamples=n_resamples, confidence_level=confidence,
                          method="percentile", random_state=rng, vectorized=False)
    return float(point), float(res.confidence_interval.low), float(res.confidence_interval.high)


def naive_ttest(x, y):
    """What a typical paper reporting N~3-5 seeds would do: Welch's t-test, p<0.05."""
    t, p = stats.ttest_ind(x, y, equal_var=False)
    return p


def rigorous_probability_of_improvement(x, y, n_resamples=2000, rng=None):
    """P(algo x > algo y) via rliable's metric, with a bootstrap CI over seed resampling.

    rliable expects scores shaped (num_runs, num_tasks); here each seed is one run on a
    single task (this environment), so the array is (n_seeds, 1).
    """
    rng = rng or np.random.default_rng(0)
    point = rmetrics.probability_of_improvement(x.reshape(-1, 1), y.reshape(-1, 1))
    boot = []
    for _ in range(n_resamples):
        xs = rng.choice(x, size=len(x), replace=True)
        ys = rng.choice(y, size=len(y), replace=True)
        boot.append(rmetrics.probability_of_improvement(xs.reshape(-1, 1), ys.reshape(-1, 1)))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def aggregate_across_envs(scalar, env_ids, algos):
    """For each algorithm, an IQM (+bootstrap CI) of per-environment min-max-normalized
    scores, pooled across all environments and seeds. This is the headline number rliable
    is built for: a single, statistically honest ranking across the whole benchmark suite,
    not just one task at a time.
    """
    normed = min_max_normalize(scalar, env_ids)
    rows = []
    for algo in algos:
        parts = [normed[(e, algo)] for e in env_ids if (e, algo) in normed]
        if not parts:
            continue  # this algorithm has no completed runs yet on any environment
        pooled = np.concatenate(parts)
        if len(pooled) < 2:
            continue
        iqm, lo, hi = iqm_with_ci(pooled)
        rows.append(dict(algo=algo, n_scores=len(pooled), iqm_normalized=iqm,
                         ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows).sort_values("iqm_normalized", ascending=False)


def audit(preset_name, out_csv=None):
    p = C.PRESETS[preset_name]
    scalar, raw = load_results(p["envs"], p["algos"], p["seeds"])
    if not scalar:
        print("No results found yet, run sweep.py first.")
        return None

    rows = []
    # Per-(env, algo) IQM with CI
    for (env_id, algo), vals in scalar.items():
        n = len(vals)
        if n < 2:
            continue
        iqm, lo, hi = iqm_with_ci(vals)
        rows.append(dict(env=env_id, algo=algo, n_seeds=n,
                         mean=vals.mean(), std=vals.std(ddof=1),
                         iqm=iqm, iqm_ci_lo=lo, iqm_ci_hi=hi))
    df_iqm = pd.DataFrame(rows)

    # Pairwise algo comparisons per environment: naive t-test vs rigorous bootstrap PoI
    pair_rows = []
    for env_id in p["envs"]:
        algos_here = sorted({a for (e, a) in scalar if e == env_id})
        for a1, a2 in itertools.combinations(algos_here, 2):
            x, y = scalar[(env_id, a1)], scalar[(env_id, a2)]
            if len(x) < 2 or len(y) < 2:
                continue
            p_naive = naive_ttest(x, y)
            poi, poi_lo, poi_hi = rigorous_probability_of_improvement(x, y)
            # does the rigorous 95% CI for P(a1>a2) exclude 0.5 (a "real" difference)?
            rigorous_significant = (poi_lo > 0.5) or (poi_hi < 0.5)
            pair_rows.append(dict(
                env=env_id, algo_a=a1, algo_b=a2,
                n_a=len(x), n_b=len(y),
                naive_ttest_p=p_naive, naive_significant_p05=p_naive < 0.05,
                prob_improvement=poi, poi_ci_lo=poi_lo, poi_ci_hi=poi_hi,
                rigorous_significant=rigorous_significant,
            ))
    df_pairs = pd.DataFrame(pair_rows)

    # Holm-Bonferroni correction of the naive p-values, applied per environment (i.e.
    # correcting across the C(n_algos, 2) pairwise tests run within that environment).
    # This shows a second, independent way naive practice inflates false "significant"
    # findings, on top of the small-N problem: most papers don't correct for running
    # several pairwise comparisons at once either.
    if len(df_pairs):
        df_pairs["naive_p_holm"] = np.nan
        df_pairs["naive_significant_holm_p05"] = False
        for env_id in df_pairs["env"].unique():
            mask = df_pairs["env"] == env_id
            reject, p_corrected, _, _ = multipletests(
                df_pairs.loc[mask, "naive_ttest_p"].values, alpha=0.05, method="holm")
            df_pairs.loc[mask, "naive_p_holm"] = p_corrected
            df_pairs.loc[mask, "naive_significant_holm_p05"] = reject

    print("\n=== Per-(env, algo) IQM with 95% bootstrap CI ===")
    print(df_iqm.to_string(index=False))
    print("\n=== Pairwise algorithm comparisons: naive t-test vs rigorous bootstrap ===")
    print(df_pairs.to_string(index=False))

    if len(df_pairs):
        disagree = df_pairs[df_pairs["naive_significant_p05"] != df_pairs["rigorous_significant"]]
        disagree_holm = df_pairs[df_pairs["naive_significant_holm_p05"] != df_pairs["rigorous_significant"]]
        print(f"\nNaive (uncorrected) vs rigorous DISAGREE on significance in "
              f"{len(disagree)}/{len(df_pairs)} algorithm-pair comparisons.")
        print(f"Naive (Holm-corrected for multiple comparisons) vs rigorous DISAGREE in "
              f"{len(disagree_holm)}/{len(df_pairs)}.")
        if len(disagree):
            print(disagree[["env", "algo_a", "algo_b", "naive_ttest_p",
                            "naive_significant_p05", "prob_improvement", "rigorous_significant"]]
                  .to_string(index=False))

    df_agg = None
    if len(p["envs"]) > 1:
        df_agg = aggregate_across_envs(scalar, p["envs"], p["algos"])
        print("\n=== Aggregate cross-environment IQM (min-max normalized, 95% bootstrap CI) ===")
        print(df_agg.to_string(index=False))

    if out_csv:
        df_iqm.to_csv(C.RESULTS / f"{preset_name}_iqm_summary.csv", index=False)
        df_pairs.to_csv(C.RESULTS / f"{preset_name}_pairwise_comparisons.csv", index=False)
        saved = f"{preset_name}_iqm_summary.csv , {preset_name}_pairwise_comparisons.csv"
        if df_agg is not None:
            df_agg.to_csv(C.RESULTS / f"{preset_name}_aggregate_across_envs.csv", index=False)
            saved += f" , {preset_name}_aggregate_across_envs.csv"
        print(f"\n[saved] {saved}")

    return df_iqm, df_pairs, df_agg


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=list(C.PRESETS))
    args = ap.parse_args()
    audit(args.preset, out_csv=True)
