"""How reliable is small-N seed reporting in RL papers?

Most published RL-for-robot-control comparisons report 3-5 random seeds. This module
asks the direct question: if you had only run 3 (or 5, 6, ...) of the seeds we actually
collected, how often would the naive significance test you'd have run on that small
subset have disagreed with the answer we get from the full data?

Where the total number of possible k-seed subsets is small, every one is enumerated
exactly (no Monte Carlo noise); only when the combinatorial space is large do we fall
back to random sampling, capped at `max_pairs` draws.
"""
import itertools
from math import comb
import numpy as np
import pandas as pd

import config as C
from analyze import load_results, rigorous_probability_of_improvement, naive_ttest


def _subsets(n, k, max_exact):
    """All C(n,k) index combinations if small enough, else None (caller falls back)."""
    if comb(n, k) <= max_exact:
        return list(itertools.combinations(range(n), k))
    return None


def _smalln_trials(n_a, n_b, k, rng, max_pairs):
    """(subset_of_a, subset_of_b) index pairs: exact cross-product if tractable, else a
    random sample of size max_pairs."""
    subs_a = _subsets(n_a, k, max_exact=int(max_pairs ** 0.5) + 5)
    subs_b = _subsets(n_b, k, max_exact=int(max_pairs ** 0.5) + 5)
    if subs_a is not None and subs_b is not None and len(subs_a) * len(subs_b) <= max_pairs:
        return list(itertools.product(subs_a, subs_b)), True
    pairs = []
    for _ in range(max_pairs):
        ia = tuple(rng.choice(n_a, size=k, replace=False))
        ib = tuple(rng.choice(n_b, size=k, replace=False))
        pairs.append((ia, ib))
    return pairs, False


def analyze_smalln_reliability(preset_name, ks=(3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 19),
                                rng_seed=0, max_pairs=4000):
    p = C.PRESETS[preset_name]
    scalar, _ = load_results(p["envs"], p["algos"], p["seeds"])
    rng = np.random.default_rng(rng_seed)

    rows = []
    for env_id in p["envs"]:
        algos_here = sorted({a for (e, a) in scalar if e == env_id})
        for a1, a2 in itertools.combinations(algos_here, 2):
            x, y = scalar[(env_id, a1)], scalar[(env_id, a2)]
            if len(x) < 4 or len(y) < 4:
                continue  # not enough seeds yet to even test small-N subsets meaningfully

            # "Ground truth": the rigorous, full-data bootstrap conclusion.
            poi, poi_lo, poi_hi = rigorous_probability_of_improvement(x, y)
            full_significant = bool((poi_lo > 0.5) or (poi_hi < 0.5))
            full_direction = "a>b" if poi > 0.5 else "a<b"

            for k in ks:
                if k >= min(len(x), len(y)):
                    continue
                pairs, exact = _smalln_trials(len(x), len(y), k, rng, max_pairs)
                n_trials = len(pairs)
                n_sig = n_disagree = n_flip = 0
                for ia, ib in pairs:
                    xs, ys = x[list(ia)], y[list(ib)]
                    p_naive = naive_ttest(xs, ys)
                    naive_sig = p_naive < 0.05
                    naive_direction = "a>b" if xs.mean() > ys.mean() else "a<b"
                    n_sig += naive_sig
                    if naive_sig != full_significant:
                        n_disagree += 1
                    if naive_sig and full_significant and naive_direction != full_direction:
                        n_flip += 1
                rows.append(dict(
                    env=env_id, algo_a=a1, algo_b=a2, k=k, n_trials=n_trials, exact=exact,
                    full_significant=full_significant, full_direction=full_direction,
                    pct_naive_significant=100 * n_sig / n_trials,
                    disagreement_rate_pct=100 * n_disagree / n_trials,
                    direction_flip_rate_pct=100 * n_flip / n_trials,
                ))
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(C.RESULTS / f"{preset_name}_smalln_reliability.csv", index=False)
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=list(C.PRESETS))
    args = ap.parse_args()
    df = analyze_smalln_reliability(args.preset)
    if len(df) == 0:
        print("No algorithm pairs with >=4 seeds each yet.")
    else:
        print(df.to_string(index=False))
        print(f"\n[saved] {args.preset}_smalln_reliability.csv")
