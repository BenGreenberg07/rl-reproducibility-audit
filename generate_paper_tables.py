"""Auto-generate LaTeX table snippets from the analysis CSVs, so the paper's numbers
always reflect the latest run rather than being hand-transcribed (and potentially stale
or transcribed wrong). Run this, then recompile paper/main.tex.
"""
import pandas as pd
import numpy as np
import config as C

OUT = C.ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def esc(s):
    return str(s).replace("_", "\\_")


def write(name, content):
    # Trailing "%" absorbs the phantom end-of-line space TeX otherwise inserts at
    # EOF, which is invisible for full table/figure environments but shows up as a
    # stray space wherever a snippet is \input inline mid-sentence (e.g. "94 %").
    (OUT / name).write_text(content.rstrip() + "%")
    print(f"wrote paper/tables/{name}")


def table_iqm_summary(preset):
    df = pd.read_csv(C.RESULTS / f"{preset}_iqm_summary.csv")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Per-environment interquartile mean (IQM) final return with 95\% stratified bootstrap confidence intervals, by algorithm.}",
        r"\label{tab:iqm_summary}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Environment & Algo. & $n$ & IQM & 95\% CI \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(f"{esc(r['env'])} & {r['algo']} & {int(r['n_seeds'])} & "
                     f"{r['iqm']:.2f} & [{r['iqm_ci_lo']:.2f}, {r['iqm_ci_hi']:.2f}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write(f"{preset}_iqm_summary.tex", "\n".join(lines))


def table_pairwise(preset):
    df = pd.read_csv(C.RESULTS / f"{preset}_pairwise_comparisons.csv")
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Pairwise algorithm comparisons: naive Welch's $t$-test (raw and Holm-corrected) versus the rigorous bootstrap-CI'd probability of improvement, on the full seed set. A ``significant'' rigorous call means the 95\% CI on $P(A>B)$ excludes 0.5.}",
        r"\label{tab:pairwise}",
        r"\begin{tabular}{llrrccccc}",
        r"\toprule",
        r"Env. & Pair ($A$ vs $B$) & $n_A$ & $n_B$ & Naive $p$ & Naive sig.\ & Holm sig.\ & $P(A{>}B)$ [95\% CI] & Rigorous sig.\ \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        pair = f"{r['algo_a']} vs {r['algo_b']}"
        naive_sig = r"\checkmark" if r["naive_significant_p05"] else "--"
        holm_sig = r"\checkmark" if r.get("naive_significant_holm_p05", False) else "--"
        rig_sig = r"\checkmark" if r["rigorous_significant"] else "--"
        lines.append(
            f"{esc(r['env'])} & {pair} & {int(r['n_a'])} & {int(r['n_b'])} & "
            f"{r['naive_ttest_p']:.2g} & {naive_sig} & {holm_sig} & "
            f"{r['prob_improvement']:.2f} [{r['poi_ci_lo']:.3f}, {r['poi_ci_hi']:.3f}] & {rig_sig} \\\\")
    n_disagree_naive = int((df["naive_significant_p05"] != df["rigorous_significant"]).sum())
    n_disagree_holm = int((df.get("naive_significant_holm_p05", df["naive_significant_p05"])
                          != df["rigorous_significant"]).sum())
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write(f"{preset}_pairwise.tex", "\n".join(lines))
    write(f"{preset}_pairwise_disagree_counts.tex",
          f"{n_disagree_naive} of the {len(df)} comparisons reported here "
          f"(Holm-corrected: {n_disagree_holm} of {len(df)})")


def table_smalln(preset):
    df = pd.read_csv(C.RESULTS / f"{preset}_smalln_reliability.csv")
    # headline rows: k=3, the most commonly reported seed count in the literature; the
    # full k=3..19 sweep for every pair is released in the CSV and plotted in Fig. 1.
    sub = df[df["k"].isin([3])].copy()
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Small-$N$ reliability at $k=3$ seeds: probability that a naive 3-seed study reaches a conclusion that disagrees with the full-data rigorous answer, computed by exact (or, where noted, sampled) enumeration of 3-seed subsets. The full $k=3,\ldots,19$ sweep for every pair is released with our data and plotted in Fig.~\ref{fig:smalln}.}",
        r"\label{tab:smalln}",
        r"\begin{tabular}{llcrrr}",
        r"\toprule",
        r"Env. & Pair & Real effect? & Trials & \% naive sig.\ & Disagreement rate \\",
        r"\midrule",
    ]
    for _, r in sub.sort_values(["env", "algo_a", "algo_b", "k"]).iterrows():
        real = r"\checkmark" if r["full_significant"] else "--"
        lines.append(
            f"{esc(r['env'])} & {r['algo_a']} vs {r['algo_b']} & {real} & "
            f"{int(r['n_trials'])} & {r['pct_naive_significant']:.1f}\\% & "
            f"{r['disagreement_rate_pct']:.1f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write(f"{preset}_smalln.tex", "\n".join(lines))

    # headline numbers for the abstract/intro: worst-case k=3 disagreement rate among
    # pairs with a genuine effect, and the seed count at which it drops under 5%.
    real_effects = df[df["full_significant"]]
    if len(real_effects):
        worst_k3 = real_effects[real_effects["k"] == 3]["disagreement_rate_pct"].max()
        write(f"{preset}_headline_worst_k3.tex", f"{worst_k3:.0f}")


def table_power(preset):
    df = pd.read_csv(C.RESULTS / f"{preset}_power_analysis.csv")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Observed effect sizes (Cohen's $d$) and the statistical power available at conventional small seed counts, with the seed count needed for 80\% power.}",
        r"\label{tab:power}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Env. & Pair & $d$ & $n{=}3$ & $n{=}5$ & $n{=}10$ & $n_{80\%}$ \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        n80 = r["n_seeds_needed_for_80pct_power"]
        n80s = f"{n80:.0f}" if np.isfinite(n80) else "--"
        p10 = r.get("power_n10", float("nan"))
        p10s = f"{p10:.2f}" if pd.notna(p10) and np.isfinite(p10) else "--"
        lines.append(
            f"{esc(r['env'])} & {r['algo_a']} vs {r['algo_b']} & {r['observed_cohens_d']:.2f} & "
            f"{r['power_n3']:.2f} & {r['power_n5']:.2f} & {p10s} & {n80s} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write(f"{preset}_power.tex", "\n".join(lines))


def table_seed_guide(preset):
    df = pd.read_csv(C.RESULTS / f"{preset}_seed_guide.csv")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Practical seed-count guide: minimum seeds for 80\%/90\% power to detect a two-sample effect of a given size $|d|$ (Welch's $t$-test, $\alpha{=}0.05$). Find your own observed $|d|$, round up to the nearest row.}",
        r"\label{tab:seed_guide}",
        r"\begin{tabular}{lrrl}",
        r"\toprule",
        r"$|d|$ & $n_{80\%}$ & $n_{90\%}$ & Note \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        n80 = r["n_for_80pct"]
        n90 = r["n_for_90pct"]
        n80s = f"{n80:.0f}" if pd.notna(n80) and np.isfinite(n80) else "--"
        n90s = f"{n90:.0f}" if pd.notna(n90) and np.isfinite(n90) else "--"
        note = r["label"] if isinstance(r["label"], str) and r["label"] else ""
        lines.append(f"{r['cohens_d']:.2f} & {n80s} & {n90s} & {note} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write(f"{preset}_seed_guide.tex", "\n".join(lines))


def table_aggregate(preset):
    df = pd.read_csv(C.RESULTS / f"{preset}_aggregate_across_envs.csv")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Aggregate cross-environment ranking: IQM of min-max normalized scores pooled across all benchmark tasks, with 95\% bootstrap CI.}",
        r"\label{tab:aggregate}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Algorithm & $n$ scores & Normalized IQM & 95\% CI \\",
        r"\midrule",
    ]
    for _, r in df.sort_values("iqm_normalized", ascending=False).iterrows():
        lines.append(f"{r['algo']} & {int(r['n_scores'])} & {r['iqm_normalized']:.3f} & "
                     f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write(f"{preset}_aggregate.tex", "\n".join(lines))


def n_runs_done(preset):
    p = C.PRESETS[preset]
    total = len(p["envs"]) * len(p["algos"]) * len(p["seeds"])
    done = len(list(C.RESULTS.glob("*.npz")))
    write(f"{preset}_n_runs.tex", f"{done}/{total}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    args = ap.parse_args()
    table_iqm_summary(args.preset)
    table_pairwise(args.preset)
    table_smalln(args.preset)
    table_power(args.preset)
    table_seed_guide(args.preset)
    table_aggregate(args.preset)
    n_runs_done(args.preset)
    print("All tables regenerated.")
