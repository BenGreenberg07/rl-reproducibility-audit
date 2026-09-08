"""Figures for the RL reproducibility audit: learning curves with bootstrap CI bands,
IQM-with-CI comparison across algorithms, and naive-vs-rigorous significance summary.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from analyze import load_results, audit

plt.rcParams.update({"figure.dpi": 130, "font.size": 10})


def fig_learning_curves(preset_name):
    p = C.PRESETS[preset_name]
    for env_id in p["envs"]:
        fig, ax = plt.subplots(figsize=(7, 5))
        for algo in p["algos"]:
            curves = []
            for seed in p["seeds"]:
                path = C.result_path(env_id, algo, seed)
                if not path.exists():
                    continue
                d = np.load(path)
                curves.append((d["checkpoints"], d["eval_means"]))
            if not curves:
                continue
            checkpoints = curves[0][0]
            mat = np.array([c[1] for c in curves if len(c[1]) == len(checkpoints)])
            if len(mat) == 0:
                continue
            mean_curve = mat.mean(axis=0)
            se = mat.std(axis=0, ddof=1) / np.sqrt(max(len(mat), 1)) if len(mat) > 1 else np.zeros_like(mean_curve)
            ax.plot(checkpoints, mean_curve, label=f"{algo} (n={len(mat)})")
            ax.fill_between(checkpoints, mean_curve - 1.96 * se, mean_curve + 1.96 * se, alpha=0.2)
        ax.set_xlabel("Training timesteps")
        ax.set_ylabel("Evaluation return")
        ax.set_title(f"Learning curves: {env_id} (shaded = 95% CI across seeds)")
        ax.legend()
        fig.tight_layout()
        safe = env_id.replace("/", "_")
        fig.savefig(C.FIGURES / f"learning_curve_{safe}.png")
        plt.close(fig)
        print(f"learning_curve_{safe}.png")


def fig_iqm_comparison(preset_name):
    result = audit(preset_name, out_csv=False)
    if result is None:
        return None
    df_iqm, _, df_agg = result
    envs = df_iqm["env"].unique()
    fig, axes = plt.subplots(1, len(envs), figsize=(5 * len(envs), 4.5), squeeze=False)
    for ax, env_id in zip(axes[0], envs):
        sub = df_iqm[df_iqm["env"] == env_id].sort_values("iqm")
        y = np.arange(len(sub))
        ax.errorbar(sub["iqm"], y,
                    xerr=[sub["iqm"] - sub["iqm_ci_lo"], sub["iqm_ci_hi"] - sub["iqm"]],
                    fmt="o", capsize=4)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["algo"])
        ax.set_xlabel("IQM final return (95% bootstrap CI)")
        ax.set_title(env_id)
    fig.suptitle("Interquartile mean performance with bootstrap confidence intervals", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(C.FIGURES / f"iqm_comparison_{preset_name}.png")
    plt.close(fig)
    print(f"iqm_comparison_{preset_name}.png")
    return df_agg


def fig_aggregate(preset_name, df_agg):
    """The headline figure: one ranking of algorithms across the whole benchmark suite."""
    if df_agg is None or len(df_agg) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    sub = df_agg.sort_values("iqm_normalized")
    y = np.arange(len(sub))
    ax.errorbar(sub["iqm_normalized"], y,
                xerr=[sub["iqm_normalized"] - sub["ci_lo"], sub["ci_hi"] - sub["iqm_normalized"]],
                fmt="o", capsize=4, color="tab:red")
    ax.set_yticks(y)
    ax.set_yticklabels(sub["algo"])
    ax.set_xlabel("Normalized IQM (pooled across all tasks, 95% bootstrap CI)")
    ax.set_title(f"Aggregate ranking across all {preset_name} benchmark tasks")
    fig.tight_layout()
    fig.savefig(C.FIGURES / f"aggregate_ranking_{preset_name}.png")
    plt.close(fig)
    print(f"aggregate_ranking_{preset_name}.png")


def fig_smalln_reliability(preset_name):
    """Headline figure: how often would a naive small-N (k seeds) study have reached the
    wrong conclusion, relative to what the full data actually shows, as a function of k.
    One panel per environment, shared axes, so each panel shows only that environment's
    six algorithm pairs instead of cramming all pairs across all environments together."""
    from smalln_reliability import analyze_smalln_reliability
    df = analyze_smalln_reliability(preset_name)
    if df is None or len(df) == 0:
        print("skip fig_smalln_reliability (not enough seeds yet)")
        return
    envs = sorted(df["env"].unique())
    ncols = 2
    nrows = int(np.ceil(len(envs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.0 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    any_no_effect = False
    for ax, env_id in zip(axes, envs):
        for (a1, a2), sub in df[df["env"] == env_id].groupby(["algo_a", "algo_b"]):
            sub = sub.sort_values("k")
            has_effect = bool(sub["full_significant"].iloc[0])
            any_no_effect = any_no_effect or not has_effect
            style = "-o" if has_effect else "--s"
            ax.plot(sub["k"], sub["disagreement_rate_pct"], style, label=f"{a1} vs {a2}",
                     alpha=0.85, ms=4)
        ax.axhline(5, color="gray", lw=0.8, ls=":")
        ax.set_title(env_id, fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
    for ax in axes[len(envs):]:
        ax.axis("off")
    for i, ax in enumerate(axes[:len(envs)]):
        if i % ncols == 0:
            ax.set_ylabel("Disagreement rate (%)")
        if i >= len(envs) - ncols:
            ax.set_xlabel("Seeds used in the naive study (k)")
    subtitle = ("solid = a real effect exists; dashed = no real effect, false-positive rate"
                if any_no_effect else "every pair shown has a genuine effect at full sample size")
    fig.suptitle(f"How often would a naive k-seed study get it wrong?\n({subtitle})")
    fig.subplots_adjust(top=0.85, hspace=0.3, wspace=0.15)
    fig.savefig(C.FIGURES / f"smalln_disagreement_{preset_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"smalln_disagreement_{preset_name}.png")


def fig_power_curves(preset_name):
    """Theoretical companion: achieved statistical power vs. sample size, for each
    algorithm pair's actually-observed effect size. One panel per environment, shared
    axes, so each panel shows only that environment's six pairs."""
    from power_analysis import power_table, _safe_power
    df = power_table(preset_name)
    if df is None or len(df) == 0:
        print("skip fig_power_curves (not enough data yet)")
        return
    ns = np.arange(2, 41)
    envs = sorted(df["env"].unique())
    ncols = 2
    nrows = int(np.ceil(len(envs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.0 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, env_id in zip(axes, envs):
        for _, row in df[df["env"] == env_id].iterrows():
            d = row["observed_cohens_d"]
            powers = [_safe_power(d, n) for n in ns]
            ax.plot(ns, powers, label=f"{row['algo_a']} vs {row['algo_b']} (d={d:.2f})")
        ax.axhline(0.8, color="black", lw=1, ls="--")
        for k in (3, 5, 10):
            ax.axvline(k, color="gray", lw=0.6, ls=":")
        ax.set_title(env_id, fontsize=10)
        ax.legend(fontsize=7, loc="lower right")
    for ax in axes[len(envs):]:
        ax.axis("off")
    for i, ax in enumerate(axes[:len(envs)]):
        if i % ncols == 0:
            ax.set_ylabel("Statistical power")
        if i >= len(envs) - ncols:
            ax.set_xlabel("Number of seeds (n per algorithm)")
    fig.suptitle("Why small-N fails: power to detect the observed effect sizes\n"
                 "(dashed black = 80% power target; dotted gray = n = 3, 5, 10)")
    fig.subplots_adjust(top=0.85, hspace=0.3, wspace=0.15)
    fig.savefig(C.FIGURES / f"power_curves_{preset_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"power_curves_{preset_name}.png")


def main(preset_name):
    fig_learning_curves(preset_name)
    df_agg = fig_iqm_comparison(preset_name)
    fig_aggregate(preset_name, df_agg)
    fig_smalln_reliability(preset_name)
    fig_power_curves(preset_name)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=list(C.PRESETS))
    args = ap.parse_args()
    main(args.preset)
