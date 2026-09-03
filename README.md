# How much of reported RL-for-robot-control improvement is noise?

A statistical-rigor audit of deep RL algorithms on continuous-control benchmarks.

## The question

Papers comparing RL algorithms on robot-control tasks often report results from a
handful of seeds (sometimes as few as 3-5) and treat the best mean as "the winner."
This project asks: how much of that reported gap survives proper statistical
treatment, stratified bootstrap confidence intervals and interquartile-mean
aggregation across many seeds (the methodology from Agarwal et al., "Deep RL at the
Edge of the Statistical Precipice"), and how often does a naive small-N significance
test (a plain t-test) disagree with the rigorous answer?

## Pipeline

1. `config.py`: defines environments, algorithms (PPO, SAC, TD3, A2C via
   Stable-Baselines3), seed counts, and training budgets, as three presets:
   - `SMOKE`: proves the pipeline works in a few minutes (Pendulum-v1, 2 algos, 5 seeds,
     20k timesteps).
   - `PILOT`: real MuJoCo robot-control tasks (Reacher-v5, HalfCheetah-v5), 4 algos,
     10 seeds, 200k timesteps; runs in hours.
   - `FULL`: paper-scale (4 MuJoCo tasks, 4 algos, 20 seeds, 1M timesteps); run this
     in the background / overnight.
2. `train.py`: trains one (env, algo, seed) combination, periodically evaluates, and
   saves the learning curve plus a large final evaluation (raw per-episode returns,
   needed for the bootstrap analysis) to `results/`.
3. `sweep.py`: runs the full grid for a preset in parallel across CPU cores, skips
   combinations already on disk (safe to re-run / resume).
4. `analyze.py`: for each (env, algo), computes the interquartile mean and a 95%
   bootstrap CI; for each pair of algorithms on each environment, computes both a
   naive Welch's t-test and a rigorous bootstrap-CI'd probability-of-improvement
   (via `rliable`'s methodology), and reports where the two disagree. When a preset
   has more than one environment, also computes a single aggregate ranking (min-max
   normalized IQM pooled across all tasks) — the headline result of this kind of study.
5. `figures.py`: learning curves with 95% CI bands, a per-environment IQM-with-CI
   comparison plot, and (when applicable) the aggregate cross-environment ranking plot.
6. `smalln_reliability.py`: the paper's central contribution. For every algorithm pair
   and every small seed count k in {3,...,9}, exactly enumerates (or, where the
   combinatorial space is too large, densely samples) every possible k-seed sub-study
   drawable from the full data, runs the naive t-test on each, and reports how often
   that small-N conclusion would have disagreed with the full-data rigorous answer.
7. `power_analysis.py`: for each algorithm pair's actually-observed effect size (Cohen's
   d), computes the statistical power available at conventional seed counts and the
   seed count needed for 80% power, explaining the small-N results mechanistically.
8. `generate_paper_tables.py`: renders every results CSV into a LaTeX table snippet
   under `paper/tables/`, so the paper's numbers are never hand-transcribed. Run this
   after any change to the underlying data, then recompile `paper/main.tex`.
9. `paper/`: the IEEE-style manuscript (`main.tex` + `refs.bib`), draft-compiled with a
   local TinyTeX install. See `paper/main.pdf` for the current draft.

## A methodological note

Observation and reward normalization (`VecNormalize`) is applied identically to all
four algorithms. Without this, PPO and A2C are known to underperform SAC and TD3 on
MuJoCo tasks for reasons that have nothing to do with the algorithms themselves (see
Engstrom et al., "Implementation Matters in Deep Policy Gradients"), which would
confound the actual question this audit is trying to answer.

## Run it

```
./rlvenv/bin/python3 sweep.py --preset SMOKE
./rlvenv/bin/python3 analyze.py --preset SMOKE
./rlvenv/bin/python3 figures.py --preset SMOKE
```

Swap `SMOKE` for `PILOT` or `FULL` once you're ready to scale up. `PILOT` is a
reasonable target for real, publishable results; `FULL` is the paper-scale version
and should be kicked off as a background run given the runtime.

## Status

`SMOKE` ran end-to-end and confirmed the pipeline works. `PILOT` (80 training runs: 2
tasks x 4 algorithms x 10 seeds x 200k timesteps) ran overnight; see `results/` for the
raw per-seed outputs and `paper/main.pdf` for the compiled draft manuscript built from
them. `FULL` (20 seeds, 4 tasks, 1M timesteps) has not been run; it would substantially
strengthen the small-N reliability analysis (more seeds to draw sub-studies from, more
environments) if pursued as a follow-up.
