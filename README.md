# How many seeds does a deep RL robot-control comparison actually need?

A statistical-rigor audit of deep RL algorithms on continuous-control benchmarks:
the full pipeline, raw per-seed results, and figures behind the paper "How Many
Seeds Are Enough? A Reliability Audit and Seed-Count Guide for Deep RL
Robot-Control Comparisons." The manuscript itself isn't published in this
repository (see "About the paper" below), but the submission links directly
here, since TNNLS review is single-blind (authors are visible to reviewers).

## The question

Papers comparing RL algorithms on robot-control tasks often report results from a
handful of seeds (sometimes as few as 3-5) and treat the best mean as "the winner."
This project asks: how much of that reported gap survives proper statistical
treatment, stratified bootstrap confidence intervals and interquartile-mean
aggregation across many seeds (the methodology from Agarwal et al., "Deep RL at the
Edge of the Statistical Precipice"), how often does a naive small-N significance
test (a plain t-test) disagree with the rigorous answer, and how many seeds would
it actually take to fix that?

## Headline results

- Four algorithms (PPO, SAC, TD3, A2C) x four MuJoCo tasks (Reacher-v5,
  HalfCheetah-v5, Hopper-v5, Walker2d-v5) x twenty seeds each = 320 independent
  training runs, 24 algorithm-pair comparisons.
- A 3-seed study gets a genuine effect wrong up to 94% of the time; for the
  smallest effect size in the dataset, that failure rate is still 95% at 19 of 20
  seeds.
- A conventional Welch's t-test misses two real effects entirely (on two
  independent environments) that a rigorous bootstrap method correctly detects,
  and both trace to the same mechanism: TD3's seed-to-seed variance, inflated by
  occasional training failures, breaks a mean-based test without breaking a
  rank-based one.
- One comparison (PPO vs. TD3, Walker2d-v5) never reaches significance under
  either method even at the full 20 seeds — a real example of "underpowered,
  unresolved" being the honest answer.
- A general, effect-size-indexed seed-count lookup table lets any author or
  reviewer look up the minimum seed count for 80%/90% power without rerunning
  this audit; see `results/PILOT4_seed_guide.csv`.

## Pipeline

1. `config.py`: defines environments, algorithms (PPO, SAC, TD3, A2C via
   Stable-Baselines3), seed counts, and training budgets, as presets:
   - `SMOKE`: proves the pipeline works in a few minutes (Pendulum-v1, 2 algos, 5
     seeds, 20k timesteps).
   - `PILOT`: Reacher-v5 and HalfCheetah-v5, 4 algos, 20 seeds, 200k timesteps.
   - `PILOT4`: the paper's actual dataset — `PILOT` plus Hopper-v5 and
     Walker2d-v5, same settings (4 tasks x 4 algos x 20 seeds x 200k timesteps,
     320 runs total). Content-addressed result filenames mean `PILOT4` reuses
     `PILOT`'s runs automatically and only trains the two new environments.
   - `FULL`: a heavier-training variant (1M timesteps) for a future extension;
     not run for this paper.
2. `train.py`: trains one (env, algo, seed) combination, periodically evaluates,
   and saves the learning curve plus a large final evaluation (raw per-episode
   returns, needed for the bootstrap analysis) to `results/`.
3. `sweep.py`: runs the full grid for a preset in parallel across CPU cores, skips
   combinations already on disk (safe to re-run / resume).
4. `analyze.py`: for each (env, algo), computes the interquartile mean and a 95%
   bootstrap CI; for each pair of algorithms on each environment, computes both a
   naive Welch's t-test (raw and Holm-corrected) and a rigorous bootstrap-CI'd
   probability-of-improvement (via `rliable`'s methodology), and reports where the
   two disagree. This is **Table III** in the paper. Also computes a single
   aggregate cross-environment ranking (min-max normalized IQM pooled across all
   tasks).
5. `figures.py`: learning curves with 95% CI bands, a per-environment
   IQM-with-CI comparison plot, the aggregate cross-environment ranking plot, the
   small-N disagreement-vs-k plot, and the power-curves plot.
6. `smalln_reliability.py`: for every algorithm pair and every small seed count k
   in {3,...,10,12,15,19}, exactly enumerates (or, where the combinatorial space
   is too large, densely samples 4,000 subsets) every possible k-seed sub-study
   drawable from the full data, runs the naive t-test on each, and reports how
   often that small-N conclusion would have disagreed with the full-data rigorous
   answer.
7. `power_analysis.py`: for each algorithm pair's actually-observed effect size
   (Cohen's d), computes the statistical power available at conventional seed
   counts and the seed count needed for 80%/90% power; also generates the
   general, algorithm-and-environment-independent seed-count lookup table
   (Table VI).
8. `generate_paper_tables.py`: renders every results CSV into a LaTeX table
   snippet (kept with the manuscript, not in this repo — see "About the paper"),
   so the paper's numbers are never hand-transcribed. Run this after any change
   to the underlying data.

## A methodological note

Observation and reward normalization (`VecNormalize`) is applied identically to
all four algorithms. Without this, PPO and A2C are known to underperform SAC and
TD3 on MuJoCo tasks for reasons that have nothing to do with the algorithms
themselves (see Engstrom et al., "Implementation Matters in Deep Policy
Gradients"), which would confound the actual question this audit is trying to
answer. Every training process is also pinned to a single CPU thread
(`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`, etc.) — without this, running
many training jobs in parallel causes severe thread-oversubscription slowdowns.

## Setup

```
python3 -m venv rlvenv
./rlvenv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
./rlvenv/bin/pip install -r requirements.txt
```

## Reproduce Table III (naive vs. rigorous significance) from the released data

The raw per-seed results for all 320 runs are checked into `results/*.npz`, so you
do not need to retrain anything to reproduce the paper's core table:

```
./rlvenv/bin/python3 analyze.py --preset PILOT4
```

This prints the full naive-vs-rigorous comparison for all 24 algorithm pairs (the
console table matches Table III in the paper) and writes
`results/PILOT4_pairwise_comparisons.csv`.

To reproduce the rest of the paper's numbers (small-N reliability, power
analysis, figures) from the same released data:

```
./rlvenv/bin/python3 smalln_reliability.py --preset PILOT4
./rlvenv/bin/python3 power_analysis.py --preset PILOT4
./rlvenv/bin/python3 figures.py --preset PILOT4
```

## Retrain from scratch

If you want to regenerate the raw results themselves rather than reuse the
checked-in `.npz` files (this took several hours across 9 CPU cores on a
consumer laptop for the full 320-run `PILOT4` grid):

```
./rlvenv/bin/python3 sweep.py --preset SMOKE     # sanity check, a few minutes
./rlvenv/bin/python3 sweep.py --preset PILOT4    # full paper-scale dataset
```

`sweep.py` skips any (env, algo, seed) combination whose result file already
exists, so deleting a subset of `results/*.npz` and rerunning only retrains what's
missing.

## About the paper

The manuscript is targeting **IEEE Transactions on Neural Networks and Learning
Systems (TNNLS)**. TNNLS review is single-blind (per its official "Information
for Authors": reviewers know the authors' identities, though not vice versa),
so there's no anonymity to protect and the submitted manuscript itself links
directly to this repository. The compiled paper is kept out of this
repository as a matter of scope (a reproducibility repo doesn't need the
manuscript itself, just the code, data, and figures it was built from), not
because of blind review.
