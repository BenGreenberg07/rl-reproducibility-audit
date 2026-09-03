"""Central config for the RL reproducibility/statistical-rigor audit.

Research question: how much of the performance gap reported between RL algorithms
on standard robot-control benchmarks survives rigorous statistical treatment
(stratified bootstrap CIs, interquartile mean across many seeds), versus how much
would look "significant" under the naive small-N comparisons common in papers.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
FIGURES = ROOT / "figures"
for d in (RESULTS, LOGS, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

ALGOS = ["PPO", "SAC", "TD3", "A2C"]

# Presets trade off runtime against how close the result is to "paper scale".
PRESETS = {
    # Fast: proves the pipeline end-to-end in a few minutes.
    "SMOKE": dict(
        envs=["Pendulum-v1"],
        algos=["PPO", "SAC"],
        seeds=list(range(5)),
        total_timesteps=20_000,
        eval_freq=2_000,
        n_eval_episodes=10,
        final_eval_episodes=20,
    ),
    # Moderate: real MuJoCo robot-control tasks, still laptop-feasible (~hours).
    "PILOT": dict(
        envs=["Reacher-v5", "HalfCheetah-v5"],
        algos=["PPO", "SAC", "TD3", "A2C"],
        seeds=list(range(10)),
        total_timesteps=200_000,
        eval_freq=10_000,
        n_eval_episodes=10,
        final_eval_episodes=30,
    ),
    # Paper scale: run overnight / in background, more envs, more seeds, longer training.
    "FULL": dict(
        envs=["Reacher-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"],
        algos=["PPO", "SAC", "TD3", "A2C"],
        seeds=list(range(20)),
        total_timesteps=1_000_000,
        eval_freq=25_000,
        n_eval_episodes=10,
        final_eval_episodes=50,
    ),
}

def result_path(env_id, algo, seed):
    safe_env = env_id.replace("/", "_")
    return RESULTS / f"{safe_env}__{algo}__seed{seed}.npz"
