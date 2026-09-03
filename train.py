"""Train one (algorithm, environment, seed) combination and record evaluation curves.

Usage:
    python train.py --env Pendulum-v1 --algo PPO --seed 0 --timesteps 20000 --eval-freq 2000
"""
import os
# Must be set before numpy/torch are imported: each of these training processes runs
# alone within its own OS process (sweep.py runs many in parallel), but PyTorch/BLAS
# default to spawning a multi-threaded pool per process regardless. With N parallel
# processes each grabbing several threads, they oversubscribe the machine's cores many
# times over and each run slows down drastically. Pinning every process to 1 thread
# lets sweep.py's process-level parallelism actually map cleanly onto the CPU cores.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import time
import numpy as np
import gymnasium as gym
import torch
torch.set_num_threads(1)
from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import config as C

ALGO_CLASSES = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "A2C": A2C}


def make_vec_env(env_id, seed, norm_reward, training):
    """A single-env VecEnv wrapped in VecNormalize.

    Observation (and, for training only, reward) normalization is applied uniformly
    to all four algorithms. Without this, PPO/A2C are known to underperform SAC/TD3
    on MuJoCo tasks for reasons unrelated to the algorithms themselves (see Engstrom
    et al., "Implementation Matters in Deep Policy Gradients"), which would confound
    the noise-vs-real-difference question this audit is trying to answer.
    """
    def _make():
        env = gym.make(env_id)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    vec = DummyVecEnv([_make])
    vec = VecNormalize(vec, norm_obs=True, norm_reward=norm_reward, training=training)
    return vec


def run_one(env_id, algo_name, seed, total_timesteps, eval_freq, n_eval_episodes,
            final_eval_episodes, verbose=0):
    out_path = C.result_path(env_id, algo_name, seed)
    if out_path.exists():
        print(f"[skip] {out_path.name} already exists")
        return out_path

    t0 = time.time()
    train_env = make_vec_env(env_id, seed, norm_reward=True, training=True)
    eval_env = make_vec_env(env_id, seed + 10_000, norm_reward=False, training=False)

    ModelCls = ALGO_CLASSES[algo_name]
    model = ModelCls("MlpPolicy", train_env, seed=seed, verbose=verbose)

    def sync_eval_normalization():
        # copy running obs stats from the training env into the (frozen) eval env
        eval_env.obs_rms = copy.deepcopy(train_env.obs_rms)

    checkpoints, means, stds = [], [], []
    steps_done = 0
    while steps_done < total_timesteps:
        chunk = min(eval_freq, total_timesteps - steps_done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        steps_done += chunk
        sync_eval_normalization()
        mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes,
                                        deterministic=True)
        checkpoints.append(steps_done)
        means.append(mean_r)
        stds.append(std_r)
        print(f"[{env_id}/{algo_name}/seed{seed}] step={steps_done} eval_mean={mean_r:.2f}")

    # final evaluation: collect raw per-episode returns directly (needed for the
    # bootstrap/rliable analysis), then derive the mean/std from those same episodes
    # rather than paying for a second, separate batch of rollouts.
    sync_eval_normalization()
    final_returns = []
    for _ in range(final_eval_episodes):
        obs = eval_env.reset()
        done = [False]
        ep_ret = 0.0
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            ep_ret += reward[0]
        final_returns.append(ep_ret)
    final_returns = np.array(final_returns)
    final_mean, final_std = float(final_returns.mean()), float(final_returns.std(ddof=1))

    elapsed = time.time() - t0
    np.savez(out_path,
             env_id=env_id, algo=algo_name, seed=seed,
             checkpoints=np.array(checkpoints), eval_means=np.array(means), eval_stds=np.array(stds),
             final_mean=final_mean, final_std=final_std,
             final_returns=final_returns,
             elapsed_seconds=elapsed)
    print(f"[done] {out_path.name} in {elapsed:.1f}s, final_mean={final_mean:.2f}")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--algo", required=True, choices=list(ALGO_CLASSES))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--timesteps", type=int, required=True)
    p.add_argument("--eval-freq", type=int, required=True)
    p.add_argument("--n-eval-episodes", type=int, default=10)
    p.add_argument("--final-eval-episodes", type=int, default=20)
    args = p.parse_args()
    run_one(args.env, args.algo, args.seed, args.timesteps, args.eval_freq,
            args.n_eval_episodes, args.final_eval_episodes)
