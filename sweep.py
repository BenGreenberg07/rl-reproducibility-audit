"""Run the full (env x algo x seed) grid for a given preset, in parallel across CPU cores.

Usage:
    python sweep.py --preset SMOKE
    python sweep.py --preset PILOT --workers 4
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import config as C


def launch(env_id, algo, seed, p):
    out_path = C.result_path(env_id, algo, seed)
    if out_path.exists():
        return f"[skip] {out_path.name}"
    log_path = C.LOGS / (out_path.stem + ".log")
    cmd = [
        sys.executable, "train.py",
        "--env", env_id, "--algo", algo, "--seed", str(seed),
        "--timesteps", str(p["total_timesteps"]),
        "--eval-freq", str(p["eval_freq"]),
        "--n-eval-episodes", str(p["n_eval_episodes"]),
        "--final-eval-episodes", str(p["final_eval_episodes"]),
    ]
    # Belt-and-suspenders: also set these in the subprocess's own environment (train.py
    # sets them internally too, but setting them here guarantees they're present before
    # the interpreter even starts, in case any C extension reads them at process init).
    env = os.environ.copy()
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[v] = "1"
    with open(log_path, "w") as logf:
        result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(C.ROOT), env=env)
    status = "ok" if result.returncode == 0 else f"FAILED (see {log_path})"
    return f"[{status}] {out_path.name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=list(C.PRESETS))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    p = C.PRESETS[args.preset]
    jobs = [(env_id, algo, seed) for env_id in p["envs"] for algo in p["algos"] for seed in p["seeds"]]
    print(f"Preset {args.preset}: {len(jobs)} runs across {args.workers} workers")
    print(f"  envs={p['envs']}  algos={p['algos']}  seeds={p['seeds']}  timesteps={p['total_timesteps']}")

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(launch, env_id, algo, seed, p): (env_id, algo, seed)
                   for env_id, algo, seed in jobs}
        done_n = 0
        for fut in as_completed(futures):
            done_n += 1
            print(f"({done_n}/{len(jobs)}) {fut.result()}")


if __name__ == "__main__":
    main()
