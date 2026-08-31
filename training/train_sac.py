from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gymnasium
import mujoco
import numpy as np
import stable_baselines3
import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from env.ur5e_planning_env import UR5ePlanningEnv, load_config
from evaluation.evaluate import evaluate_model, save_results


def _software_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "mujoco": mujoco.__version__,
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


def train(
    config_path: str,
    seed: int,
    timesteps: int,
    run_name: str,
    eval_episodes: int | None,
    save_trajectories: bool,
) -> Path:
    config = load_config(config_path)
    rl_cfg = config["rl"]
    policy_cfg = config["policy"]

    out_dir = REPO_ROOT / "experiments" / "sac" / run_name / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_log_dir = REPO_ROOT / "runs" / "sac" / run_name / f"seed_{seed}"

    env = Monitor(UR5ePlanningEnv(config=config))

    ent_coef = rl_cfg["ent_coef"]
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=float(rl_cfg["learning_rate"]),
        buffer_size=int(rl_cfg["buffer_size"]),
        learning_starts=int(rl_cfg["learning_starts"]),
        batch_size=int(rl_cfg["batch_size"]),
        tau=float(rl_cfg["tau"]),
        gamma=float(rl_cfg["gamma"]),
        train_freq=int(rl_cfg["train_freq"]),
        gradient_steps=int(rl_cfg["gradient_steps"]),
        ent_coef=ent_coef,
        policy_kwargs={"net_arch": list(policy_cfg["net_arch"])},
        seed=seed,
        verbose=1,
        tensorboard_log=str(tb_log_dir),
    )

    print(f"Training SAC: run_name={run_name} seed={seed} timesteps={timesteps}")
    print(f"Config: {config_path}")
    t0 = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    training_wall_time_s = time.time() - t0
    print(f"\nTraining {timesteps} timesteps took {training_wall_time_s:.1f}s "
          f"({timesteps / training_wall_time_s:.1f} steps/s)")

    model_path = out_dir / "model"
    model.save(str(model_path))
    print(f"Saved model to {model_path}.zip")

    resolved_config_path = out_dir / "config.yaml"
    with open(resolved_config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    n_episodes = eval_episodes if eval_episodes is not None else int(config["evaluation"]["n_episodes"])
    seed_start = int(config["evaluation"]["seed"])
    print(f"\nEvaluating over {n_episodes} held-out episodes (seed_start={seed_start})...")
    summary, per_episode, trajectories = evaluate_model(
        model, config, n_episodes=n_episodes, seed_start=seed_start, record_trajectories=save_trajectories
    )
    for k, v in summary.items():
        print(f"  {k}: {v}")
    save_results(out_dir, summary, per_episode, trajectories)

    metadata = {
        "run_name": run_name,
        "seed": seed,
        "timesteps": timesteps,
        "config_path": str(Path(config_path).resolve()),
        "robot_model": "third_party/mujoco_menagerie/universal_robots_ur5e/ (official DeepMind MuJoCo Menagerie)",
        "obstacles_min_count": config["obstacles"]["min_count"],
        "obstacles_max_count": config["obstacles"]["max_count"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "training_wall_time_s": training_wall_time_s,
        "training_steps_per_s": timesteps / training_wall_time_s,
        "software_versions": _software_versions(),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    env.close()
    print(f"\nAll outputs saved to {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--eval-episodes", type=int, default=None, help="default: config's evaluation.n_episodes")
    parser.add_argument("--save-trajectories", action="store_true")
    args = parser.parse_args()

    train(
        config_path=args.config,
        seed=args.seed,
        timesteps=args.timesteps,
        run_name=args.run_name,
        eval_episodes=args.eval_episodes,
        save_trajectories=args.save_trajectories,
    )


if __name__ == "__main__":
    main()
