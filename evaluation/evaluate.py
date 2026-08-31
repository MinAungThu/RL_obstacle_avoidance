from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco
import numpy as np
from stable_baselines3 import SAC

from env.ur5e_planning_env import UR5ePlanningEnv, load_config
from planning.path_utils import path_length, smoothness


def _collision_type(env: UR5ePlanningEnv) -> str:
    has_obstacle = False
    has_ground_or_self = False
    for i in range(env.data.ncon):
        contact = env.data.contact[i]
        body1 = env.model.geom_bodyid[contact.geom1]
        body2 = env.model.geom_bodyid[contact.geom2]
        if body1 not in env._robot_body_ids and body2 not in env._robot_body_ids:
            continue
        for g in (contact.geom1, contact.geom2):
            name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, g)
            if name and name.startswith("obstacle_"):
                has_obstacle = True
            elif name:
                has_ground_or_self = True
    if has_obstacle:
        return "obstacle"
    if has_ground_or_self:
        return "ground_or_self"
    return "none"


def evaluate_model(
    model: SAC,
    config: dict[str, Any],
    n_episodes: int,
    seed_start: int,
    record_trajectories: bool = False,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]] | None]:
    env = UR5ePlanningEnv(config=config)

    per_episode: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] | None = [] if record_trajectories else None

    for i in range(n_episodes):
        seed = seed_start + i
        obs, info = env.reset(seed=seed)
        terminated = truncated = False
        ep_return = 0.0
        ep_len = 0
        joint_traj = [env.data.qpos[: env.n_robot_dof].copy()]
        ee_traj = [info["ee_pos"].copy()]
        actions = []
        rewards = []
        collision_flags = []
        inference_time = 0.0

        while not (terminated or truncated):
            t0 = time.perf_counter()
            action, _ = model.predict(obs, deterministic=True)
            inference_time += time.perf_counter() - t0

            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            ep_len += 1
            joint_traj.append(env.data.qpos[: env.n_robot_dof].copy())
            ee_traj.append(info["ee_pos"].copy())
            actions.append(np.asarray(action).copy())
            rewards.append(reward)
            collision_flags.append(bool(info["collision"]))

        joint_path_len = path_length(joint_traj)
        ee_path_len = path_length(ee_traj)
        joint_smoothness = smoothness(joint_traj)
        collision_type = _collision_type(env) if info["collision"] else "none"

        per_episode.append(
            {
                "seed": seed,
                "success": bool(info["success"]),
                "collision": bool(info["collision"]),
                "collision_type": collision_type,
                "return": float(ep_return),
                "episode_length": ep_len,
                "final_distance": float(info["distance"]),
                "joint_path_length": joint_path_len,
                "ee_path_length": ee_path_len,
                "joint_smoothness": joint_smoothness,
                "inference_time_s": inference_time,
                "mean_inference_time_per_step_s": inference_time / max(ep_len, 1),
            }
        )

        if trajectories is not None:
            trajectories.append(
                {
                    "seed": seed,
                    "joint_trajectory": np.array(joint_traj),
                    "ee_trajectory": np.array(ee_traj),
                    "actions": np.array(actions),
                    "rewards": np.array(rewards),
                    "collisions": np.array(collision_flags),
                    "success": bool(info["success"]),
                    "target_pos": info["target_pos"].copy(),
                }
            )

    env.close()

    def _mean(key: str) -> float:
        return float(np.mean([row[key] for row in per_episode]))

    def _std(key: str) -> float:
        return float(np.std([row[key] for row in per_episode]))

    n_success = sum(1 for row in per_episode if row["success"])
    n_collision = sum(1 for row in per_episode if row["collision"])
    n_obstacle_collision = sum(1 for row in per_episode if row["collision_type"] == "obstacle")
    n_ground_or_self_collision = sum(1 for row in per_episode if row["collision_type"] == "ground_or_self")

    summary = {
        "n_episodes": n_episodes,
        "success_rate": n_success / n_episodes,
        "mean_return": _mean("return"),
        "std_return": _std("return"),
        "mean_episode_length": _mean("episode_length"),
        "mean_final_distance": _mean("final_distance"),
        "collision_rate": n_collision / n_episodes,
        "collision_rate_obstacle": n_obstacle_collision / n_episodes,
        "collision_rate_ground_or_self": n_ground_or_self_collision / n_episodes,
        "mean_joint_path_length": _mean("joint_path_length"),
        "mean_ee_path_length": _mean("ee_path_length"),
        "mean_joint_smoothness": _mean("joint_smoothness"),
        "mean_inference_time_per_episode_s": _mean("inference_time_s"),
        "mean_inference_time_per_step_s": _mean("mean_inference_time_per_step_s"),
    }
    return summary, per_episode, trajectories


def save_results(
    out_dir: Path,
    summary: dict[str, Any],
    per_episode: list[dict[str, Any]],
    trajectories: list[dict[str, Any]] | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if per_episode:
        with open(out_dir / "eval_episodes.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_episode[0].keys()))
            writer.writeheader()
            writer.writerows(per_episode)

    if trajectories:
        traj_dir = out_dir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        for traj in trajectories:
            np.savez_compressed(
                traj_dir / f"episode_seed{traj['seed']}.npz",
                joint_trajectory=traj["joint_trajectory"],
                ee_trajectory=traj["ee_trajectory"],
                actions=traj["actions"],
                rewards=traj["rewards"],
                collisions=traj["collisions"],
                success=traj["success"],
                target_pos=traj["target_pos"],
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="path to a saved SAC model .zip")
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--episodes", type=int, default=None, help="default: config's evaluation.n_episodes")
    parser.add_argument("--seed-start", type=int, default=None, help="default: config's evaluation.seed")
    parser.add_argument("--save-trajectories", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None, help="default: alongside --model")
    args = parser.parse_args()

    config = load_config(args.config)
    n_episodes = args.episodes if args.episodes is not None else int(config["evaluation"]["n_episodes"])
    seed_start = args.seed_start if args.seed_start is not None else int(config["evaluation"]["seed"])

    model = SAC.load(args.model)
    summary, per_episode, trajectories = evaluate_model(
        model, config, n_episodes=n_episodes, seed_start=seed_start, record_trajectories=args.save_trajectories
    )

    print(f"Evaluation over {n_episodes} episodes (measured, not fabricated):")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.model).resolve().parent
    save_results(out_dir, summary, per_episode, trajectories)
    print(f"\nSaved eval_summary.json / eval_episodes.csv to {out_dir}")


if __name__ == "__main__":
    main()
