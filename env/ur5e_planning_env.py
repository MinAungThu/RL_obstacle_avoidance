from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import yaml
from gymnasium import spaces

from env.collision import robot_in_collision
from env.scene_builder import COMPILED_HALF_EXTENT, build_obstacle_scene_xml, parked_pos
from planning.collision_checker import CollisionChecker
from planning.ik import IKSolver

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


class UR5ePlanningEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        render_mode: str | None = None,
    ) -> None:
        self.config = config if config is not None else load_config()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise NotImplementedError(
                f"render_mode={render_mode!r} not supported (only 'rgb_array' is "
                "implemented at this phase; interactive viewing comes later)."
            )
        self.render_mode = render_mode

        obstacles_cfg = self.config["obstacles"]
        self.obstacle_padding_capacity = int(obstacles_cfg["padding_capacity"])
        self.obstacle_min_count = int(obstacles_cfg["min_count"])
        self.obstacle_max_count = int(obstacles_cfg["max_count"])
        self._obstacle_half_extent_range = tuple(obstacles_cfg["half_extent_range"])
        self._obstacle_min_clearance_target = float(obstacles_cfg["min_clearance_target"])
        self._obstacle_min_clearance_start_ee = float(obstacles_cfg["min_clearance_start_ee"])
        self._obstacle_base_exclusion_radius = float(obstacles_cfg["base_exclusion_radius"])
        if not (0 <= self.obstacle_min_count <= self.obstacle_max_count <= self.obstacle_padding_capacity):
            raise ValueError(
                "obstacles config must satisfy 0 <= min_count <= max_count <= padding_capacity, "
                f"got min_count={self.obstacle_min_count}, max_count={self.obstacle_max_count}, "
                f"padding_capacity={self.obstacle_padding_capacity}"
            )
        if max(self._obstacle_half_extent_range) > min(COMPILED_HALF_EXTENT):
            raise ValueError(
                f"obstacles.half_extent_range={self._obstacle_half_extent_range} exceeds "
                f"env/scene_builder.py's COMPILED_HALF_EXTENT={COMPILED_HALF_EXTENT}. Obstacle "
                "geoms are compiled at a fixed max size and only ever shrunk at runtime (see "
                "scene_builder.py's module docstring for why); raise COMPILED_HALF_EXTENT if "
                "you need larger obstacles."
            )

        model_path = build_obstacle_scene_xml(self.obstacle_padding_capacity)
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        self._obstacle_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle_{i}")
            for i in range(self.obstacle_padding_capacity)
        ]
        self._obstacle_qpos_adr = [
            self.model.jnt_qposadr[self.model.body_jntadr[bid]] for bid in self._obstacle_body_ids
        ]
        self._obstacle_qvel_adr = [
            self.model.jnt_dofadr[self.model.body_jntadr[bid]] for bid in self._obstacle_body_ids
        ]
        self._obstacle_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle_{i}_geom")
            for i in range(self.obstacle_padding_capacity)
        ]
        self._obstacle_parked_pos = [
            np.array(parked_pos(i)) for i in range(self.obstacle_padding_capacity)
        ]
        self._obstacle_pin_pos = [p.copy() for p in self._obstacle_parked_pos]
        self._robot_body_ids = set()
        for b in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
            if name and name != "world" and not name.startswith("obstacle_"):
                self._robot_body_ids.add(b)

        self._home_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if self._home_key_id == -1:
            raise ValueError("Model has no 'home' keyframe.")
        self._home_qpos = self.model.key_qpos[self._home_key_id, : self.model.nu].copy()

        self._ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        if self._ee_site_id == -1:
            raise ValueError("Model has no 'attachment_site' end-effector site.")

        self._floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        env_cfg = self.config["environment"]
        self.frame_skip = int(env_cfg["frame_skip"])
        self.max_episode_steps = int(env_cfg["max_episode_steps"])
        self.success_threshold = float(env_cfg["success_threshold"])
        self.collision_terminates = bool(env_cfg["collision_terminates"])

        action_cfg = self.config["action"]
        self.max_joint_delta = float(action_cfg["max_joint_delta_rad"])

        ws = self.config["workspace"]
        self._ws_x = tuple(ws["x"])
        self._ws_y = tuple(ws["y"])
        self._ws_z = tuple(ws["z"])
        self._min_reach = float(ws["min_reach"])
        self._max_reach = float(ws["max_reach"])

        start_cfg = self.config["start_config"]
        self.randomize_start = bool(start_cfg["randomize"])
        self.start_noise_rad = float(start_cfg["noise_rad"])

        reward_cfg = self.config["reward"]
        self.w_distance = float(reward_cfg["distance_weight"])
        self.w_action = float(reward_cfg["action_penalty_weight"])
        self.collision_penalty = float(reward_cfg["collision_penalty"])
        self.success_reward = float(reward_cfg["success_reward"])

        n_joints = self.model.nu
        self.n_robot_dof = n_joints
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_joints,), dtype=np.float32)
        obs_dim = n_joints + n_joints + 3 + 3 + 3 + self.obstacle_padding_capacity * 7
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._joint_low = self.model.actuator_ctrlrange[:, 0].copy()
        self._joint_high = self.model.actuator_ctrlrange[:, 1].copy()

        self._ctrl_setpoint = self._home_qpos.copy()
        self._target_pos = np.zeros(3, dtype=np.float64)
        self._active_obstacles: list[tuple[np.ndarray, np.ndarray]] = []
        self._step_count = 0
        self._renderer: mujoco.Renderer | None = None
        self._max_reset_attempts = 20

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        for _attempt in range(self._max_reset_attempts):
            if self.randomize_start:
                noise = self.np_random.uniform(-self.start_noise_rad, self.start_noise_rad, size=6)
                qpos = np.clip(self._home_qpos + noise, self._joint_low, self._joint_high)
            else:
                qpos = self._home_qpos.copy()
            self.data.qpos[: self.n_robot_dof] = qpos
            self.data.qvel[: self.n_robot_dof] = 0.0
            self._ctrl_setpoint = qpos.copy()
            self.data.ctrl[:] = qpos
            mujoco.mj_forward(self.model, self.data)

            self._target_pos = self._sample_target()
            start_ee_pos = self._get_ee_pos()
            self._active_obstacles = self._sample_obstacles(start_ee_pos)
            self._apply_obstacles(self._active_obstacles)
            mujoco.mj_forward(self.model, self.data)

            if not self._check_collision():
                break

        self._step_count = 0

        obs = self._get_obs()
        info = self._get_info(reward_components=None, success=False, collision=False)
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        self._ctrl_setpoint = np.clip(
            self._ctrl_setpoint + action * self.max_joint_delta,
            self._joint_low,
            self._joint_high,
        )
        self.data.ctrl[:] = self._ctrl_setpoint

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            self._pin_obstacles()

        self._step_count += 1

        finite_state = bool(np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel)))
        collision = self._check_collision() if finite_state else True

        ee_pos = self._get_ee_pos()
        distance = float(np.linalg.norm(self._target_pos - ee_pos)) if finite_state else float("inf")
        success = finite_state and (distance < self.success_threshold)

        reward, reward_components = self._compute_reward(distance, action, collision, success, finite_state)

        terminated = success or (not finite_state) or (collision and self.collision_terminates)
        truncated = self._step_count >= self.max_episode_steps and not terminated

        obs = self._get_obs() if finite_state else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = self._get_info(reward_components, success, collision, distance=distance)
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _get_ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._ee_site_id].copy()

    def _get_obs(self) -> np.ndarray:
        ee_pos = self._get_ee_pos()
        obs = np.concatenate(
            [
                self.data.qpos[: self.n_robot_dof].copy(),
                self.data.qvel[: self.n_robot_dof].copy(),
                ee_pos,
                self._target_pos,
                self._target_pos - ee_pos,
                self._get_obstacle_obs(),
            ]
        ).astype(np.float32)
        return obs

    def _get_obstacle_obs(self) -> np.ndarray:
        slots = np.zeros((self.obstacle_padding_capacity, 7), dtype=np.float32)
        for i, (center, half_extent) in enumerate(self._active_obstacles):
            slots[i, 0:3] = center
            slots[i, 3:6] = half_extent
            slots[i, 6] = 1.0
        return slots.flatten()

    def _get_info(
        self,
        reward_components: dict[str, float] | None,
        success: bool,
        collision: bool,
        distance: float | None = None,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            "success": success,
            "collision": collision,
            "target_pos": self._target_pos.copy(),
            "ee_pos": self._get_ee_pos(),
            "n_obstacles": len(self._active_obstacles),
        }
        if distance is not None:
            info["distance"] = distance
        if reward_components is not None:
            info["reward_components"] = reward_components
        return info

    def _sample_target(self) -> np.ndarray:
        for _ in range(1000):
            candidate = np.array(
                [
                    self.np_random.uniform(*self._ws_x),
                    self.np_random.uniform(*self._ws_y),
                    self.np_random.uniform(*self._ws_z),
                ]
            )
            dist = np.linalg.norm(candidate)
            if self._min_reach <= dist <= self._max_reach:
                return candidate
        raise RuntimeError(
            "Failed to sample a reachable target after 1000 attempts; "
            "check configs/default.yaml workspace bounds against min_reach/max_reach."
        )

    def _sample_obstacles(self, start_ee_pos: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        n_requested = int(self.np_random.integers(self.obstacle_min_count, self.obstacle_max_count + 1))
        obstacles: list[tuple[np.ndarray, np.ndarray]] = []
        for _ in range(n_requested):
            for _attempt in range(200):
                half_extent = self.np_random.uniform(*self._obstacle_half_extent_range, size=3)
                center = np.array(
                    [
                        self.np_random.uniform(*self._ws_x),
                        self.np_random.uniform(*self._ws_y),
                        self.np_random.uniform(*self._ws_z),
                    ]
                )
                center[2] = max(center[2], half_extent[2] + 0.01)
                radius = float(np.max(half_extent))

                if np.linalg.norm(center - self._target_pos) - radius < self._obstacle_min_clearance_target:
                    continue
                if np.linalg.norm(center - start_ee_pos) - radius < self._obstacle_min_clearance_start_ee:
                    continue
                if np.linalg.norm(center[:2]) - radius < self._obstacle_base_exclusion_radius:
                    continue
                if any(
                    np.linalg.norm(center - c) < (radius + float(np.max(h)) + self._obstacle_min_clearance_target)
                    for c, h in obstacles
                ):
                    continue

                obstacles.append((center, half_extent))
                break
        return obstacles

    def _apply_obstacles(self, obstacles: list[tuple[np.ndarray, np.ndarray]]) -> None:
        for i in range(self.obstacle_padding_capacity):
            geom_id = self._obstacle_geom_ids[i]
            if i < len(obstacles):
                center, half_extent = obstacles[i]
                self._obstacle_pin_pos[i] = center
                self.model.geom_size[geom_id] = half_extent
            else:
                self._obstacle_pin_pos[i] = self._obstacle_parked_pos[i].copy()
        self._pin_obstacles()

    def _pin_obstacles(self) -> None:
        for i in range(self.obstacle_padding_capacity):
            qpos_adr = self._obstacle_qpos_adr[i]
            qvel_adr = self._obstacle_qvel_adr[i]
            self.data.qpos[qpos_adr : qpos_adr + 3] = self._obstacle_pin_pos[i]
            self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
            self.data.qvel[qvel_adr : qvel_adr + 6] = 0.0

    def _check_collision(self) -> bool:
        return robot_in_collision(self.model, self.data, self._robot_body_ids)

    def make_collision_checker(self) -> CollisionChecker:
        data = mujoco.MjData(self.model)
        for i in range(self.obstacle_padding_capacity):
            qpos_adr = self._obstacle_qpos_adr[i]
            data.qpos[qpos_adr : qpos_adr + 3] = self._obstacle_pin_pos[i]
            data.qpos[qpos_adr + 3 : qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
        return CollisionChecker(
            model=self.model,
            data=data,
            n_robot_dof=self.n_robot_dof,
            joint_low=self._joint_low,
            joint_high=self._joint_high,
            robot_body_ids=self._robot_body_ids,
        )

    def make_ik_solver(self) -> IKSolver:
        return IKSolver(
            model=self.model,
            ee_site_id=self._ee_site_id,
            n_robot_dof=self.n_robot_dof,
            joint_low=self._joint_low,
            joint_high=self._joint_high,
        )

    def _compute_reward(
        self,
        distance: float,
        action: np.ndarray,
        collision: bool,
        success: bool,
        finite_state: bool,
    ) -> tuple[float, dict[str, float]]:
        if not finite_state:
            components = {
                "distance": 0.0,
                "action_penalty": 0.0,
                "collision_penalty": -self.collision_penalty,
                "success_reward": 0.0,
            }
            return -self.collision_penalty, components

        r_distance = -self.w_distance * distance
        r_action = -self.w_action * float(np.sum(action**2))
        r_collision = -self.collision_penalty if collision else 0.0
        r_success = self.success_reward if success else 0.0

        components = {
            "distance": r_distance,
            "action_penalty": r_action,
            "collision_penalty": r_collision,
            "success_reward": r_success,
        }
        total = r_distance + r_action + r_collision + r_success
        return total, components
