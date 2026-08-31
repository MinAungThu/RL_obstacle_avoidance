from __future__ import annotations

import mujoco
import numpy as np

from env.collision import robot_in_collision


class CollisionChecker:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        n_robot_dof: int,
        joint_low: np.ndarray,
        joint_high: np.ndarray,
        robot_body_ids: set[int],
    ) -> None:
        self.model = model
        self.data = data
        self.n_robot_dof = n_robot_dof
        self.joint_low = np.asarray(joint_low, dtype=np.float64)
        self.joint_high = np.asarray(joint_high, dtype=np.float64)
        self.robot_body_ids = robot_body_ids

    def in_joint_limits(self, q: np.ndarray) -> bool:
        return bool(np.all(q >= self.joint_low) and np.all(q <= self.joint_high))

    def is_valid(self, q: np.ndarray) -> bool:
        if not self.in_joint_limits(q):
            return False
        self.data.qpos[: self.n_robot_dof] = q
        self.data.qvel[: self.n_robot_dof] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return not robot_in_collision(self.model, self.data, self.robot_body_ids)

    def is_edge_valid(self, q_start: np.ndarray, q_end: np.ndarray, resolution: float) -> bool:
        max_delta = float(np.max(np.abs(q_end - q_start)))
        n_steps = max(1, int(np.ceil(max_delta / resolution)))
        for i in range(n_steps + 1):
            alpha = i / n_steps
            q = (1.0 - alpha) * q_start + alpha * q_end
            if not self.is_valid(q):
                return False
        return True
