from __future__ import annotations

import mujoco
import numpy as np


class IKSolver:
    def __init__(
        self,
        model: mujoco.MjModel,
        ee_site_id: int,
        n_robot_dof: int,
        joint_low: np.ndarray,
        joint_high: np.ndarray,
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.ee_site_id = ee_site_id
        self.n_robot_dof = n_robot_dof
        self.joint_low = np.asarray(joint_low, dtype=np.float64)
        self.joint_high = np.asarray(joint_high, dtype=np.float64)
        self._jacp = np.zeros((3, model.nv))

    def solve(
        self,
        target_pos: np.ndarray,
        q_init: np.ndarray,
        max_iterations: int = 200,
        tol: float = 0.01,
        damping: float = 0.05,
    ) -> tuple[np.ndarray, bool]:
        q = np.clip(np.asarray(q_init, dtype=np.float64), self.joint_low, self.joint_high)

        for _ in range(max_iterations):
            ee_pos, err = self._forward_and_error(q, target_pos)
            if np.linalg.norm(err) < tol:
                return q, True

            mujoco.mj_jacSite(self.model, self.data, self._jacp, None, self.ee_site_id)
            J = self._jacp[:, : self.n_robot_dof]
            JJt = J @ J.T + (damping**2) * np.eye(3)
            dq = J.T @ np.linalg.solve(JJt, err)
            q = np.clip(q + dq, self.joint_low, self.joint_high)

        _, err = self._forward_and_error(q, target_pos)
        return q, bool(np.linalg.norm(err) < tol)

    def solve_with_restarts(
        self,
        target_pos: np.ndarray,
        q_init: np.ndarray,
        n_restarts: int,
        rng: np.random.Generator,
        max_iterations: int = 200,
        tol: float = 0.01,
        damping: float = 0.05,
    ) -> tuple[np.ndarray, bool]:
        q, converged = self.solve(target_pos, q_init, max_iterations, tol, damping)
        if converged:
            return q, True
        for _ in range(n_restarts):
            seed = rng.uniform(self.joint_low, self.joint_high)
            q, converged = self.solve(target_pos, seed, max_iterations, tol, damping)
            if converged:
                return q, True
        return q, False

    def _forward_and_error(self, q: np.ndarray, target_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[: self.n_robot_dof] = q
        mujoco.mj_forward(self.model, self.data)
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        return ee_pos, target_pos - ee_pos
