from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from planning.collision_checker import CollisionChecker


@dataclass
class RRTNode:
    q: np.ndarray
    parent: int | None


@dataclass
class PlanResult:
    success: bool
    path: list[np.ndarray] | None
    n_iterations: int
    n_nodes: int


class RRT:
    def __init__(
        self,
        collision_checker: CollisionChecker,
        step_size: float = 0.2,
        goal_threshold: float = 0.1,
        max_iterations: int = 5000,
        goal_sample_rate: float = 0.1,
        collision_check_resolution: float = 0.05,
        seed: int | None = None,
    ) -> None:
        self.cc = collision_checker
        self.step_size = step_size
        self.goal_threshold = goal_threshold
        self.max_iterations = max_iterations
        self.goal_sample_rate = goal_sample_rate
        self.collision_check_resolution = collision_check_resolution
        self.rng = np.random.default_rng(seed)

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> PlanResult:
        q_start = np.asarray(q_start, dtype=np.float64)
        q_goal = np.asarray(q_goal, dtype=np.float64)

        if not self.cc.is_valid(q_start):
            raise ValueError("q_start is out of joint limits or in collision")
        if not self.cc.is_valid(q_goal):
            raise ValueError("q_goal is out of joint limits or in collision")

        nodes = [RRTNode(q=q_start, parent=None)]

        for it in range(self.max_iterations):
            q_rand = self._sample(q_goal)
            nearest_idx = self._nearest(nodes, q_rand)
            q_near = nodes[nearest_idx].q
            q_new = self._steer(q_near, q_rand)
            q_new = np.clip(q_new, self.cc.joint_low, self.cc.joint_high)

            if not self.cc.is_edge_valid(q_near, q_new, self.collision_check_resolution):
                continue

            new_idx = len(nodes)
            nodes.append(RRTNode(q=q_new, parent=nearest_idx))

            if np.linalg.norm(q_new - q_goal) < self.goal_threshold and self.cc.is_edge_valid(
                q_new, q_goal, self.collision_check_resolution
            ):
                goal_idx = len(nodes)
                nodes.append(RRTNode(q=q_goal, parent=new_idx))
                path = self._reconstruct(nodes, goal_idx)
                return PlanResult(success=True, path=path, n_iterations=it + 1, n_nodes=len(nodes))

        return PlanResult(success=False, path=None, n_iterations=self.max_iterations, n_nodes=len(nodes))

    def _sample(self, q_goal: np.ndarray) -> np.ndarray:
        if self.rng.random() < self.goal_sample_rate:
            return q_goal.copy()
        return self.rng.uniform(self.cc.joint_low, self.cc.joint_high)

    @staticmethod
    def _nearest(nodes: list[RRTNode], q: np.ndarray) -> int:
        dists = [np.linalg.norm(n.q - q) for n in nodes]
        return int(np.argmin(dists))

    def _steer(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        delta = q_to - q_from
        dist = np.linalg.norm(delta)
        if dist <= self.step_size:
            return q_to.copy()
        return q_from + delta / dist * self.step_size

    @staticmethod
    def _reconstruct(nodes: list[RRTNode], goal_idx: int) -> list[np.ndarray]:
        path = []
        idx: int | None = goal_idx
        while idx is not None:
            path.append(nodes[idx].q)
            idx = nodes[idx].parent
        path.reverse()
        return path
