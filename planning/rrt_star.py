from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from planning.collision_checker import CollisionChecker


@dataclass
class RRTStarNode:
    q: np.ndarray
    parent: int | None
    cost: float
    children: list[int] = field(default_factory=list)


@dataclass
class PlanResult:
    success: bool
    path: list[np.ndarray] | None
    cost: float | None
    n_iterations: int
    n_nodes: int


class RRTStar:
    def __init__(
        self,
        collision_checker: CollisionChecker,
        step_size: float = 0.2,
        goal_threshold: float = 0.1,
        max_iterations: int = 5000,
        goal_sample_rate: float = 0.1,
        collision_check_resolution: float = 0.05,
        rewire_radius: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self.cc = collision_checker
        self.step_size = step_size
        self.goal_threshold = goal_threshold
        self.max_iterations = max_iterations
        self.goal_sample_rate = goal_sample_rate
        self.collision_check_resolution = collision_check_resolution
        self.rewire_radius = rewire_radius
        self.rng = np.random.default_rng(seed)

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> PlanResult:
        q_start = np.asarray(q_start, dtype=np.float64)
        q_goal = np.asarray(q_goal, dtype=np.float64)

        if not self.cc.is_valid(q_start):
            raise ValueError("q_start is out of joint limits or in collision")
        if not self.cc.is_valid(q_goal):
            raise ValueError("q_goal is out of joint limits or in collision")

        nodes = [RRTStarNode(q=q_start, parent=None, cost=0.0)]
        goal_idx: int | None = None

        for it in range(self.max_iterations):
            q_rand = self._sample(q_goal)
            nearest_idx = self._nearest(nodes, q_rand)
            q_new = self._steer(nodes[nearest_idx].q, q_rand)
            q_new = np.clip(q_new, self.cc.joint_low, self.cc.joint_high)

            if not self.cc.is_edge_valid(nodes[nearest_idx].q, q_new, self.collision_check_resolution):
                continue

            near_indices = self._near(nodes, q_new)

            best_parent = nearest_idx
            best_cost = nodes[nearest_idx].cost + float(np.linalg.norm(q_new - nodes[nearest_idx].q))
            for idx in near_indices:
                candidate_cost = nodes[idx].cost + float(np.linalg.norm(q_new - nodes[idx].q))
                if candidate_cost < best_cost and self.cc.is_edge_valid(
                    nodes[idx].q, q_new, self.collision_check_resolution
                ):
                    best_parent = idx
                    best_cost = candidate_cost

            new_idx = len(nodes)
            nodes.append(RRTStarNode(q=q_new, parent=best_parent, cost=best_cost))
            nodes[best_parent].children.append(new_idx)

            for idx in near_indices:
                if idx == best_parent:
                    continue
                new_cost = best_cost + float(np.linalg.norm(nodes[idx].q - q_new))
                if new_cost < nodes[idx].cost and self.cc.is_edge_valid(
                    q_new, nodes[idx].q, self.collision_check_resolution
                ):
                    self._rewire(nodes, idx, new_parent=new_idx, new_cost=new_cost)

            if np.linalg.norm(q_new - q_goal) < self.goal_threshold and self.cc.is_edge_valid(
                q_new, q_goal, self.collision_check_resolution
            ):
                candidate_goal_cost = best_cost + float(np.linalg.norm(q_goal - q_new))
                if goal_idx is None:
                    goal_idx = len(nodes)
                    nodes.append(RRTStarNode(q=q_goal, parent=new_idx, cost=candidate_goal_cost))
                    nodes[new_idx].children.append(goal_idx)
                elif candidate_goal_cost < nodes[goal_idx].cost:
                    self._rewire(nodes, goal_idx, new_parent=new_idx, new_cost=candidate_goal_cost)

        if goal_idx is not None:
            path = self._reconstruct(nodes, goal_idx)
            return PlanResult(
                success=True, path=path, cost=nodes[goal_idx].cost, n_iterations=self.max_iterations, n_nodes=len(nodes)
            )
        return PlanResult(success=False, path=None, cost=None, n_iterations=self.max_iterations, n_nodes=len(nodes))

    def _sample(self, q_goal: np.ndarray) -> np.ndarray:
        if self.rng.random() < self.goal_sample_rate:
            return q_goal.copy()
        return self.rng.uniform(self.cc.joint_low, self.cc.joint_high)

    @staticmethod
    def _nearest(nodes: list[RRTStarNode], q: np.ndarray) -> int:
        dists = [np.linalg.norm(n.q - q) for n in nodes]
        return int(np.argmin(dists))

    def _near(self, nodes: list[RRTStarNode], q: np.ndarray) -> list[int]:
        return [i for i, n in enumerate(nodes) if np.linalg.norm(n.q - q) <= self.rewire_radius]

    def _steer(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        delta = q_to - q_from
        dist = np.linalg.norm(delta)
        if dist <= self.step_size:
            return q_to.copy()
        return q_from + delta / dist * self.step_size

    @staticmethod
    def _rewire(nodes: list[RRTStarNode], idx: int, new_parent: int, new_cost: float) -> None:
        old_parent = nodes[idx].parent
        if old_parent is not None:
            nodes[old_parent].children.remove(idx)
        nodes[idx].parent = new_parent
        nodes[new_parent].children.append(idx)

        delta = new_cost - nodes[idx].cost
        nodes[idx].cost = new_cost

        stack = list(nodes[idx].children)
        while stack:
            child_idx = stack.pop()
            nodes[child_idx].cost += delta
            stack.extend(nodes[child_idx].children)

    @staticmethod
    def _reconstruct(nodes: list[RRTStarNode], goal_idx: int) -> list[np.ndarray]:
        path = []
        idx: int | None = goal_idx
        while idx is not None:
            path.append(nodes[idx].q)
            idx = nodes[idx].parent
        path.reverse()
        return path
