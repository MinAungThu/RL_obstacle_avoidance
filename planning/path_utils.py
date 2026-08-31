from __future__ import annotations

import numpy as np


def path_length(path: list[np.ndarray]) -> float:
    return float(sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1)))


def smoothness(path: list[np.ndarray]) -> float:
    if len(path) < 3:
        return 0.0
    total = 0.0
    for t in range(1, len(path) - 1):
        d = path[t + 1] - 2.0 * path[t] + path[t - 1]
        total += float(np.dot(d, d))
    return total


def resample_path(path: list[np.ndarray], step_size: float) -> list[np.ndarray]:
    if len(path) < 2:
        return list(path)
    resampled = [path[0]]
    for i in range(len(path) - 1):
        segment = path[i + 1] - path[i]
        dist = float(np.linalg.norm(segment))
        n_steps = max(1, int(np.ceil(dist / step_size)))
        for j in range(1, n_steps + 1):
            resampled.append(path[i] + segment * (j / n_steps))
    return resampled
