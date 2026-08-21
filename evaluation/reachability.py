"""Reachability placeholder via forward-kinematics workspace sampling.

Computes absolute reach and a reachable-volume proxy [MICARD 1.0.6].
Stored for inspection; not an optimization objective (correlated with
manipulability) [user].
"""
import numpy as np

from evaluation.base import register_metric
from evaluation.kinematics import forward_kinematics


@register_metric("reachability")
def reachability(design, n_samples=500, rng=None):
    if design.dof == 0:
        return {"abs_reach_m": 0.0, "workspace_volume_m3": 0.0}

    rng = rng or np.random.default_rng(1)
    pts = np.array([
        forward_kinematics(design, rng.uniform(-np.pi, np.pi, size=design.dof))
        for _ in range(n_samples)
    ])

    abs_reach = float(np.linalg.norm(pts, axis=1).max())
    # Simple axis-aligned bounding-box volume as a workspace proxy.
    span = pts.max(axis=0) - pts.min(axis=0)
    volume = float(np.prod(span))

    return {"abs_reach_m": abs_reach, "workspace_volume_m3": volume}