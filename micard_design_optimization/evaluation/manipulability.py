"""Yoshikawa manipulability index [MICARD 1.0.5]."""
import numpy as np

from micard_design_optimization.evaluation.base import register_metric
from micard_design_optimization.evaluation.kinematics import jacobian


def yoshikawa(J):
    """sqrt(det(J J^T)) for a single Jacobian (translational part by default)."""
    Jp = J[:3, :]
    JJt = Jp @ Jp.T
    det = np.linalg.det(JJt)
    return float(np.sqrt(max(det, 0.0)))


@register_metric("manipulability")
def mean_manipulability(design, n_samples=200, rng=None):
    """Average Yoshikawa index over random joint configurations.

    A single scalar dexterity measure used as an optimization objective
    (maximize) [MICARD 1.0.5]. Returns extra stats for later inspection.
    """
    if design.dof == 0:
        return {"manipulability": 0.0, "manipulability_max": 0.0}

    rng = rng or np.random.default_rng(0)  # fixed seed => reproducible metric
    vals = []
    for _ in range(n_samples):
        angles = rng.uniform(-np.pi, np.pi, size=design.dof)
        vals.append(yoshikawa(jacobian(design, angles)))
    vals = np.array(vals)
    return {
        "manipulability": float(vals.mean()),
        "manipulability_max": float(vals.max()),
    }