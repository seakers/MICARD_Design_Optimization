"""Custom Pareto-front and hypervolume utilities (no pymoo dependency).

Drop-in replacement for the is_pareto_efficient + HV(ref_point) pattern used
throughout the reference optimizers [genetic_algorithm 3][random_search 5].
Inputs to `hypervolume` are expected to be normalized to minimization against
a unit reference point, matching the existing convention [design_repair 2].
"""
from copy import deepcopy
import numpy as np


def is_pareto_efficient(points, return_mask=True):
    """Find Pareto-efficient points (minimization). Matches reference usage.

    Args:
        points: (n, m) array of objective vectors (lower is better).
        return_mask: if True, return a boolean mask; else return indices.
    """
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    is_efficient = np.arange(n)
    next_point_index = 0
    pts = points.copy()

    while next_point_index < len(pts):
        nondominated = np.any(pts < pts[next_point_index], axis=1)
        nondominated[next_point_index] = True
        is_efficient = is_efficient[nondominated]
        pts = pts[nondominated]
        next_point_index = np.sum(nondominated[:next_point_index]) + 1

    if return_mask:
        mask = np.zeros(n, dtype=bool)
        mask[is_efficient] = True
        return mask
    return is_efficient


def _hv_recursive(points, ref_point):
    """Exact hypervolume for minimization via dimension sweep.

    Fast and exact for the 2-3 objectives used here. Generalizes to higher
    dimensions if ever needed, at increasing cost.
    """
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return 0.0

    # Keep only points that dominate the reference (strictly below it).
    points = points[np.all(points < ref_point, axis=1)]
    if points.size == 0:
        return 0.0

    m = points.shape[1]

    if m == 1:
        return float(ref_point[0] - np.min(points[:, 0]))

    if m == 2:
        # Sort by first objective ascending, sweep the second.
        order = np.argsort(points[:, 0])
        pts = points[order]
        volume = 0.0
        prev_x = ref_point[0]
        best_y = ref_point[1]
        # Walk from best (lowest) x outward.
        for x, y in pts:
            if y < best_y:
                volume += (prev_x - x) * (ref_point[1] - y)
                prev_x = x
                best_y = y
        return float(volume)

    # General case (m >= 3): recurse by slicing along the first axis.
    order = np.argsort(points[:, 0])
    pts = points[order]
    volume = 0.0
    prev_x = ref_point[0]
    for i, x in enumerate(pts[:, 0]):
        # Slab thickness along axis 0 from this point out to the previous slice.
        slab = prev_x - x
        if slab > 0:
            # Project remaining points (those with x <= current) onto lower dims.
            slice_pts = pts[: i + 1, 1:]
            base_area = _hv_recursive(slice_pts, ref_point[1:])
            volume += slab * base_area
        prev_x = x
    return float(volume)


def hypervolume(front_points, ref_point):
    """Exact hypervolume of a Pareto front (minimization, unit ref by default).

    Args:
        front_points: (k, m) array already normalized to minimization.
        ref_point: (m,) reference point (typically np.ones(m)) [genetic_algorithm 3].
    """
    front_points = np.asarray(front_points, dtype=float)
    if front_points.ndim == 1:
        front_points = front_points.reshape(1, -1)
    if front_points.size == 0:
        return 0.0
    return _hv_recursive(front_points, np.asarray(ref_point, dtype=float))


def normalize_objectives(all_obj, objective_min_max, max_values=None):
    """Normalize + flip-to-minimization, matching the reference convention.

    Divides by max_values and applies (1 - norm) for 'max' objectives so the
    result is minimization against a unit reference point [random_search 5].

    Args:
        all_obj: (n, m) raw objective array.
        objective_min_max: list of 'min'/'max' per objective.
        max_values: optional (m,) normalizer. If None, derived from the data
                    as max * 1.1 + 1e-6 (the random-search fallback [5]).

    Returns:
        (hv_obj, max_values) where hv_obj is minimization-normalized.
    """
    all_obj = np.asarray(all_obj, dtype=float)
    if max_values is None:
        max_values = np.max(all_obj, axis=0) * 1.1 + 1e-6  # avoid div-by-zero [5]

    norm_obj = all_obj / max_values
    hv_obj = deepcopy(norm_obj)
    for i, mm in enumerate(objective_min_max):
        if mm.lower() == "max":
            hv_obj[:, i] = 1.0 - norm_obj[:, i]  # flip maximization [3][5]
    return hv_obj, max_values


def pareto_progress(all_obj, all_constraints, objective_min_max,
                    max_values=None, ref_point=None):
    """Compute Pareto front + hypervolume history over evaluations (NFE).

    Mirrors calculate_pareto_progress / the post-processing loop in the
    reference optimizers, including infeasible-design handling and carrying
    the last hypervolume forward when no improvement occurs [design_repair 2][3][5].
    """
    all_obj = np.asarray(all_obj, dtype=float).copy()
    all_constraints = np.asarray(all_constraints, dtype=bool)
    num_objectives = all_obj.shape[1]
    min_mask = np.array([mm.lower() == "min" for mm in objective_min_max])

    if ref_point is None:
        ref_point = np.ones(num_objectives)

    # Map infeasible designs to the reference point (max for min-objs, 0 for max-objs) [3][5].
    hv_obj_tmp, max_values = normalize_objectives(
        np.where(all_constraints[:, None], 0.0, all_obj), objective_min_max, max_values
    )
    ref_values = max_values * ref_point
    ref_values[~min_mask[:num_objectives]] = 0.0
    all_obj[all_constraints] = ref_values

    hv_obj, _ = normalize_objectives(all_obj, objective_min_max, max_values)

    hypervolumes = []
    pareto_front_obj = []
    for i in range(len(all_obj)):
        if (not all_constraints[i]) or len(hypervolumes) == 0:
            sub = hv_obj[: i + 1]
            mask = is_pareto_efficient(sub, return_mask=True)
            if mask[-1]:
                pareto_front_obj.append(all_obj[: i + 1][mask])
                hypervolumes.append(hypervolume(sub[mask], ref_point))
            else:
                hypervolumes.append(hypervolumes[-1] if hypervolumes else 0.0)
                pareto_front_obj.append(pareto_front_obj[-1] if pareto_front_obj else [])
        else:
            hypervolumes.append(hypervolumes[-1] if hypervolumes else 0.0)
            pareto_front_obj.append(pareto_front_obj[-1] if pareto_front_obj else [])

    return pareto_front_obj, hypervolumes