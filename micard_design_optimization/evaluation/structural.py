"""Link structural model: circular-tube deflection + yield checks.

General beam physics with a small material table (defaults to aluminum)
[Slack][utils.materials]. Placeholder for the required structural (FEM)
engineering analysis [MICARD 4.0.4].
"""
import numpy as np

from micard_design_optimization.evaluation.base import register_metric
from micard_design_optimization.utils.materials import get_material

GRAVITY = 9.81  # used only for a conservative self-weight + payload estimate


def tube_area_moment(outer_d, wall):
    """Second moment of area for a circular tube (m^4)."""
    ro = outer_d / 2.0
    ri = ro - wall
    return (np.pi / 4.0) * (ro**4 - ri**4)


def tube_area(outer_d, wall):
    ro = outer_d / 2.0
    ri = ro - wall
    return np.pi * (ro**2 - ri**2)


def link_mass(seg):
    mat = get_material(seg.material)
    area = tube_area(seg.tube_outer_d_m, seg.tube_wall_m)
    return area * seg.link_length_m * mat.density_kg_m3


@register_metric("structural")
def structural(design, tip_payload_kg=1.0,
               max_deflection_ratio=0.01, safety_factor=2.0):
    """Evaluate each link as a cantilever under tip load; report worst margins.

    - deflection: delta = F L^3 / (3 E I); flagged if delta/L exceeds limit.
    - yield: bending stress = M c / I; flagged if it exceeds yield/SF.
    Returns feasibility plus the total structural mass [MICARD 1.0.3].
    """
    if design.dof == 0:
        return {"struct_feasible": False, "struct_mass_kg": 0.0,
                "max_deflection_ratio": 0.0, "min_yield_margin": 0.0}

    # Load carried by a link = payload + weight of everything distal to it.
    distal_motor_mass = 0.0
    for seg in reversed(design.segments):
        distal_motor_mass += (seg.motor.get("weight_g", 0.0) or 0.0) / 1000.0

    total_mass = 0.0
    worst_defl_ratio = 0.0
    min_yield_margin = np.inf
    feasible = True

    carried = tip_payload_kg
    # Walk distal -> proximal so 'carried' accumulates outward mass.
    for seg in reversed(design.segments):
        mat = get_material(seg.material)
        I = tube_area_moment(seg.tube_outer_d_m, seg.tube_wall_m)
        L = seg.link_length_m
        c = seg.tube_outer_d_m / 2.0
        m_link = link_mass(seg)
        total_mass += m_link

        F = (carried + m_link) * GRAVITY   # conservative tip-equivalent load
        deflection = F * L**3 / (3.0 * mat.elastic_modulus_pa * I)
        defl_ratio = deflection / L if L > 0 else 0.0
        worst_defl_ratio = max(worst_defl_ratio, defl_ratio)

        moment = F * L
        bending_stress = moment * c / I if I > 0 else np.inf
        allowable = mat.yield_strength_pa / safety_factor
        yield_margin = (allowable - bending_stress) / allowable
        min_yield_margin = min(min_yield_margin, yield_margin)

        if defl_ratio > max_deflection_ratio or bending_stress > allowable:
            feasible = False

        carried += m_link

    return {
        "struct_feasible": bool(feasible),
        "struct_mass_kg": float(total_mass),
        "max_deflection_ratio": float(worst_defl_ratio),
        "min_yield_margin": float(min_yield_margin),
    }