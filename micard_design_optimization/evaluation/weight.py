"""Total weight: motors (catalog) + link structural mass [MICARD 1.0.3]."""
from micard_design_optimization.evaluation.base import register_metric
from micard_design_optimization.evaluation.structural import link_mass


@register_metric("weight")
def total_weight(design):
    motor_kg = sum((s.motor.get("weight_g", 0.0) or 0.0) / 1000.0
                   for s in design.segments)
    link_kg = sum(link_mass(s) for s in design.segments)
    return {"weight": float(motor_kg + link_kg),
            "motor_mass_kg": float(motor_kg),
            "link_mass_kg": float(link_kg)}