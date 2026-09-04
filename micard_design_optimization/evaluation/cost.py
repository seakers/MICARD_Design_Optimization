"""Total cost from the catalog (optimization objective: minimize) [MICARD 1.0.1]."""
from micard_design_optimization.evaluation.base import register_metric


@register_metric("cost")
def total_cost(design):
    """Sum of motor costs. Link/material cost can be added here later [Slack]."""
    cost = 0.0
    for seg in design.segments:
        cost += seg.motor.get("cost_usd", 0.0) or 0.0
    return {"cost": float(cost)}