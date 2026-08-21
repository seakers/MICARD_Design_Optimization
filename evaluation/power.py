"""Power estimator STUB.

Accurate power draw depends on joint angles/velocity and requires a motion
simulation not yet available [Slack]. This placeholder keeps the metric slot
so it can be swapped in later without touching optimizers [MICARD 1.0.2].
"""
from evaluation.base import register_metric


@register_metric("power")
def estimate_power(design):
    """Placeholder: rough static proxy (sum of nominal torque as a stand-in).

    Replace with a simulation-driven estimate once available [Slack].
    """
    torque_proxy = sum((s.motor.get("nominal_torque_nm", 0.0) or 0.0)
                       for s in design.segments)
    return {"power_proxy": float(torque_proxy), "power_w": None}