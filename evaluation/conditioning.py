"""Conditioning metrics backed by the micard-metrics service.

Registers trac_ik-backed metrics (w, k_inv, k_min, k, alpha, beta, in_collision)
into the evaluator registry so compute_all_metrics picks them up transparently.
'w' is sqrt(det(J Jᵀ)) — the Yoshikawa index [micard-metrics README][MICARD 1.0.5]
— and is exposed as 'manipulability' so it can drive the objective in place of
the placeholder. Degrades gracefully: if the service is unreachable, returns {}
and the placeholder manipulability.py remains authoritative [MICARD 5.0].
"""
import numpy as np
from xml.etree import ElementTree as ET

from evaluation.base import register_metric
from outputs.urdf_export import design_to_urdf, design_to_srdf
from utils.metrics_client import MetricsClient

# --- Easily-changed configuration (single place to edit when talking to team) ---
CONDITIONING_CONFIG = {
    "socket_path": "run/api.sock",
    "objective_metric": "w",     # <-- change to "k_inv" here if the team prefers
    "task_dir": [1.0, 0.0, 0.0], # toward the connection port; adjust as needed
    "n_samples": 200,
    "collision": True,
    "enabled": True,
    "seed": 0,
}

_CLIENT = None
_SERVICE_OK = None


def _client():
    global _CLIENT, _SERVICE_OK
    if _CLIENT is None:
        _CLIENT = MetricsClient(CONDITIONING_CONFIG["socket_path"])
        _SERVICE_OK = CONDITIONING_CONFIG["enabled"] and _CLIENT.available()
    return _CLIENT if _SERVICE_OK else None


def _urdf_xml(design):
    root = design_to_urdf(design, name="micard_arm")
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


@register_metric("conditioning")
def conditioning_metrics(design, session=None):
    """Batch-sample configurations and aggregate trac_ik conditioning metrics."""
    client = _client()
    if client is None or design.dof == 0:
        return {}  # graceful fallback: placeholder manipulability stays in charge

    try:
        urdf_xml = _urdf_xml(design)
        tip = f"link_{design.dof - 1}"
        reg = client.register_chain(
            urdf_xml, base_link="base_link", tip_link=tip,
            srdf=ET.tostring(design_to_srdf(design), encoding="unicode"),
            collision=CONDITIONING_CONFIG["collision"],
        )
        chain_id = reg["chain_id"]

        # Surface SRDF health once per chain into traceability [micard-metrics README].
        if session and (reg.get("always_colliding_pairs")
                        or reg.get("missing_collision_geometry")):
            session.log_intermediate("chain_registration_warnings", {
                "chain_id": chain_id,
                "always_colliding_pairs": reg.get("always_colliding_pairs"),
                "missing_collision_geometry": reg.get("missing_collision_geometry"),
            })

        rng = np.random.default_rng(CONDITIONING_CONFIG["seed"])
        configs = rng.uniform(-np.pi, np.pi,
                              size=(CONDITIONING_CONFIG["n_samples"], design.dof)).tolist()
        rows = client.metrics(chain_id, configs,
                              task_dir=CONDITIONING_CONFIG["task_dir"])
        rows = rows.get("results", rows)  # tolerate either envelope

        def agg(key):
            vals = [r[key] for r in rows if r.get(key) is not None]
            return (float(np.mean(vals)), float(np.max(vals))) if vals else (0.0, 0.0)

        w_mean, w_max = agg("w")
        kinv_mean, _ = agg("k_inv")
        kmin_mean, _ = agg("k_min")
        collide_frac = float(np.mean([1.0 if r.get("in_collision") else 0.0
                                      for r in rows])) if rows else 0.0

        objective_key = CONDITIONING_CONFIG["objective_metric"]
        objective_mean = {"w": w_mean, "k_inv": kinv_mean, "k_min": kmin_mean}\
            .get(objective_key, w_mean)

        return {
            # Overwrite the placeholder: real trac_ik w becomes 'manipulability'.
            "manipulability": objective_mean,
            "manipulability_max": w_max,
            "cond_w_mean": w_mean,
            "cond_k_inv_mean": kinv_mean,
            "cond_k_min_mean": kmin_mean,
            "cond_collision_fraction": collide_frac,
            "cond_source": "micard-metrics",
        }
    except Exception as exc:
        # Any service hiccup -> fall back silently to the placeholder.
        return {"cond_error": str(exc)}