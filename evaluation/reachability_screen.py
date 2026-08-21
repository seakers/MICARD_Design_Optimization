"""Pose-based reachability screening against the scenario targets.

High-fidelity stage of the two-step use case [9]: run only on feasible Pareto
designs, not the per-design sweep. Uses /optimize_at_poses, which returns
reachability + best-conditioned config per pose in one batched request and
flags unreachable poses rather than failing [micard-metrics]. Targets are the
platform's connection ports and the ORU pickup location [9].
"""
import numpy as np
from xml.etree import ElementTree as ET

from outputs.urdf_export import design_to_urdf, design_to_srdf
from utils.metrics_client import MetricsClient
from evaluation.conditioning import CONDITIONING_CONFIG


# --- Editable scenario config (platform 1.4m x 0.875m, 6 ports + 1 side) [9] ---
SCENARIO_CONFIG = {
    "objective_metric": "k_inv",   # best-conditioned way to reach each pose
    "port_task_dir": [0.0, 0.0, 1.0],   # insertion axis into a surface port
    "seed": 0,
    # Six ports evenly across the 1.4 x 0.875 m surface + one side port [9].
    "surface_ports": [
        {"position": [x, y, 0.0], "rpy": [0.0, 0.0, 0.0]}
        for x in (-0.35, 0.35)
        for y in (-0.29, 0.0, 0.29)
    ],
    "side_port": {"position": [0.70, 0.0, 0.10], "rpy": [0.0, -1.5708, 0.0]},
    # ORU on a visiting vehicle within 2 m of the platform [9].
    "oru_pickup": {"position": [1.20, 0.0, 0.30], "rpy": [0.0, 1.5708, 0.0]},
}


def _targets():
    """Assemble the full target list; easy to reorder or extend [9]."""
    targets = list(SCENARIO_CONFIG["surface_ports"])
    targets.append(SCENARIO_CONFIG["side_port"])
    targets.append(SCENARIO_CONFIG["oru_pickup"])
    return targets


def screen_reachability(design, session=None, client=None):
    """Return per-target reachability + best value for a single Design.

    Runs one /optimize_at_poses call for all scenario targets. Returns {} if the
    service is unreachable so this stage degrades gracefully like the rest.
    """
    client = client or MetricsClient(CONDITIONING_CONFIG["socket_path"])
    if design.dof == 0 or not client.available():
        return {}

    try:
        urdf_xml = ET.tostring(design_to_urdf(design, "micard_arm"),
                               encoding="unicode")
        srdf_xml = ET.tostring(design_to_srdf(design), encoding="unicode")
        tip = f"link_{design.dof - 1}"
        reg = client.register_chain(
            urdf_xml, base_link="base_link", tip_link=tip,
            srdf=srdf_xml, collision=CONDITIONING_CONFIG["collision"],
        )
        chain_id = reg["chain_id"]

        targets = _targets()
        resp = client.optimize_at_poses(
            chain_id,
            objective=SCENARIO_CONFIG["objective_metric"],
            poses=targets,
            seed=SCENARIO_CONFIG["seed"],
            task_dir=SCENARIO_CONFIG["port_task_dir"],
        )
        results = resp.get("results", resp)

        reachable = [bool(r.get("reachable")) for r in results]
        values = [r.get("value") for r in results]
        n_reach = int(sum(reachable))

        report = {
            "reach_targets_total": len(targets),
            "reach_targets_reachable": n_reach,
            "reach_fraction": n_reach / len(targets) if targets else 0.0,
            "reach_all_ports": all(reachable[:-1]),   # every port but the ORU
            "reach_oru_pickup": reachable[-1] if reachable else False,
            "reach_per_target": [
                {"pose": t, "reachable": rc, "value": v}
                for t, rc, v in zip(targets, reachable, values)
            ],
        }

        if session:
            session.log_intermediate("reachability_screen", {
                "chain_id": chain_id, **report,
            })
        return report
    except Exception as exc:
        return {"reach_error": str(exc)}