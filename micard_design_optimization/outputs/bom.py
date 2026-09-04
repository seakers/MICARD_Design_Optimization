"""Bill of Materials for a Design, including selected materials [MICARD 4.0.3].

Pulls per-component cost and weight from the catalog [dynamixel_single_axis 1]
and lists link tubes with their chosen material [Slack][utils.materials].
"""
import csv
import json
from pathlib import Path

from micard_design_optimization.evaluation.structural import link_mass


def build_bom(design):
    """Return a list of BOM line items + a totals summary [MICARD 4.0.3]."""
    items = []
    for i, seg in enumerate(design.segments):
        m = seg.motor
        items.append({
            "ref": f"motor_{i}",
            "type": "actuator",
            "part": m.get("motor_model_number", "unknown"),
            "brand": m.get("motor_brand", ""),
            "orientation": seg.orientation,
            "qty": 1,
            "mass_kg": (m.get("weight_g", 0.0) or 0.0) / 1000.0,
            "cost_usd": m.get("cost_usd", 0.0) or 0.0,
            "source_url": m.get("source_url", ""),
        })
        items.append({
            "ref": f"link_{i}",
            "type": "structural_link",
            "part": (f"tube_{int(seg.tube_outer_d_m*1000)}x"
                     f"{int(seg.tube_wall_m*1000)}mm"),
            "brand": seg.material,
            "orientation": "-",
            "qty": 1,
            "mass_kg": link_mass(seg),
            "cost_usd": None,   # link cost model TBD [Slack]
            "source_url": "",
        })

    summary = {
        "total_cost_usd": sum((it["cost_usd"] or 0.0) for it in items),
        "total_mass_kg": sum((it["mass_kg"] or 0.0) for it in items),
        "num_actuators": design.dof,
    }
    return items, summary


def export_bom(design, out_dir, name="micard_arm"):
    """Write the BOM as both JSON and CSV; return their paths [MICARD 4.0.3]."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items, summary = build_bom(design)

    json_path = out_dir / f"{name}_bom.json"
    json_path.write_text(json.dumps({"items": items, "summary": summary},
                                    indent=2), encoding="utf-8")

    csv_path = out_dir / f"{name}_bom.csv"
    if items:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(items[0].keys()))
            writer.writeheader()
            writer.writerows(items)
    return json_path, csv_path