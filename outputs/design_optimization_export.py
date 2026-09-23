"""Endpoint for a frontend to persist a design-optimization result into the
shared project contract:

  CSV:   projects/design_evaluation.csv
  URDF:  tool_artifacts/design_optimization/<run_id>/designs/<design_id>/robot.urdf
  JSON:  tool_artifacts/design_optimization/<run_id>/designs/<design_id>/design.json

Call `export_design(...)` per design; it writes all three.
"""
import csv
import json
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

from outputs.urdf_export import design_to_urdf

# Exact contract column order. `null` is emitted for any missing metric.
CSV_COLUMNS = [
    "design id", "optimization run id", "cost", "struct_feasible",
    "struct_mass_kg", "max_deflection_ratio", "min_yield_margin", "weight",
    "motor_mass_kg", "link_mass_kg", "manipulability", "manipulability_max",
    "abs_reach_m", "workspace_volume_m3", "power_proxy", "power_w",
    "total_violation",
]

# Metric columns -> keys in design.metrics. LHS is the CSV column, RHS the
# metrics key. Keys confirmed present in the code: cost/manipulability/weight
# [10], total_violation [10], min_yield_margin/max_deflection_ratio/abs_reach_m
# [10]. Others resolve if the metric modules register them [11]; else -> null.
METRIC_KEYS = {
    "cost": "cost",
    "struct_feasible": "struct_feasible",
    "struct_mass_kg": "struct_mass_kg",
    "max_deflection_ratio": "max_deflection_ratio",
    "min_yield_margin": "min_yield_margin",
    "weight": "weight",
    "motor_mass_kg": "motor_mass_kg",
    "link_mass_kg": "link_mass_kg",
    "manipulability": "manipulability",
    "manipulability_max": "manipulability_max",
    "abs_reach_m": "abs_reach_m",
    "workspace_volume_m3": "workspace_volume_m3",
    "power_proxy": "power_proxy",
    "power_w": "power_w",
    "total_violation": "total_violation",
}

PROJECTS_CSV = Path("projects/design_evaluation.csv")
ARTIFACT_ROOT = Path("tool_artifacts/design_optimization")


# --- ID generation (reuses the existing timestamp style) ---
def new_run_id():
    """Unique id per optimization run."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def new_design_id():
    """Unique id per design, in the design_<HHMMSS>_<micros> shape."""
    now = datetime.now()
    return f"design_{now.strftime('%H%M%S')}_{now.microsecond:06d}"


def _fmt(value):
    """CSV formatting: None -> 'null', bools lowercase (matches sample row)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _ensure_csv_header(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_COLUMNS)


def append_csv_row(design, design_id, run_id, csv_path=PROJECTS_CSV):
    """Append one contract row for a design to the shared CSV."""
    csv_path = Path(csv_path)
    _ensure_csv_header(csv_path)

    metrics = getattr(design, "metrics", {}) or {}
    row = {"design id": design_id, "optimization run id": run_id}
    for col, key in METRIC_KEYS.items():
        row[col] = _fmt(metrics.get(key))

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(row)


def build_design_json(design, design_id, method=None):
    """Build the design.json payload. Joint names match design_to_urdf's
    f"joint_{i}" convention [6]; motor brand/model come from seg.motor,
    the same fields the BOM reads [5]."""
    components = []
    for i, seg in enumerate(design.segments):
        m = seg.motor
        components.append({
            "name": f"joint_{i}",
            "type": "motor",
            "suggestions": [{
                "brand": m.get("motor_brand", ""),
                "model": m.get("motor_model_number", "unknown"),
                "recommended": "true",
            }],
        })
    payload = {"id": design_id, "components": components}
    if method is not None:
        payload["method"] = method  # optimization method tag
    return payload


def export_artifacts(design, design_id, run_id, method=None):
    """Write robot.urdf and design.json into the contract directory layout."""
    design_dir = ARTIFACT_ROOT / run_id / "designs" / design_id
    design_dir.mkdir(parents=True, exist_ok=True)

    urdf_root = design_to_urdf(design, name=design_id)   # reuse exporter [6]
    ET.ElementTree(urdf_root).write(
        design_dir / "robot.urdf", encoding="unicode", xml_declaration=True)

    (design_dir / "design.json").write_text(
        json.dumps(build_design_json(design, design_id, method), indent=2),
        encoding="utf-8")

    return design_dir


def export_design(design, run_id=None, design_id=None, method=None,
                  csv_path=PROJECTS_CSV):
    """Endpoint: persist one design's CSV row + robot.urdf + design.json.

    Args:
        design: a decoded Design (has .segments and .metrics) [8].
        run_id: optimization run id; generated if omitted.
        design_id: unique design id; generated if omitted.
        method: optimizer name, stored as a tag in design.json.
        csv_path: override for the shared evaluation CSV.

    Returns:
        dict with design_id, run_id, and the artifact directory path.
    """
    run_id = run_id or new_run_id()
    design_id = design_id or new_design_id()

    design_dir = export_artifacts(design, design_id, run_id, method=method)
    append_csv_row(design, design_id, run_id, csv_path=csv_path)

    return {
        "design_id": design_id,
        "run_id": run_id,
        "artifact_dir": str(design_dir),
    }