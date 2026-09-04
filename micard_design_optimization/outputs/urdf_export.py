"""URDF + SRDF export for a Design (XML output) [MICARD 4.0.1].

Mirrors the prototype's URDF conventions: chain grows along local +X, revolute
joint axes are X/Y/Z for roll/pitch/yaw, and links are cylinders [robot_chat 11].
The generated XML is directly consumable by the three.js viewer [robot_chat 11].
"""
from pathlib import Path
from xml.etree import ElementTree as ET

from micard_design_optimization.utils.design import ORIENTATION_AXES


def _axis_str(orientation):
    x, y, z = ORIENTATION_AXES[orientation]
    return f"{x} {y} {z}"


def design_to_urdf(design, name="micard_arm"):
    """Build a URDF ElementTree for a Design [MICARD 4.0.1]."""
    robot = ET.Element("robot", {"name": name})

    # Base link (fixed mount on the platform port) [MICARD scenario].
    ET.SubElement(robot, "link", {"name": "base_link"})

    prev_link = "base_link"
    for i, seg in enumerate(design.segments):
        link_name = f"link_{i}"

        # Link visual: a cylinder along local +X (rotated from default +Z) [robot_chat 11].
        link = ET.SubElement(robot, "link", {"name": link_name})
        visual = ET.SubElement(link, "visual")
        # Shift so the tube spans from the joint outward along +X.
        ET.SubElement(visual, "origin",
                      {"xyz": f"{seg.link_length_m / 2.0} 0 0",
                       "rpy": "0 1.5707963 0"})   # rotate +Z cylinder onto +X
        geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(geometry, "cylinder",
                      {"radius": f"{seg.tube_outer_d_m / 2.0}",
                       "length": f"{seg.link_length_m}"})

        # Revolute joint connecting prev_link -> this link.
        joint = ET.SubElement(robot, "joint",
                              {"name": f"joint_{i}", "type": "revolute"})
        ET.SubElement(joint, "parent", {"link": prev_link})
        ET.SubElement(joint, "child", {"link": link_name})
        # Place the child link one previous-link-length along +X (0 for first).
        offset = 0.0 if i == 0 else design.segments[i - 1].link_length_m
        ET.SubElement(joint, "origin", {"xyz": f"{offset} 0 0", "rpy": "0 0 0"})
        ET.SubElement(joint, "axis", {"xyz": _axis_str(seg.orientation)})
        # Effort from the motor's nominal torque; limits generous for now [1].
        effort = seg.motor.get("nominal_torque_nm") or 0.0
        ET.SubElement(joint, "limit",
                      {"lower": "-3.14159", "upper": "3.14159",
                       "effort": f"{effort}", "velocity": "3.0"})

        prev_link = link_name

    ET.indent(robot, space="  ")
    return robot


def design_to_srdf(design, name="micard_arm"):
    """Minimal SRDF declaring the arm as a planning group [MICARD 4.0.1]."""
    robot = ET.Element("robot", {"name": name})
    group = ET.SubElement(robot, "group", {"name": "arm"})
    for i in range(design.dof):
        ET.SubElement(group, "joint", {"name": f"joint_{i}"})
    ET.indent(robot, space="  ")
    return robot


def export_urdf(design, out_dir, name="micard_arm"):
    """Write <name>.urdf and <name>.srdf; return their paths [MICARD 4.0.1]."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    urdf_path = out_dir / f"{name}.urdf"
    srdf_path = out_dir / f"{name}.srdf"

    ET.ElementTree(design_to_urdf(design, name)).write(
        urdf_path, encoding="unicode", xml_declaration=True)
    ET.ElementTree(design_to_srdf(design, name)).write(
        srdf_path, encoding="unicode", xml_declaration=True)
    return urdf_path, srdf_path