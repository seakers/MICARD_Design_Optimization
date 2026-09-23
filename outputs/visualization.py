"""Design 3D viewer (three.js) + design-space Pareto plot.

The URDF viewer reuses the prototype's three.js renderer (importmap, pose
helpers, joint sliders) so exported URDFs render interactively in a browser
[robot_chat 11]. The Pareto plot gives the required graphical representation of
the design space [MICARD 3.0.6], echoing the reference 3D Pareto plotting [main 8].
"""
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from outputs.urdf_export import design_to_urdf


# --- 3D robot viewer (reuses the prototype three.js template) [robot_chat 11] ---

def _urdf_to_viewer_data(urdf_root):
    """Parse a URDF ElementTree into the viewer JSON shape [robot_chat 11]."""
    def parse_xyz(v):
        parts = [float(p) for p in v.split()] if v else [0, 0, 0]
        return (parts + [0, 0, 0])[:3]

    links = {}
    for link in urdf_root.findall("link"):
        visuals = []
        for visual in link.findall("visual"):
            origin = visual.find("origin")
            cyl = visual.find("geometry/cylinder")
            if cyl is None:
                continue
            visuals.append({
                "type": "cylinder",
                "radius": float(cyl.attrib["radius"]),
                "length": float(cyl.attrib["length"]),
                "xyz": parse_xyz(origin.attrib.get("xyz") if origin is not None else None),
                "rpy": parse_xyz(origin.attrib.get("rpy") if origin is not None else None),
                "material": "micard_link",
            })
        links[link.attrib["name"]] = visuals

    joints = []
    for joint in urdf_root.findall("joint"):
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        joints.append({
            "name": joint.attrib["name"],
            "type": joint.attrib["type"],
            "parent": joint.find("parent").attrib["link"],
            "child": joint.find("child").attrib["link"],
            "xyz": parse_xyz(origin.attrib.get("xyz") if origin is not None else None),
            "rpy": parse_xyz(origin.attrib.get("rpy") if origin is not None else None),
            "axis": parse_xyz(axis.attrib.get("xyz") if axis is not None else None),
            "lower": float(limit.attrib.get("lower", "-3.14159")) if limit is not None else 0.0,
            "upper": float(limit.attrib.get("upper", "3.14159")) if limit is not None else 0.0,
        })

    return {"name": urdf_root.attrib.get("name", "micard_arm"),
            "materials": {}, "links": links, "joints": joints}


# The viewer HTML is the prototype's template (importmap, pose helpers, joint
# sliders, orbit controls). Kept minimal here; robot data is injected as JSON [robot_chat 11].
_VIEWER_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ URDF Viewer</title>
<style>
:root { color-scheme: light; font-family: Arial, sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; background: #edf2f6; color: #17202a; overflow: hidden; }
#scene { position: fixed; inset: 0; }
canvas { display: block; width: 100%; height: 100%; }
aside {
position: fixed; left: 16px; top: 16px; bottom: 16px;
width: min(340px, calc(100vw - 32px)); overflow: auto;
background: rgba(255,255,255,0.9); border: 1px solid rgba(124,140,156,0.35);
box-shadow: 0 18px 42px rgba(31,42,55,0.16); padding: 14px; backdrop-filter: blur(10px);
}
h1 { font-size: 18px; margin: 0 0 8px; line-height: 1.2; }
.meta { color: #526070; font-size: 12px; margin-bottom: 12px; overflow-wrap: anywhere; }
.actions { display: flex; gap: 8px; margin-bottom: 8px; }
button { border: 1px solid #c9d2dc; background: #f7f9fb; border-radius: 6px;
min-height: 32px; padding: 0 10px; color: #17202a; cursor: pointer; }
button:hover { background: #edf2f6; }
.joint { display: grid; gap: 6px; padding: 10px 0; border-top: 1px solid #e2e7ed; }
.joint label { font-size: 13px; font-weight: 700; overflow-wrap: anywhere; }
.joint output { font-variant-numeric: tabular-nums; color: #526070; font-size: 12px; }
input[type="range"] { width: 100%; }
.badge { display: inline-block; padding: 2px 6px; border-radius: 999px;
background: #e8edf3; color: #435263; font-size: 11px; margin-left: 6px; }
</style>
</head>
<body>
<div id="scene"></div>
<aside>
<h1>__TITLE__</h1>
<div class="meta">__META__</div>
<div class="actions">
<button id="reset" type="button">Reset</button>
<button id="home" type="button">Home</button>
</div>
<div id="controls"></div>
</aside>

<script type="importmap">
{ "imports": {
  "three": "https://unpkg.com/three@0.165.0/build/three.module.js",
  "three/addons/": "https://unpkg.com/three@0.165.0/examples/jsm/"
} }
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const robot = __ROBOT_JSON__;
const sceneHost = document.getElementById('scene');
const controls = document.getElementById('controls');
const jointNodes = new Map();

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xedf2f6);
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 100);
camera.up.set(0, 0, 1);
camera.position.set(0.75, -1.15, 0.72);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
sceneHost.appendChild(renderer.domElement);

const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;
orbit.enablePan = false;

scene.add(new THREE.HemisphereLight(0xffffff, 0xa8b4bf, 1.55));
const key = new THREE.DirectionalLight(0xffffff, 3.0);
key.position.set(1.8, -2.4, 2.8);
scene.add(key);

const fallbackMaterialColors = { micard_link: 0x7b8073, micard_interface: 0x5ca273 };

function vector(v) { return new THREE.Vector3(v[0] || 0, v[1] || 0, v[2] || 0); }
function applyPose(o, xyz, rpy) {
  o.position.set(xyz[0] || 0, xyz[1] || 0, xyz[2] || 0);
  o.rotation.set(rpy[0] || 0, rpy[1] || 0, rpy[2] || 0, 'XYZ');
}
function materialFor(visual) {
  return new THREE.MeshStandardMaterial({
    color: fallbackMaterialColors[visual.material] || 0x8a98a6,
    roughness: 0.48, metalness: 0.18
  });
}
function makeVisualMesh(visual) {
  let mesh;
  if (visual.type === 'cylinder') {
    mesh = new THREE.Mesh(new THREE.CylinderGeometry(visual.radius, visual.radius, visual.length, 40), materialFor(visual));
    mesh.rotation.x = Math.PI / 2;
  } else if (visual.type === 'box') {
    mesh = new THREE.Mesh(new THREE.BoxGeometry(visual.size[0], visual.size[1], visual.size[2]), materialFor(visual));
  } else if (visual.type === 'sphere') {
    mesh = new THREE.Mesh(new THREE.SphereGeometry(visual.radius, 32, 20), materialFor(visual));
  } else { return null; }
  const group = new THREE.Group();
  applyPose(group, visual.xyz, visual.rpy);
  group.add(mesh);
  return group;
}
function addLinkVisuals(group, linkName) {
  for (const visual of robot.links[linkName] || []) {
    const mesh = makeVisualMesh(visual);
    if (mesh) group.add(mesh);
  }
}

const childrenByParent = new Map();
const childLinks = new Set();
for (const joint of robot.joints) {
  if (!childrenByParent.has(joint.parent)) childrenByParent.set(joint.parent, []);
  childrenByParent.get(joint.parent).push(joint);
  childLinks.add(joint.child);
}
const rootLink = Object.keys(robot.links).find((n) => !childLinks.has(n)) || Object.keys(robot.links)[0];
const robotRoot = new THREE.Group();
scene.add(robotRoot);

function buildLink(linkName, parentGroup) {
  const linkGroup = new THREE.Group();
  addLinkVisuals(linkGroup, linkName);
  parentGroup.add(linkGroup);
  for (const joint of childrenByParent.get(linkName) || []) {
    const originGroup = new THREE.Group();
    applyPose(originGroup, joint.xyz, joint.rpy);
    linkGroup.add(originGroup);
    const axisGroup = new THREE.Group();
    originGroup.add(axisGroup);
    if (joint.type === 'revolute' || joint.type === 'continuous') {
      jointNodes.set(joint.name, { node: axisGroup, axis: vector(joint.axis).normalize() });
    }
    buildLink(joint.child, axisGroup);
  }
}
buildLink(rootLink, robotRoot);

function setJointAngle(name, value) {
  const entry = jointNodes.get(name);
  if (entry) entry.node.setRotationFromAxisAngle(entry.axis, value);
}
function addJointControls() {
  const moving = robot.joints.filter((j) => j.type === 'revolute' || j.type === 'continuous');
  if (!moving.length) { controls.textContent = 'No movable joints.'; return; }
  for (const joint of moving) {
    const row = document.createElement('div');
    row.className = 'joint';
    const label = document.createElement('label');
    label.textContent = joint.name;
    const badge = document.createElement('span');
    badge.className = 'badge';
    badge.textContent = `axis ${joint.axis.map((v) => Number(v).toFixed(0)).join(' ')}`;
    label.append(badge);
    const output = document.createElement('output');
    output.textContent = '0.00 rad';
    const input = document.createElement('input');
    input.type = 'range';
    input.min = joint.type === 'continuous' ? -Math.PI : joint.lower;
    input.max = joint.type === 'continuous' ? Math.PI : joint.upper;
    input.step = 0.01; input.value = 0;
    input.addEventListener('input', () => {
      const v = Number(input.value);
      setJointAngle(joint.name, v);
      output.textContent = `${v.toFixed(2)} rad`;
    });
    row.append(label, input, output);
    controls.append(row);
  }
}
function frameRobot() {
  const box = new THREE.Box3().setFromObject(robotRoot);
  if (box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z, 0.3);
  orbit.target.copy(center);
  camera.position.set(center.x + radius * 1.45, center.y - radius * 1.8, center.z + radius * 1.15);
  camera.near = Math.max(radius / 100, 0.001);
  camera.far = Math.max(radius * 20, 10);
  camera.updateProjectionMatrix();
  orbit.update();
}
function resetJoints() {
  for (const name of jointNodes.keys()) setJointAngle(name, 0);
  for (const input of controls.querySelectorAll('input[type="range"]')) input.value = 0;
  for (const output of controls.querySelectorAll('output')) output.textContent = '0.00 rad';
}
document.getElementById('reset').addEventListener('click', resetJoints);
document.getElementById('home').addEventListener('click', frameRobot);
addJointControls();
frameRobot();
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
function animate() { orbit.update(); renderer.render(scene, camera); requestAnimationFrame(animate); }
animate();
</script>
</body>
</html>
"""


def export_viewer(design, out_dir, name="micard_arm"):
    """Write an interactive three.js URDF viewer HTML for a Design [robot_chat 11]."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    urdf_root = design_to_urdf(design, name)
    data = _urdf_to_viewer_data(urdf_root)
    payload = json.dumps(data)

    html_text = (_VIEWER_HTML
                 .replace("__TITLE__", name)
                 .replace("__META__", f"{design.dof} DOF, base port {design.base_port}")
                 .replace("__ROBOT_JSON__", payload))

    html_path = out_dir / f"{name}_viewer.html"
    html_path.write_text(html_text, encoding="utf-8")
    return html_path


# --- Design-space Pareto plot (graphical representation of design space) [9] ---

def export_pareto_plot(all_obj, all_constraints, objective_labels, out_dir,
                       name="pareto", highlight_indices=None):
    """Scatter of the 3 optimization objectives, highlighting the Pareto front.

    Gives the required graphical representation of where each design falls in
    the design space [MICARD 3.0.6], echoing the reference 3D Pareto plot [main 8].
    Written as a self-contained interactive HTML (Plotly via CDN) so it needs
    no extra Python dependency and opens in any browser.
    """
    import numpy as np
    from utils.pareto import is_pareto_efficient, normalize_objectives

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_obj = np.asarray(all_obj, dtype=float)
    all_constraints = np.asarray(all_constraints, dtype=bool)

    # Determine the Pareto front among feasible designs (minimization-normalized) [utils.pareto].
    objective_min_max = ["max", "min", "min"]  # matches RobotArmProblem [problem.py]
    feasible_idx = np.where(~all_constraints)[0]
    pareto_idx = []
    if len(feasible_idx) > 0:
        hv_obj, _ = normalize_objectives(all_obj[feasible_idx], objective_min_max)
        mask = is_pareto_efficient(hv_obj, return_mask=True)
        pareto_idx = list(feasible_idx[mask])

    # Build point traces.
    def rows(indices):
        return [all_obj[i].tolist() for i in indices]

    all_idx = list(range(len(all_obj)))
    dominated_idx = [i for i in all_idx if i not in pareto_idx]

    data = {
        "labels": objective_labels,
        "dominated": rows(dominated_idx),
        "pareto": rows(pareto_idx),
    }
    payload = json.dumps(data)

    html_text = _PARETO_HTML.replace("__DATA_JSON__", payload).replace("__TITLE__", name)
    html_path = out_dir / f"{name}_design_space.html"
    html_path.write_text(html_text, encoding="utf-8")
    return html_path


_PARETO_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ - Design Space</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
body { margin: 0; font-family: Arial, sans-serif; background: #edf2f6; color: #17202a; }
#plot { width: 100vw; height: 100vh; }
</style>
</head>
<body>
<div id="plot"></div>
<script>
const data = __DATA_JSON__;
function col(rows, i) { return rows.map(r => r[i]); }
const dominated = {
  type: 'scatter3d', mode: 'markers', name: 'Dominated',
  x: col(data.dominated, 0), y: col(data.dominated, 1), z: col(data.dominated, 2),
  marker: { size: 3, color: '#9fb0bf', opacity: 0.55 }
};
const pareto = {
  type: 'scatter3d', mode: 'markers', name: 'Pareto front',
  x: col(data.pareto, 0), y: col(data.pareto, 1), z: col(data.pareto, 2),
  marker: { size: 6, color: '#1f6fe5' }
};
const layout = {
  title: '__TITLE__ design space',
  scene: {
    xaxis: { title: data.labels[0] },
    yaxis: { title: data.labels[1] },
    zaxis: { title: data.labels[2] }
  },
  margin: { l: 0, r: 0, t: 40, b: 0 }
};
Plotly.newPlot('plot', [dominated, pareto], layout, { responsive: true });
</script>
</body>
</html>
"""

def export_hypervolume_plot(storage, out_dir, name="hypervolume_comparison"):
    """Plot median hypervolume with an IQR band across independent runs, per method.

    Consumes storage[method]["all_runs_hypervolumes"] (one per-NFE HV history
    per run) and overlays median + interquartile band, echoing the reference
    median/IQR hypervolume plot [2]. Self-contained interactive HTML via CDN.
    """
    from pathlib import Path
    import numpy as np

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    series = []
    for method_name, results in storage.items():
        runs = [list(r) for r in results.get("all_runs_hypervolumes", []) if len(r)]
        if not runs:
            continue
        min_len = min(len(r) for r in runs)                 # align to shortest run
        mat = np.array([r[:min_len] for r in runs], dtype=float)

        series.append({
            "name": method_name,
            "x": list(range(1, min_len + 1)),
            "median": np.median(mat, axis=0).tolist(),
            "q25": np.percentile(mat, 25, axis=0).tolist(),
            "q75": np.percentile(mat, 75, axis=0).tolist(),
        })

    payload = json.dumps({"series": series, "title": name})
    html_text = _HYPERVOLUME_HTML.replace("__DATA_JSON__", payload).replace("__TITLE__", name)
    html_path = out_dir / f"{name}.html"
    html_path.write_text(html_text, encoding="utf-8")
    return html_path


_HYPERVOLUME_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
body { margin: 0; font-family: Arial, sans-serif; background: #edf2f6; color: #17202a; }
#plot { width: 100vw; height: 100vh; }
</style>
</head>
<body>
<div id="plot"></div>
<script>
const data = __DATA_JSON__;
const palette = ['#1f6fe5', '#2ca02c', '#d62728', '#9467bd', '#ff7f0e'];
function hexToRgba(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n>>16)&255},${(n>>8)&255},${n&255},${a})`;
}
const traces = [];
data.series.forEach((s, k) => {
  const color = palette[k % palette.length];
  const xRev = s.x.slice().reverse();
  const q75Rev = s.q75.slice().reverse();
  traces.push({                       // IQR band
    type: 'scatter', mode: 'lines', name: s.name + ' IQR',
    x: s.x.concat(xRev), y: s.q25.concat(q75Rev),
    fill: 'toself', fillcolor: hexToRgba(color, 0.15),
    line: { width: 0 }, hoverinfo: 'skip', showlegend: false
  });
  traces.push({                       // median line
    type: 'scatter', mode: 'lines', name: s.name,
    x: s.x, y: s.median, line: { color: color, width: 2 }
  });
});
const layout = {
  title: 'Hypervolume vs. Evaluations (median & IQR)',
  xaxis: { title: 'Evaluations (NFE)' },
  yaxis: { title: 'Hypervolume' },
  margin: { l: 60, r: 20, t: 50, b: 50 }
};
Plotly.newPlot('plot', traces, layout, { responsive: true });
</script>
</body>
</html>
"""