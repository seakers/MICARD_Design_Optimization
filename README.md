# MICARD — Robotic Arm Design Optimization

Multi-objective design-space exploration for modular serial robotic arms. Arms are
assembled from a catalog of interchangeable motors and tube segments, evaluated with
pluggable kinematic/manipulability metrics, and optimized with interchangeable search
strategies (random search, a custom NSGA-II genetic algorithm, and PPO). The top
feasible Pareto-optimal designs from each run are exported as URDF/SRDF, a bill of
materials, a 3D viewer, and design-space plots — all logged to a traceable session
folder so a full run can be reconstructed later.

> This README is based on a subset of the codebase (files listed at the bottom are
> still referenced but not included). Happy to refine details further once those are
> available.

## Install

```bash
./setup.sh                  # creates .venv here, installs the ppo (torch) extra too
```

`torch` is heavy and version-pinned, so this repo keeps its own venv rather than
assuming it's installed into a caller's. `pip install -e .` (or `-e ".[ppo]"`) also
works directly, e.g. for development against this repo in its own right.

## Pipeline

`micard_design_optimization.run.run()` — or `main.py` / the `micard-design-optimization`
console script as a CLI wrapper over it — runs one end-to-end session:

1. **Session** — a timestamped run folder is created for traceability (designs,
   intermediate results, plots) via `utils/session.py`.
2. **Catalog** — `utils/catalog.load_catalog()` loads a JSON list of motor/actuator
   components into a `Catalog`, dropping any missing `nominal_torque_nm`, `weight_g`,
   or `cost_usd` (configurable via `required_fields`). Optional `defaults` can backfill
   missing fields instead of dropping, and kept-but-incomplete components are tagged
   `_incomplete: True` for downstream transparency. The schema is otherwise
   unconstrained, so an SME can edit the catalog JSON without touching code.
3. **Design space + problem** — `DesignSpace` (catalog, max joints, port count) and
   `RobotArmProblem` are built once and shared across all optimizers.
4. **Optimizers** — Random Search, Genetic Algorithm, and PPO run through the same
   `OptimizationMethod` wrapper and `evaluate()` contract, each writing into its own
   storage bucket.
5. **Hypervolume comparison** — a combined convergence plot is exported across methods.
6. **Export** — for each method, the feasible Pareto front is recomputed, and the top
   designs are reachability-screened, saved, and exported (URDF/SRDF, BOM, viewer).

## Design representation

- `Design` (`utils/design.py`): a `base_port` plus an ordered list of `Segment`s, each
  a motor (from the catalog), joint orientation (`roll`/`pitch`/`yaw`), link length,
  tube diameter/wall, and material. `metrics` holds every computed quantity.
- `DesignSpace` (`utils/design_space.py`): encodes/decodes designs as a **fixed-length
  gene vector** so random search, the GA, and PPO can all operate on one shared
  representation. Layout: `[base_port, active_length, (motor_idx, orient_idx,
  link_len, tube_idx) × max_joints]`. `decode()` only reads the first `active_length`
  slots, so unused slots act as padding for variable-length (4–7 DoF) arms.

## Optimizers

All three optimizers share the same return contract — `all_des`, `all_obj`,
`all_constraints`, `pareto_front_obj`, `hypervolumes`, `num_objectives` — which is what
the hypervolume plot and export step consume, making them drop-in swappable.

| File | Approach |
|---|---|
| `optimization/random_search.py` | Uniform sampling directly from the gene space. |
| `optimization/genetic_algorithm.py` | Custom NSGA-II: fast non-dominated sort, crowding distance, and Deb constrained-dominance (feasible beats infeasible; among infeasible, lower total violation wins). No `pygad`/`pymoo` dependency. |
| `optimization/ppo_optimizer.py` | Transformer actor/critic PPO. The actor emits a full gene vector per sample, conditioned on random preference weights (weighted-sum scalarization); reward is penalized by constraint violation. `torch` is imported locally so RS/GA still run without it installed. |

## Pareto front & hypervolume (`utils/pareto.py`)

A dependency-free (no `pymoo`) replacement for the usual Pareto/HV utilities, shared
by all optimizers and the export step:

- `is_pareto_efficient(points)` — standard minimization Pareto mask.
- `normalize_objectives(all_obj, objective_min_max)` — scales objectives into
  `[0, 1]` and flips `"max"` objectives to `1 - norm` so everything is minimization
  against a unit reference point.
- `hypervolume(front_points, ref_point)` — exact hypervolume via recursive dimension
  sweep (cheap and exact at the 2–3 objectives used here; generalizes to more at
  higher cost).
- `pareto_progress(all_obj, all_constraints, objective_min_max)` — the main entry
  point: walks evaluations in order, mapping infeasible designs to the reference
  point, and returns the running Pareto front and hypervolume **history** (one value
  per NFE, carrying the last value forward when there's no improvement). This is what
  produces each optimizer's `hypervolumes` list and what `export_hypervolume_plot`
  compares across methods.

## Evaluation / metrics

Metrics are computed through a **pluggable registry** so new metrics can be added
without touching the optimizers:

- `evaluation/base.py` — `@register_metric(name)` decorator; `compute_all_metrics()`
  runs every registered function and merges results into `design.metrics`.
- `evaluation/conditioning.py` — registers a `"conditioning"` metric that calls an
  external `micard-metrics` service (trac_ik-backed) over a Unix socket to sample
  configurations and compute manipulability (`w`, the Yoshikawa index), `k_inv`,
  `k_min`, and collision fraction. The objective metric is a single config value
  (`CONDITIONING_CONFIG["objective_metric"]`). If the service is unreachable, it
  degrades gracefully and returns `{}`, leaving a placeholder manipulability metric
  authoritative.
- `evaluation/reachability_screen.py` — a **second, higher-fidelity stage** run only
  on the final exported Pareto designs (not the whole sweep). One batched
  `/optimize_at_poses` call checks reachability + best-conditioned config against the
  scenario's fixed targets: six surface mounting ports, one side port, and an ORU
  pickup pose.
- `evaluation/problem.py` (`RobotArmProblem`) — the shared evaluation entry point all
  three optimizers call:
  - **Objectives** (`OBJECTIVE_KEYS`): `manipulability` (max), `cost` (min), `weight`
    (min) — pulled straight out of `design.metrics` after `compute_all_metrics()` runs,
    so `conditioning.py`'s real `manipulability` transparently replaces any placeholder.
  - **Constraints** (`CONSTRAINT_KEYS`): `mass`, `structural`, `reach`, `dof` — each
    returned as a **graded violation** (`0.0` = satisfied, positive = distance to
    feasible) rather than a flat penalty, so GA/PPO get a feasibility gradient to
    optimize against:
    - `mass`: kg over `max_mass_kg` (launch-mass budget).
    - `structural`: yield-margin overrun + deflection-ratio overrun summed.
    - `reach`: meters short of `required_reach_m` (ORU retrieval distance).
    - `dof`: flat `1.0` if the design has zero joints, else `0.0`.
  - `evaluate(genes)` decodes the genes into a `Design`, runs `compute_all_metrics()`,
    and returns `(objectives, is_constrained, constraint_vals)` — `is_constrained` is
    `True` when total violation exceeds `feasibility_tol`. The decoded design is
    cached on `self.last_design` so optimizers can save it via `session.save_design()`.
  - At construction, `RobotArmProblem` checks that every key in `OBJECTIVE_KEYS` is
    actually registered as a metric and raises immediately if not — a guard against
    silently optimizing against a missing/placeholder objective.

## Outputs

- `outputs/urdf_export.py` — URDF + SRDF generation from a `Design`.
- `outputs/bom.py` — bill of materials export.
- `outputs/visualization.py` — 3D viewer render, per-method Pareto plot, and the
  cross-method hypervolume comparison plot.
- `utils/session.py` — a `Session` creates a timestamped folder
  (`<output_root>/<session_id>/`) with subfolders for every traceability artifact:
  `designs/` (`save_design`), `intermediate/` (`log_intermediate`, used for
  per-design objectives, GA/PPO training curves, reachability reports, run config),
  `prompts/` and `responses/` (`log_prompt` / `log_response`, for any LLM-in-the-loop
  steps), `outputs/` (URDF/SRDF/BOM/plots/viewer), and `meta/` (auto-logged
  session id + creation time, and anywhere model/version metadata is recorded). Every
  write is timestamped and JSON-serialized, so a full run is reconstructable after
  the fact. `output_root` (default `"designs"`, relative to the caller's cwd) is a
  `run()` parameter — an embedding caller passes its own `output_root` (e.g. a
  subfolder of its project's own storage) so runs land there instead of this repo's
  working directory. Each call still gets its own (datetime-based) session-id
  subfolder underneath it, so concurrent/repeated runs against the same
  `output_root` never collide.

## Running

As a library:

```python
from micard_design_optimization.run import run

result = run(
    methods=["random_search", "genetic_algorithm"],  # default: all three, incl. "ppo"
    output_root="/path/to/some/project/design-optimization",
    seed=0,
)
```

Returns a JSON-serializable dict — session id/root, and per method: evaluation counts,
final hypervolume, the design-space plot, and one manifest entry per exported Pareto
design (objectives, reachability, URDF/SRDF/BOM/viewer paths). See `run()`'s docstring
for the full shape.

As a CLI:

```bash
python main.py                      # from a checkout, no install needed
micard-design-optimization          # equivalent, once pip installed
micard-design-optimization --json   # one JSON line on stdout instead of a summary
```

| Flag | Meaning |
|---|---|
| `--catalog-path` | Component catalog JSON (default: the bundled Dynamixel catalog). |
| `--max-joints` | Max DoF an arm can have (gene vector length). |
| `--n-ports` | Platform mounting ports the base can attach to. |
| `--random-evals` | NFE for random search (default: `ga_pop * ga_gen`). |
| `--ga-pop`, `--ga-gen` | GA population size / generations (also PPO's `epochs`/`mini_batch_size`). |
| `--seed` | RNG seed. |
| `--methods` | Any subset of `random_search genetic_algorithm ppo` (default: all three). |
| `--output-root` | See `Session`, above. |
| `--json` | One JSON line on stdout; everything else goes to stderr, so stdout is safe to parse directly. |

## Extending

- **New metric**: write `fn(design) -> dict`, decorate with
  `@register_metric("name")` in a module under `evaluation/`, and it's picked up by
  `compute_all_metrics()` automatically.
- **New optimizer**: implement `run_x(problem, session=None, **kwargs)` returning the
  shared result dict, then add it to `_build_methods()` in
  `run.py`.
- **New catalog**: any JSON catalog works as long as entries carry torque, weight, and
  cost; `load_catalog()` filters out anything missing those.

## Dependencies

- `numpy` — required everywhere.
- `torch` — only needed if the PPO optimizer is used (imported lazily).
- `micard-metrics` service (Unix socket) — optional; conditioning and reachability
  screening both degrade gracefully without it.

