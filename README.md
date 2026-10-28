# MICARD — Robotic Arm Design Optimization

Multi-objective design-space exploration for modular serial robotic arms. Arms are
assembled from a catalog of interchangeable motors and tube segments, evaluated with
pluggable kinematic/manipulability metrics, and optimized with interchangeable search
strategies (random search, a custom NSGA-II genetic algorithm, and PPO). The
objectives and feasibility constraints are themselves configurable — the larger tool
driving a run can pass them in as JSON without touching code. The top feasible
Pareto-optimal designs from each run are exported as URDF/SRDF, a bill of materials, a
3D viewer, and design-space plots — all logged to a traceable session folder so a full
run can be reconstructed later.

> This README is based on a subset of the codebase (files listed at the bottom are
> still referenced but not included). Happy to refine details further once those are
> available.

## Pipeline

`main.py` runs one end-to-end session:

1. **Session** — a timestamped run folder is created for traceability (designs,
   intermediate results, plots) via `utils/session.py`.
2. **Catalog** — `utils/catalog.load_catalog()` loads a JSON list of motor/actuator
   components into a `Catalog`, dropping any missing `nominal_torque_nm`, `weight_g`,
   or `cost_usd` (configurable via `required_fields`). Optional `defaults` can backfill
   missing fields instead of dropping, and kept-but-incomplete components are tagged
   `_incomplete: True` for downstream transparency. The schema is otherwise
   unconstrained, so an SME can edit the catalog JSON without touching code.
3. **Design space + problem** — `DesignSpace` (catalog, max joints, port count) and
   `RobotArmProblem` are built once and shared across all optimizers. The problem's
   objectives and constraints come from a `ProblemConfig` (defaults reproduce the
   original mission setup; a caller can override them via `--problem-config` or by
   passing `problem_config` into `run()`).
4. **Optimizers** — Random Search, Genetic Algorithm, and PPO run through the same
   `OptimizationMethod` wrapper and `evaluate()` contract, each writing into its own
   storage bucket.
5. **Hypervolume comparison** — a combined convergence plot is exported across methods.
6. **Export** — for each method, the feasible Pareto front is recomputed, and the top
   designs are reachability-screened, saved, and exported (URDF/SRDF, BOM, viewer).

## Design representation

- `Design` (`utils/design.py`): a `base_port` plus an ordered list of `Segment`s, each
  a motor (from the catalog), joint orientation (roll/pitch/yaw), link length,
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
without touching the optimizers [3]:

- `evaluation/base.py` — `@register_metric(name)` decorator; `compute_all_metrics()`
  runs every registered function and merges results into `design.metrics` [3].
- `evaluation/conditioning.py` — registers a `"conditioning"` metric that calls an
  external `micard-metrics` service (trac_ik-backed) over a Unix socket to sample
  configurations and compute manipulability (`w`, the Yoshikawa index), `k_inv`,
  `k_min`, and collision fraction [6]. The objective metric is a single config value
  (`CONDITIONING_CONFIG["objective_metric"]`) [6]. If the service is unreachable, it
  degrades gracefully and returns `{}`, leaving a placeholder manipulability metric
  authoritative [6].
- `evaluation/reachability_screen.py` — a **second, higher-fidelity stage** run only
  on the final exported Pareto designs (not the whole sweep). One batched
  `/optimize_at_poses` call checks reachability + best-conditioned config against the
  scenario's fixed targets: six surface mounting ports, one side port, and an ORU
  pickup pose [11].
- `evaluation/problem.py` (`RobotArmProblem` + `ProblemConfig`) — the shared
  evaluation entry point all three optimizers call. **Objectives and constraints are
  config-driven**: they come from a `ProblemConfig` passed in at construction, with
  class-level defaults that reproduce the original mission setup. The caller (or the
  larger driving tool) supplies the config as a dict, a JSON string, or a JSON file
  path via `ProblemConfig.from_json(...)`.
  - **Objectives**: a list of `{"metric", "direction"}` entries. The default is
    `manipulability` (max), `cost` (min), `weight` (min) [9] — pulled straight out of
    `design.metrics` after `compute_all_metrics()` runs, so `conditioning.py`'s real
    `manipulability` transparently replaces any placeholder [6]. `objective_keys`,
    `objective_min_max`, and `num_objectives` are set per-instance from the config.
  - **Constraints**: a list of declarative **bound** specs, each a graded violation
    (`0.0` = satisfied, positive = distance to feasible) so GA/PPO get a feasibility
    gradient rather than a flat penalty [2]. Every constraint is a uniform
    `upper_bound` (violated when `metric > limit`) or `lower_bound` (violated when
    `metric < limit`) on a produced metric. The default set is:
    - `mass`: `upper_bound` on `weight` vs. the 50 kg launch-mass budget [2][13].
    - `yield`: `lower_bound` on `min_yield_margin` at 0 (margin below 0 means stress
      exceeded the allowable) [2][12].
    - `deflection`: `upper_bound` on `max_deflection_ratio` at the 1% limit [2][12].
    - `reach`: `lower_bound` on `abs_reach_m` vs. the 2 m ORU-retrieval distance [2][4].
    - `dof`: `lower_bound` on `dof` at 1, as a flat `1.0` flag (zero-joint designs have
      no meaningful gradient) [2].

    > **Note:** the original single `structural` constraint has been split into two
    > uniform bound constraints, `yield` + `deflection`, so every constraint now follows
    > the same declarative scheme. The only numeric difference from the old behavior is
    > that a design violating *both* now contributes two independent (each `[0,1]`-clipped)
    > entries instead of one combined clipped entry; feasibility flipping is unchanged.
  - `evaluate(genes)` decodes the genes into a `Design`, runs `compute_all_metrics()`,
    and returns `(objectives, is_constrained, constraint_vals, metrics)` —
    `is_constrained` is `True` when total violation exceeds `feasibility_tol` [2]. The
    decoded design is cached on `self.last_design` so optimizers can save it via
    `session.save_design()`.
  - **Validation.** Objective directions and constraint types/names are checked at
    construction (fast-fail). Metric *names* are validated lazily on the first
    `evaluate()` call, against the keys metrics actually produce rather than the
    registry names from `available_metrics()` — so an objective/constraint can
    reference any produced key (e.g. `abs_reach_m`, `min_yield_margin`) or the `dof`
    design attribute, and a typo'd key raises a clear "no metric produces" error on the
    first evaluation [3][4][12].

## Problem configuration (`--problem-config`)

The objectives and constraints are defined in a JSON config. Omit it to use the
built-in defaults (which match the original mission setup); pass a file to customize.

```json
{
  "objectives": [
    {"metric": "manipulability", "direction": "max"},
    {"metric": "cost", "direction": "min"},
    {"metric": "weight", "direction": "min"}
  ],
  "constraints": [
    {"name": "mass", "type": "upper_bound", "metric": "weight",
     "limit": 50.0, "scale": 50.0, "default": 0.0, "clip": [0.0, 1.0]},
    {"name": "yield", "type": "lower_bound", "metric": "min_yield_margin",
     "limit": 0.0, "scale": 1.0, "default": 1.0, "clip": [0.0, 1.0]},
    {"name": "deflection", "type": "upper_bound", "metric": "max_deflection_ratio",
     "limit": 0.01, "scale": 1.0, "default": 0.0, "clip": [0.0, 1.0]},
    {"name": "reach", "type": "lower_bound", "metric": "abs_reach_m",
     "limit": 2.0, "scale": 2.0, "default": 0.0, "clip": [0.0, 1.0]},
    {"name": "dof", "type": "lower_bound", "metric": "dof",
     "limit": 1.0, "scale": 1.0, "default": 0.0, "graded": false}
  ],
  "feasibility_tol": 1e-9
}
```

**Objective fields:** `metric` (any produced metric key) and `direction` (`"min"` or `"max"`).

**Constraint fields:**
- `name` — unique label, used for logging and plot columns.
- `type` — `"upper_bound"` (violated when `metric > limit`) or `"lower_bound"` (violated when `metric < limit`).
- `metric` — the produced metric key (or the `dof` design attribute) to read.
- `limit` — the threshold the metric is compared against.
- `scale` — divides the raw overrun/shortfall to normalize it (usually the budget/requirement); clamped to at least `feasibility_tol`.
- `default` — fallback value when the metric key is missing; set it to a non-violating value.
- `clip` — `[low, high]` bounds on the final graded violation (default `[0.0, 1.0]`).
- `graded` — if `false`, any violation collapses to a flat `1.0` flag (used by `dof`).

Usable metric keys (anything the registered metrics emit): `manipulability`, `manipulability_max` [9][6], `cost` [7], `weight`, `motor_mass_kg`, `link_mass_kg` [13], `abs_reach_m`, `workspace_volume_m3` [4], `struct_feasible`, `struct_mass_kg`, `max_deflection_ratio`, `min_yield_margin` [12], `power_proxy` [10], conditioning extras (`cond_w_mean`, `cond_k_inv_mean`, `cond_k_min_mean`, `cond_collision_fraction`) when the service is up [6], plus the special `dof` design attribute.

## Outputs

- `outputs/urdf_export.py` — URDF + SRDF generation from a `Design`.
- `outputs/bom.py` — bill of materials export.
- `outputs/visualization.py` — 3D viewer render, per-method Pareto plot, and the
  cross-method hypervolume comparison plot [14]. The Pareto plot distinguishes three
  point classes: the feasible Pareto front (blue), feasible-but-dominated designs
  (light grey), and infeasible designs (light red) [14]. The hypervolume plot shows a
  median line with an interquartile band across runs; when only a single HV history is
  available it falls back to plotting that one run [14].
- `utils/session.py` — a `Session` creates a timestamped folder
  (`<output_root>/<session_id>/`, default `output_root="designs"`) with subfolders for
  every traceability artifact: `designs/` (`save_design`), `intermediate/`
  (`log_intermediate`, used for per-design objectives, GA/PPO training curves,
  reachability reports, and the run config — which now also records the resolved
  problem config), `prompts/` and `responses/` (`log_prompt`/`log_response`, for any
  LLM-in-the-loop steps), `outputs/` (URDF/SRDF/BOM/plots/viewer), and `meta/`
  (auto-logged session id + creation time). Every write is timestamped and
  JSON-serialized, so a full run is reconstructable after the fact.

## Running

```bash
python main.py
python main.py --json                                  # one JSON line on stdout instead of a summary
python main.py --problem-config problem_config.json    # custom objectives/constraints
```

Key parameters (see `run()` in `main.py`) [1]:

| Flag | Meaning |
|---|---|
| `--catalog-path` | Path to the component catalog JSON. |
| `--max-joints` | Max DoF an arm can have (gene vector length). |
| `--n-ports` | Platform mounting ports the base can attach to. |
| `--random-evals` | NFE for random search (defaults to `ga_pop * ga_gen` for a fair comparison). |
| `--ga-pop`, `--ga-gen` | GA population size / generations (also reused as PPO `epochs`/`mini_batch_size`). |
| `--seed` | RNG seed for reproducibility. |
| `--problem-config` | Path to a JSON file defining objectives and constraints (default: the built-in mission setup). |
| `--no-random-search`, `--no-genetic-algorithm`, `--no-ppo` | Skip that method (default: all three run). |
| `--output-root` | Where the session folder is created (default: `designs`; see `Session`, above). |
| `--json` | Print `run()`'s result as one JSON line on stdout instead of a human-readable summary. |

## Extending

- **New metric**: write `fn(design) -> dict`, decorate with `@register_metric("name")`
  in a module under `evaluation/`, and it's picked up by `compute_all_metrics()`
  automatically [3]. Its output keys immediately become usable as objective/constraint
  metrics in the problem config.
- **New objective or constraint**: no code change — add an entry to the `objectives` or
  `constraints` list in a problem-config JSON. Objectives take a `metric` + `direction`;
  constraints are `upper_bound`/`lower_bound` specs on any produced metric.
- **New optimizer**: implement `run_x(problem, session=None, **kwargs)` returning the
  shared result dict, then add it to the `method_objs` list (and a corresponding
  enable/disable param) in `run()`, `main.py` [1].
- **New catalog**: any JSON catalog works as long as entries carry torque, weight, and
  cost; `load_catalog()` filters out anything missing those.

## Dependencies

- `numpy` — required everywhere.
- `torch` — only needed if the PPO optimizer is used (imported lazily).
- `micard-metrics` service (Unix socket) — optional; conditioning and reachability
  screening both degrade gracefully without it [6][11].