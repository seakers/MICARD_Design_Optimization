"""Library entry point for one MICARD design-exploration run.

run() takes structured params in and returns a JSON-serializable result --
call it directly, in-process, if you're already in this venv (dev/test use,
or any caller that doesn't need this package's own venv kept separate from
its own). An embedding caller that DOES need that separation (e.g.
design_team's design_optimization Tool) instead invokes this package's
`--json` CLI mode via subprocess against this venv's own python -- see
cli.py's own docstring for that contract. The root-level main.py is a thin
CLI shim over cli.py, for standalone/dev use.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from micard_design_optimization.evaluation.problem import RobotArmProblem
from micard_design_optimization.evaluation.reachability_screen import screen_reachability
from micard_design_optimization.optimization.genetic_algorithm import run_genetic_algorithm
from micard_design_optimization.optimization.ppo_optimizer import run_ppo_optimization
from micard_design_optimization.optimization.random_search import run_random_search
from micard_design_optimization.outputs.bom import export_bom
from micard_design_optimization.outputs.urdf_export import export_urdf
from micard_design_optimization.outputs.visualization import (
    export_hypervolume_plot,
    export_pareto_plot,
    export_viewer,
)
from micard_design_optimization.utils.catalog import load_catalog
from micard_design_optimization.utils.design_space import DesignSpace
from micard_design_optimization.utils.pareto import is_pareto_efficient, normalize_objectives
from micard_design_optimization.utils.session import Session

log = logging.getLogger(__name__)

# Bundled alongside this package, so the default resolves regardless of the
# caller's own working directory (unlike the old cwd-relative "utils/..." path).
_DEFAULT_CATALOG_PATH = Path(__file__).resolve().parent / "utils" / "dynamixel_single_axis.json"

# Canonical order results/exports are produced in, regardless of the order
# `methods` was given in -- keeps output (and the hypervolume plot's legend)
# consistent across calls.
_METHOD_ORDER = ("random_search", "genetic_algorithm", "ppo")


@dataclass
class OptimizationMethod:
    key: str
    name: str
    runner: Callable
    kwargs: dict


def _build_methods(selected, ga_pop, ga_gen, random_evals, rng):
    available = {
        "random_search": lambda: OptimizationMethod(
            "random_search", "Random Search", run_random_search,
            {"num_exec": random_evals, "rng": rng}),
        "genetic_algorithm": lambda: OptimizationMethod(
            "genetic_algorithm", "Genetic Algorithm", run_genetic_algorithm,
            {"pop_size": ga_pop, "n_gen": ga_gen, "rng": rng}),
        # torch (imported lazily inside run_ppo_optimization) is required
        # only if "ppo" is actually selected.
        "ppo": lambda: OptimizationMethod(
            "ppo", "PPO", run_ppo_optimization,
            {"epochs": ga_pop, "mini_batch_size": ga_gen, "rng": rng}),
    }
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"unknown method(s) {sorted(unknown)} -- one of {sorted(available)}")
    if not selected:
        raise ValueError("methods must select at least one of " + str(sorted(available)))
    return [available[k]() for k in _METHOD_ORDER if k in selected]


def _feasible_pareto_indices(all_obj, all_constraints):
    """Feasible Pareto-optimal indices (minimization-normalized)."""
    all_obj = np.asarray(all_obj, dtype=float)
    all_constraints = np.asarray(all_constraints, dtype=bool)
    feasible = np.where(~all_constraints)[0]
    if len(feasible) == 0:
        return []
    hv_obj, _ = normalize_objectives(all_obj[feasible], RobotArmProblem.OBJECTIVE_MIN_MAX)
    mask = is_pareto_efficient(hv_obj, return_mask=True)
    return list(feasible[mask])


def _export_top_designs(method, results, design_space, session, max_designs=5):
    """Export URDF/SRDF, BOM, and a viewer for a method's top Pareto designs.

    Returns {"design_space_plot": path, "designs": [manifest entries]} --
    a caller consumes this directly instead of scraping stdout, unlike the
    original prototype's main.py.
    """
    all_des = results["all_des"]
    all_obj = results["all_obj"]
    all_constraints = results["all_constraints"]

    pareto_idx = _feasible_pareto_indices(all_obj, all_constraints)
    if not pareto_idx:
        log.info("%s: no feasible designs to export.", method.name)
        return {"design_space_plot": None, "designs": []}

    plot_path = export_pareto_plot(
        all_obj, all_constraints,
        objective_labels=list(RobotArmProblem.OBJECTIVE_KEYS),
        out_dir=session.dirs["outputs"],
        name=method.key,
    )

    manifest = []
    for rank, idx in enumerate(pareto_idx[:max_designs]):
        design = design_space.decode(np.asarray(all_des[idx], dtype=float))
        name = f"{method.key}_design_{rank}"

        report = screen_reachability(design, session=session)
        if report:
            design.metrics.update(report)

        session.save_design(design, name=name)  # traceability (re-saved with reachability, if any)
        urdf_path, srdf_path = export_urdf(design, session.dirs["outputs"], name)
        bom_json_path, bom_csv_path = export_bom(design, session.dirs["outputs"], name)
        viewer_path = export_viewer(design, session.dirs["outputs"], name)

        objectives = dict(zip(RobotArmProblem.OBJECTIVE_KEYS, (float(v) for v in all_obj[idx])))
        session.log_intermediate(f"{name}_objectives", objectives)
        log.info("%s: exported %s: %s", method.name, name, objectives)

        manifest.append({
            "name": name,
            "rank": rank,
            "objectives": objectives,
            "reachability": report or None,
            "urdf_path": str(urdf_path),
            "srdf_path": str(srdf_path),
            "bom_json_path": str(bom_json_path),
            "bom_csv_path": str(bom_csv_path),
            "viewer_path": str(viewer_path),
        })

    return {"design_space_plot": str(plot_path), "designs": manifest}


def run(catalog_path=None, max_joints=6, n_ports=7,
        random_evals=None, ga_pop=100, ga_gen=16, seed=None,
        methods=None, output_root="designs"):
    """Run one MICARD design-exploration session end to end.

    Args:
        catalog_path: JSON parts catalog; defaults to the Dynamixel catalog
            bundled with this package.
        max_joints, n_ports: design-space shape (see DesignSpace).
        random_evals: NFE for random search; defaults to ga_pop * ga_gen so
            it's a fair comparison against the GA/PPO budgets.
        ga_pop, ga_gen: GA population/generations; also reused as PPO's
            epochs/mini_batch_size so every selected method sees a
            comparable evaluation budget.
        seed: RNG seed for reproducibility.
        methods: which optimizer(s) to run -- any subset of
            {"random_search", "genetic_algorithm", "ppo"}; None (default)
            runs all three, matching the original prototype's behavior.
            "ppo" additionally requires torch to be installed.
        output_root: where this run's session folder is created (see
            utils.session.Session) -- an embedding caller should point this
            at its own project's storage (e.g. a subfolder of the project's
            shared folder) rather than accepting the "designs" default,
            which is relative to the current process's working directory.
            Each call gets its own (datetime-based) session subfolder, so
            repeated calls against the same output_root never collide or
            overwrite each other's outputs.

    Returns:
        A JSON-serializable dict:
        {
          "session_id": ..., "session_root": ...,
          "methods": {
            "<method_key>": {
              "name", "num_evaluated", "num_feasible", "final_hypervolume",
              "design_space_plot", "designs": [
                {"name", "rank", "objectives", "reachability",
                 "urdf_path", "srdf_path", "bom_json_path", "bom_csv_path",
                 "viewer_path"}, ...
              ],
            }, ...
          },
          "hypervolume_plot": ... or None,
        }
    """
    rng = np.random.default_rng(seed)
    selected = list(methods) if methods is not None else list(_METHOD_ORDER)
    random_evals = ga_pop * ga_gen if random_evals is None else random_evals
    catalog_path = catalog_path or _DEFAULT_CATALOG_PATH

    method_objs = _build_methods(selected, ga_pop, ga_gen, random_evals, rng)

    session = Session(root=output_root)
    log.info("Session: %s (root=%s)", session.session_id, session.root)
    session.log_intermediate("run_config", {
        "catalog_path": str(catalog_path), "max_joints": max_joints,
        "n_ports": n_ports, "random_evals": random_evals,
        "ga_pop": ga_pop, "ga_gen": ga_gen, "seed": seed,
        "methods": selected,
    })

    catalog = load_catalog(catalog_path)
    design_space = DesignSpace(catalog, max_joints=max_joints, n_ports=n_ports)
    problem = RobotArmProblem(design_space)

    storage = {}
    method_summaries = {}
    for method in method_objs:
        log.info("=== Running %s ===", method.name)
        results = method.runner(problem, session=session, **method.kwargs)
        storage[method.key] = results
        n_valid = int(np.sum(~np.asarray(results["all_constraints"], dtype=bool)))
        final_hv = float(results["hypervolumes"][-1]) if results["hypervolumes"] else 0.0
        log.info("%s: %d feasible, final hypervolume %.4f", method.name, n_valid, final_hv)
        method_summaries[method.key] = {
            "name": method.name,
            "num_evaluated": len(results["all_des"]),
            "num_feasible": n_valid,
            "final_hypervolume": final_hv,
        }

    hv_plot_path = export_hypervolume_plot(storage, session.dirs["outputs"]) if storage else None
    if hv_plot_path:
        log.info("Hypervolume comparison: %s", hv_plot_path)

    for method in method_objs:
        export = _export_top_designs(method, storage[method.key], design_space, session)
        method_summaries[method.key].update(export)

    log.info("Done. All artifacts under: %s", session.root)
    return {
        "session_id": session.session_id,
        "session_root": str(session.root),
        "methods": method_summaries,
        "hypervolume_plot": str(hv_plot_path) if hv_plot_path else None,
    }
