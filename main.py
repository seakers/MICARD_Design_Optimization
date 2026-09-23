"""MICARD prototype orchestrator.

Runs a traceable design-space-exploration session: load catalog, build the
problem, run optimizers, then export URDF/SRDF, BOM, viewers, and a design-
space plot for the top Pareto designs [MICARD 4.0]. Mirrors the interchangeable
OptimizationMethod / storage pattern of the reference main.py [main 8], and
saves all designs + intermediate results for traceability [MICARD 5.0].
"""
import argparse
import contextlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from utils.catalog import load_catalog
from utils.design_space import DesignSpace
from utils.session import Session
from evaluation.problem import RobotArmProblem
from evaluation.reachability_screen import screen_reachability
from optimization.random_search import run_random_search
from optimization.genetic_algorithm import run_genetic_algorithm
from optimization.ppo_optimizer import run_ppo_optimization
from outputs.urdf_export import export_urdf
from outputs.bom import export_bom
from outputs.visualization import export_viewer, export_pareto_plot, export_hypervolume_plot

_DEFAULT_CATALOG_PATH = Path(__file__).resolve().parent / "utils" / "dynamixel_single_axis.json"


@dataclass
class OptimizationMethod:
    key: str
    name: str
    runner: Callable
    kwargs: dict


def _feasible_pareto_indices(all_obj, all_constraints):
    """Feasible Pareto-optimal indices (minimization-normalized) [utils.pareto]."""
    from utils.pareto import is_pareto_efficient, normalize_objectives
    all_obj = np.asarray(all_obj, dtype=float)
    all_constraints = np.asarray(all_constraints, dtype=bool)
    feasible = np.where(~all_constraints)[0]
    if len(feasible) == 0:
        return []
    hv_obj, _ = normalize_objectives(all_obj[feasible],
                                     RobotArmProblem.OBJECTIVE_MIN_MAX)
    mask = is_pareto_efficient(hv_obj, return_mask=True)
    return list(feasible[mask])


def export_top_designs(method, results, design_space, session,
                       max_designs=5):
    """Export URDF/SRDF, BOM, and a viewer for a method's top Pareto designs."""
    all_des = results["all_des"]
    all_obj = results["all_obj"]
    all_constraints = results["all_constraints"]

    pareto_idx = _feasible_pareto_indices(all_obj, all_constraints)
    if not pareto_idx:
        print(f"{method.name}: no feasible designs to export.")
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
        print(f"  exported {name}: {objectives}")

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
        random_search=True, genetic_algorithm=True, ppo=True,
        output_root="designs"):
    """Run one MICARD design-exploration session end to end.

    random_search/genetic_algorithm/ppo each enable or disable that method
    (default: all three run). Returns a JSON-serializable dict: session_id,
    session_root, per-method summaries (each with its exported designs), and
    hypervolume_plot.
    """
    rng = np.random.default_rng(seed)
    random_evals = ga_pop * ga_gen if random_evals is None else random_evals
    catalog_path = catalog_path or _DEFAULT_CATALOG_PATH

    session = Session(root=output_root)
    print(f"Session: {session.session_id} (root={session.root})")
    session.log_intermediate("run_config", {
        "catalog_path": str(catalog_path), "max_joints": max_joints,
        "n_ports": n_ports, "random_evals": random_evals,
        "ga_pop": ga_pop, "ga_gen": ga_gen, "seed": seed,
        "random_search": random_search, "genetic_algorithm": genetic_algorithm, "ppo": ppo,
    })

    # 2. Load catalog generically; drop entries missing torque/weight/cost
    catalog = load_catalog(catalog_path)

    # 3. Build design space + problem. 7 ports match the platform scenario
    design_space = DesignSpace(catalog, max_joints=max_joints, n_ports=n_ports)
    problem = RobotArmProblem(design_space)
    random_evals = ga_pop * ga_gen if random_evals is None else random_evals

    #4. Run optimizations
    method_objs = []
    if random_search:
        method_objs.append(OptimizationMethod("random_search", "Random Search", run_random_search,
                                               {"num_exec": random_evals, "rng": rng}))
    if genetic_algorithm:
        method_objs.append(OptimizationMethod("genetic_algorithm", "Genetic Algorithm", run_genetic_algorithm,
                                               {"pop_size": ga_pop, "n_gen": ga_gen, "rng": rng}))
    if ppo:
        method_objs.append(OptimizationMethod("ppo", "PPO", run_ppo_optimization,
                                               {"epochs": ga_pop, "mini_batch_size": ga_gen, "rng": rng}))
    if not method_objs:
        raise ValueError("at least one of random_search/genetic_algorithm/ppo must be enabled")

    storage = {}
    method_summaries = {}
    for method in method_objs:
        print(f"\n=== Running {method.name} ===")
        results = method.runner(problem, session=session, **method.kwargs)
        storage[method.key] = results
        n_valid = int(np.sum(~np.asarray(results["all_constraints"], dtype=bool)))
        final_hv = float(results["hypervolumes"][-1]) if results["hypervolumes"] else 0.0
        print(f"{method.name}: {n_valid} feasible, final hypervolume {final_hv:.4f}")
        method_summaries[method.key] = {
            "name": method.name,
            "num_evaluated": len(results["all_des"]),
            "num_feasible": n_valid,
            "final_hypervolume": final_hv,
        }

    hv_plot_path = export_hypervolume_plot(storage, session.dirs["outputs"])
    print(f"Hypervolume comparison: {hv_plot_path}")

    # 5. Export deliverables for the top designs from each method
    for method in method_objs:
        export = export_top_designs(method, storage[method.key], design_space, session)
        method_summaries[method.key].update(export)

    print(f"\nDone. All artifacts under: {session.root}")
    return {
        "session_id": session.session_id,
        "session_root": str(session.root),
        "methods": method_summaries,
        "hypervolume_plot": str(hv_plot_path) if hv_plot_path else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-path", default=None,
                        help="JSON parts catalog (default: the bundled Dynamixel catalog)")
    parser.add_argument("--max-joints", type=int, default=6)
    parser.add_argument("--n-ports", type=int, default=7)
    parser.add_argument("--random-evals", type=int, default=None,
                        help="default: ga_pop * ga_gen")
    parser.add_argument("--ga-pop", type=int, default=100)
    parser.add_argument("--ga-gen", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-random-search", dest="random_search", action="store_false",
                        help="skip random search (default: run it)")
    parser.add_argument("--no-genetic-algorithm", dest="genetic_algorithm", action="store_false",
                        help="skip the genetic algorithm (default: run it)")
    parser.add_argument("--no-ppo", dest="ppo", action="store_false",
                        help="skip PPO (default: run it)")
    parser.add_argument("--output-root", default="designs",
                        help="where the session folder is created (default: ./designs)")
    parser.add_argument("--json", action="store_true",
                        help="print run()'s result as one JSON line on stdout instead of "
                             "a human-readable summary")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    kwargs = vars(args)
    json_mode = kwargs.pop("json")

    if json_mode:
        # send run()'s progress print()s to stderr so stdout is just the JSON line below
        with contextlib.redirect_stdout(sys.stderr):
            result = run(**kwargs)
        print(json.dumps(result))
        return

    result = run(**kwargs)
    print(f"\nSession {result['session_id']}: {result['session_root']}")


if __name__ == "__main__":
    main()
