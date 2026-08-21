"""MICARD prototype orchestrator.

Runs a traceable design-space-exploration session: load catalog, build the
problem, run optimizers, then export URDF/SRDF, BOM, viewers, and a design-
space plot for the top Pareto designs [MICARD 4.0]. Mirrors the interchangeable
OptimizationMethod / storage pattern of the reference main.py [main 8], and
saves all designs + intermediate results for traceability [MICARD 5.0].
"""
from dataclasses import dataclass
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


# --- Interchangeable optimizer wrapper (like the reference OptimizationMethod) [main 8] ---
@dataclass
class OptimizationMethod:
    name: str
    runner: Callable
    color: str
    kwargs: dict


def initialize_storage(methods):
    """One storage bucket per method [main 8]."""
    return {m.name: {} for m in methods}


def store_results(storage, name, results):
    storage[name] = results


def run_method(method, problem, session):
    """Call an optimizer with a shared problem + session [main 8]."""
    print(f"\n=== Running {method.name} ===")
    return method.runner(problem, session=session, **method.kwargs)


# --- Selecting + exporting the best designs ---
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


def export_top_designs(method_name, results, design_space, session,
                       max_designs=5):
    """Export URDF/SRDF, BOM, and a viewer for the top Pareto designs [MICARD 4.0]."""
    all_des = results["all_des"]
    all_obj = results["all_obj"]
    all_constraints = results["all_constraints"]

    pareto_idx = _feasible_pareto_indices(all_obj, all_constraints)
    if not pareto_idx:
        print(f"{method_name}: no feasible designs to export.")
        return

    # Design-space plot for this method (graphical representation) [MICARD 3.0.6][main 8].
    export_pareto_plot(
        all_obj, all_constraints,
        objective_labels=list(RobotArmProblem.OBJECTIVE_KEYS),
        out_dir=session.dirs["outputs"],
        name=f"{method_name.replace(' ', '_').lower()}",
    )

    safe = method_name.replace(" ", "_").lower()
    for rank, idx in enumerate(pareto_idx[:max_designs]):
        design = design_space.decode(np.asarray(all_des[idx], dtype=float))
        name = f"{safe}_design_{rank}"

        report = screen_reachability(design, session=session)
        if report:
            design.metrics.update(report)
            session.save_design(design, name=name)  # re-save with reachability
            print(f"  {name}: {report.get('reach_targets_reachable', 0)}"
                  f"/{report.get('reach_targets_total', 0)} targets reachable")

        session.save_design(design, name=name)              # traceability [5.0.1]
        export_urdf(design, session.dirs["outputs"], name)  # URDF + SRDF [4.0.1]
        export_bom(design, session.dirs["outputs"], name)   # BOM [4.0.3]
        export_viewer(design, session.dirs["outputs"], name)  # 3D render [4.0.2][robot_chat 11]

        obj = dict(zip(RobotArmProblem.OBJECTIVE_KEYS, all_obj[idx]))
        session.log_intermediate(f"{name}_objectives", obj)  # intermediate result [5.0.4]
        print(f"  exported {name}: {obj}")


def main(catalog_path="utils/dynamixel_single_axis.json",
         max_joints=6, n_ports=7,
         random_evals=None, ga_pop=100, ga_gen=16,
         seed=None):
    """Run one MICARD design-exploration session end to end."""
    rng = np.random.default_rng(seed)

    random_evals = ga_pop * ga_gen if random_evals is None else random_evals

    # 1. Session with datestring ID + traceability folders [MICARD 5.0].
    session = Session()
    print(f"Session: {session.session_id}")
    session.log_intermediate("run_config", {
        "catalog_path": catalog_path, "max_joints": max_joints,
        "n_ports": n_ports, "random_evals": random_evals,
        "ga_pop": ga_pop, "ga_gen": ga_gen, "seed": seed,
    })

    # 2. Load catalog generically; drop entries missing torque/weight/cost [MICARD 2.0][1].
    catalog = load_catalog(catalog_path)

    # 3. Build design space + problem. 7 ports match the platform scenario [MICARD scenario].
    design_space = DesignSpace(catalog, max_joints=max_joints, n_ports=n_ports)
    problem = RobotArmProblem(design_space)

    # 4. Interchangeable optimizers [main 8]. GA is custom (no pygad/pymoo) [3][user].
    methods = [
        OptimizationMethod("Random Search", run_random_search, "blue",
                        {"num_exec": random_evals, "rng": rng}),
        OptimizationMethod("Genetic Algorithm", run_genetic_algorithm, "green",
                        {"pop_size": ga_pop, "n_gen": ga_gen, "rng": rng}),
        OptimizationMethod("PPO", run_ppo_optimization, "red",
                        {"epochs": ga_pop, "mini_batch_size": ga_gen, "rng": rng}),
    ]

    storage = initialize_storage(methods)
    for method in methods:
        results = run_method(method, problem, session)
        store_results(storage, method.name, results)
        n_valid = int(np.sum(~np.asarray(results["all_constraints"], dtype=bool)))
        final_hv = results["hypervolumes"][-1] if results["hypervolumes"] else 0.0
        print(f"{method.name}: {n_valid} feasible, final hypervolume {final_hv:.4f}")

    hv_path = export_hypervolume_plot(storage, session.dirs["outputs"])
    print(f"Hypervolume comparison: {hv_path}")

    # 5. Export deliverables for the top designs from each method [MICARD 4.0].
    for method in methods:
        export_top_designs(method.name, storage[method.name],
                           design_space, session)

    print(f"\nDone. All artifacts under: {session.root}")
    return session, storage


if __name__ == "__main__":
    main()