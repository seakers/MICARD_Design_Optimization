"""Random search over the robot-arm design space.

Structure mirrors the reference random_search.py (design-space walk with
filter_des_space / choose_components hooks, then Pareto progress) [random_search 5],
but uses the custom hypervolume in utils.pareto instead of pymoo [random_search 5].
"""
import numpy as np

from utils.pareto import pareto_progress


def run_random_search(problem, num_exec=2000, rng=None, session=None, max_values=None):
    """Sample num_exec designs, evaluating each via problem.evaluate() [random_search 5].

    Args:
        problem: a RobotArmProblem (design_space, evaluate, hooks) [problem.py].
        num_exec: number of design evaluations (NFE).
        rng: optional np.random.Generator for reproducibility.
        session: optional Session for traceability logging [MICARD 5.0].
    """
    rng = rng or np.random.default_rng()
    num_objectives = problem.num_objectives
    objective_min_max = problem.objective_min_max

    all_des, all_obj = [], []
    all_constraints, all_constraint_vals = [], []
    all_metrics = []

    for run in range(num_exec):
        # Walk the (possibly dynamic) design space [random_search 5].
        des_space = problem.design_space
        design, ind = [], 0
        current_context = []
        while ind < len(des_space):
            if "filter_des_space" in des_space[ind]:
                des_space = problem.filter_des_space(des_space[ind], design)
            var = des_space[ind]
            if var["type"] == "continuous":
                design.append(rng.uniform(var["range"][0], var["range"][1]))
            elif var["type"] == "discrete":
                design.append(rng.choice(np.array(var["range"])))
            else:
                raise ValueError("INVALID DESIGN SPACE")
            current_context.append(design[-1])
            if "select_components" in des_space[ind]:
                problem.choose_components(des_space[ind], current_context)
            ind += 1

        objectives, is_constrained, constraint_vals, metrics = problem.evaluate(design)
        all_des.append(design)
        all_obj.append(objectives)
        all_constraints.append(is_constrained)
        all_constraint_vals.append(constraint_vals)
        all_metrics.append(metrics)

        if session and not is_constrained:
            session.save_design(problem.last_design, algorithm="random_search")

        if run % max(1, num_exec // 10) == 0:
            print(f"Random search {run + 1}/{num_exec}")

    all_obj = np.array(all_obj)
    all_constraints = np.array(all_constraints)

    max_values = np.max(all_obj, axis=0) * 1.1 + 1e-6  # avoid div-by-zero [random_search 5]

    # Custom Pareto + hypervolume (replaces pymoo HV) [random_search 5][utils.pareto].
    pareto_front_obj, hypervolumes = pareto_progress(
        all_obj, all_constraints, objective_min_max, max_values=max_values
    )

    n_valid = int(np.sum(all_constraints == False))
    print(f"Random search: {n_valid} valid designs of {num_exec}.")

    return {
        "all_des": all_des,
        "all_obj": all_obj,
        "all_constraints": all_constraints,
        "all_constraint_vals": all_constraint_vals,
        "all_metrics": all_metrics,
        "pareto_front_obj": pareto_front_obj,
        "hypervolumes": hypervolumes,
        "num_objectives": num_objectives,
        "max_values": max_values,
    }