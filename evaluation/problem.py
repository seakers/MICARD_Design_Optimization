"""RobotArmProblem: evaluation entry point mirroring the Mission_Problem API.

Provides num_objectives, objective_min_max, evaluate(), and the
filter_des_space / choose_components hooks the optimizers expect [3][5][2].
Optimization objectives: manipulability (max), cost (min), weight (min) [9];
all other registered metrics are still computed and stored.

Constraint values follow a GRADED convention: 0.0 when satisfied, positive and
proportional to the violation otherwise, giving optimizers a feasibility
gradient rather than a flat penalty [4].
"""
import numpy as np

from evaluation.base import compute_all_metrics, available_metrics


class RobotArmProblem:
    OBJECTIVE_KEYS = ("manipulability", "cost", "weight")
    OBJECTIVE_MIN_MAX = ("max", "min", "min")

    # Names line up with the constraint_vals vector order below (for logging/plots).
    CONSTRAINT_KEYS = ("mass", "structural", "reach", "dof")

    def __init__(self, design_space, max_reach_m=2.0, max_mass_kg=50.0,
                 required_reach_m=2.0, feasibility_tol=1e-9):
        self.design_space = design_space.build_gene_space()
        self._ds = design_space
        self.num_objectives = len(self.OBJECTIVE_KEYS)
        self.objective_min_max = list(self.OBJECTIVE_MIN_MAX)
        self.max_reach_m = max_reach_m
        self.max_mass_kg = max_mass_kg           # launch-mass budget [9]
        self.required_reach_m = required_reach_m # ORU ~2 m from platform [9]
        self.feasibility_tol = feasibility_tol

        # Guard against the silent under-registration bug we fixed earlier.
        missing = set(self.OBJECTIVE_KEYS) - set(available_metrics())
        if missing:
            raise RuntimeError(
                f"Metrics not registered: {missing}. "
                f"Ensure evaluation/__init__.py imports every metric module.")

    def filter_des_space(self, var, current_design_context):
        return self.design_space

    def choose_components(self, var, current_design_context):
        return

    def _constraint_vals(self, design, metrics):
        """Graded violations: 0.0 = satisfied, positive = distance to feasible [4]."""
        # Normalize each violation to [0, 1].
        # Mass: normalized excess over the launch budget.
        mass_viol = np.clip(
            max(0.0, metrics.get("weight", np.inf) - self.max_mass_kg)
            / max(self.max_mass_kg, self.feasibility_tol),
            0.0, 1.0,
        )

        # Structural: summed normalized overrun of deflection + bending stress.
        # min_yield_margin < 0 means stress exceeded the allowable; deflection
        # overrun uses the worst link ratio vs. the 1% limit used in structural.py.
        yield_overrun = max(0.0, -metrics.get("min_yield_margin", 1.0))
        defl_overrun = max(0.0, metrics.get("max_deflection_ratio", 0.0) - 0.01)
        struct_viol = np.clip(yield_overrun + defl_overrun, 0.0, 1.0)

        # Reach: normalized distance short of the required ORU-retrieval reach [9].
        reach_viol = np.clip(
            max(0.0, self.required_reach_m - metrics.get("abs_reach_m", 0.0))
            / max(self.required_reach_m, self.feasibility_tol),
            0.0, 1.0,
        )

        # DOF: a design with no joints has no meaningful gradient; flat flag.
        dof_viol = 0.0 if design.dof > 0 else 1.0

        return np.array([mass_viol, struct_viol, reach_viol, dof_viol], dtype=float)

    def evaluate(self, genes):
        """Return (objectives, is_constrained, constraint_vals).

        Matches the 3-value contract used across the optimizers [2][3].
        'is_constrained' is True when the design is INFEASIBLE [3][5].
        """
        design = self._ds.decode(np.asarray(genes, dtype=float))
        metrics = compute_all_metrics(design)

        objectives = np.array([metrics.get(k, 0.0) for k in self.OBJECTIVE_KEYS],
                              dtype=float)

        constraint_vals = self._constraint_vals(design, metrics)
        total_violation = float(constraint_vals.sum())
        is_constrained = total_violation > self.feasibility_tol

        # Store the total violation so consumers (PPO reward, GA sort) can read it.
        metrics["total_violation"] = total_violation

        self.last_design = design
        return objectives, is_constrained, constraint_vals