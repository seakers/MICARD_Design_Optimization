"""RobotArmProblem: evaluation entry point mirroring the Mission_Problem API.

Provides num_objectives, objective_min_max, evaluate(), and the
filter_des_space / choose_components hooks the optimizers expect [3].

Objectives and constraints are driven by a ProblemConfig (optionally loaded
from JSON). Defaults reproduce the original mission setup: objectives =
manipulability (max), cost (min), weight (min) [9]; constraints = mass, yield,
deflection, reach, dof. The original single 'structural' constraint is split
into two uniform bound constraints ('yield' + 'deflection'), so every
constraint is a declarative bound on a produced metric. All registered metrics
are still computed and stored [1].

Constraint values follow a GRADED convention: 0.0 when satisfied, positive and
proportional to the violation otherwise, giving optimizers a feasibility
gradient rather than a flat penalty [2].
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from evaluation.base import compute_all_metrics, available_metrics  # noqa: F401


# --- Default spec: reproduces the original hard-coded behavior exactly -------

_DEFAULT_OBJECTIVES = [
    {"metric": "manipulability", "direction": "max"},
    {"metric": "cost", "direction": "min"},
    {"metric": "weight", "direction": "min"},
]

# 'structural' is decomposed into 'yield' + 'deflection' (Option A). yield is a
# lower_bound on min_yield_margin at 0 (margin < 0 => stress over allowable);
# deflection is an upper_bound on max_deflection_ratio at the 1% limit used in
# structural.py [12]. mass/reach carry the original 50 kg / 2 m budgets [2].
_DEFAULT_CONSTRAINTS = [
    {"name": "mass", "type": "upper_bound", "metric": "weight",
     "limit": 50.0, "scale": 50.0, "default": float("inf"), "clip": [0.0, 1.0]},
    {"name": "yield", "type": "lower_bound", "metric": "min_yield_margin",
     "limit": 0.0, "scale": 1.0, "default": 1.0, "clip": [0.0, 1.0]},
    {"name": "deflection", "type": "upper_bound", "metric": "max_deflection_ratio",
     "limit": 0.01, "scale": 1.0, "default": 0.0, "clip": [0.0, 1.0]},
    {"name": "reach", "type": "lower_bound", "metric": "abs_reach_m",
     "limit": 2.0, "scale": 2.0, "default": 0.0, "clip": [0.0, 1.0]},
    {"name": "dof", "type": "lower_bound", "metric": "dof",
     "limit": 1.0, "scale": 1.0, "default": 0.0, "graded": False},
]

_VALID_TYPES = {"upper_bound", "lower_bound"}
_VALID_DIRECTIONS = {"min", "max"}

# Keys read directly off the Design object rather than the metrics dict.
# 'dof' is a design attribute, not a registered metric output key.
_DESIGN_ATTRS = {"dof"}


@dataclass
class ProblemConfig:
    """External spec for what to optimize and what counts as feasible.

    The larger tool can build this directly or load it from JSON via
    ProblemConfig.from_json(path_or_dict_or_jsonstring).
    """
    objectives: list = field(
        default_factory=lambda: [dict(o) for o in _DEFAULT_OBJECTIVES])
    constraints: list = field(
        default_factory=lambda: [dict(c) for c in _DEFAULT_CONSTRAINTS])
    feasibility_tol: float = 1e-9

    @classmethod
    def from_json(cls, source):
        """Build from a ProblemConfig, a dict, a JSON file path, or a JSON string."""
        if isinstance(source, ProblemConfig):
            return source
        if isinstance(source, dict):
            data = source
        else:
            text = str(source)
            # Only try the filesystem for short, path-like strings. A JSON blob is
            # long and/or starts with '{'/'[', and would otherwise overflow the
            # OS filename limit inside Path.exists() (OSError: File name too long).
            looks_like_path = (
                len(text) < 260
                and "\n" not in text
                and not text.lstrip().startswith(("{", "["))
            )
            if looks_like_path:
                try:
                    p = Path(text)
                    if p.exists():
                        text = p.read_text(encoding="utf-8")
                except OSError:
                    pass  # not a usable path; fall through to JSON parsing
            data = json.loads(text)
        return cls(
            objectives=data.get("objectives", [dict(o) for o in _DEFAULT_OBJECTIVES]),
            constraints=data.get("constraints", [dict(c) for c in _DEFAULT_CONSTRAINTS]),
            feasibility_tol=float(data.get("feasibility_tol", 1e-9)),
        )

    def as_dict(self):
        """JSON-serializable form, for logging into run_config [1]."""
        return {
            "objectives": self.objectives,
            "constraints": self.constraints,
            "feasibility_tol": self.feasibility_tol,
        }


class RobotArmProblem:
    # Kept as fallback class defaults so any legacy reference still resolves.
    OBJECTIVE_KEYS = ("manipulability", "cost", "weight")
    OBJECTIVE_MIN_MAX = ("max", "min", "min")
    # Default constraint vector order (for logging/plots). Now split so that
    # 'structural' -> 'yield' + 'deflection' under the uniform bound scheme.
    CONSTRAINT_KEYS = ("mass", "yield", "deflection", "reach", "dof")

    def __init__(self, design_space, config=None, feasibility_tol=None):
        self.design_space = design_space.build_gene_space()
        self._ds = design_space

        config = ProblemConfig.from_json(config) if config is not None else ProblemConfig()
        self.config = config
        self.feasibility_tol = (feasibility_tol if feasibility_tol is not None
                                else config.feasibility_tol)

        self._validate(config)

        # Instance-level objectives (prefer config over the class defaults).
        self.objective_keys = tuple(o["metric"] for o in config.objectives)
        self.objective_min_max = [o["direction"] for o in config.objectives]
        self.num_objectives = len(self.objective_keys)

        # Instance-level constraints.
        self._constraints = config.constraints
        self.constraint_keys = tuple(c["name"] for c in config.constraints)

        # Metric-key validation is deferred to the first evaluate() call, since
        # it must check against produced dict keys (e.g. abs_reach_m), not the
        # registry names available_metrics() returns (e.g. reachability) [3][4].
        self._keys_validated = False

    def _validate(self, config):
        """Fast-fail on structural config errors (directions/types/duplicates).

        Metric *name* validation happens lazily in evaluate() — see
        _validate_metric_keys — because available_metrics() lists registry
        names, not the dict keys metric functions emit [3][4][12].
        """
        if not config.objectives:
            raise RuntimeError("ProblemConfig must define at least one objective.")

        for o in config.objectives:
            if "metric" not in o:
                raise RuntimeError(f"Objective missing 'metric': {o!r}.")
            if o.get("direction") not in _VALID_DIRECTIONS:
                raise RuntimeError(
                    f"Objective {o.get('metric')!r} has invalid direction "
                    f"{o.get('direction')!r}; expected one of {_VALID_DIRECTIONS}.")

        seen = set()
        for c in config.constraints:
            if "metric" not in c:
                raise RuntimeError(f"Constraint missing 'metric': {c!r}.")
            if c.get("type") not in _VALID_TYPES:
                raise RuntimeError(
                    f"Constraint {c.get('name')!r} has invalid type "
                    f"{c.get('type')!r}; expected one of {_VALID_TYPES}.")
            name = c.get("name")
            if name in seen:
                raise RuntimeError(f"Duplicate constraint name: {name!r}.")
            seen.add(name)

    def _validate_metric_keys(self, metrics):
        """One-time check that every referenced key is produceable.

        Valid keys = whatever the metrics dict contains this run, plus known
        Design attributes like 'dof'. Runs once on the first real design, which
        is a normal multi-joint design (so conditional keys are present) [4][12].
        """
        produceable = set(metrics) | _DESIGN_ATTRS
        referenced = ({o["metric"] for o in self.config.objectives}
                      | {c["metric"] for c in self._constraints})
        missing = referenced - produceable
        if missing:
            raise RuntimeError(
                f"Config references metric keys that no metric produces: "
                f"{sorted(missing)}. Available keys this run: {sorted(produceable)}.")
        self._keys_validated = True

    def filter_des_space(self, var, current_design_context):
        return self.design_space

    def choose_components(self, var, current_design_context):
        return

    def _metric_value(self, spec, design, metrics):
        """Read a referenced value from either the Design object or metrics dict."""
        if spec["metric"] in _DESIGN_ATTRS:
            return float(getattr(design, spec["metric"]))
        return metrics.get(spec["metric"], spec.get("default", 0.0))

    def _eval_constraint(self, spec, design, metrics):
        """One graded violation value from a declarative bound spec [2].

        upper_bound: violated when value > limit  -> (value - limit) / scale
        lower_bound: violated when value < limit  -> (limit - value) / scale
        'graded': false collapses any shortfall to a flat 1.0 flag (e.g. dof).
        """
        value = self._metric_value(spec, design, metrics)
        limit = spec.get("limit", 0.0)
        scale = max(spec.get("scale", 1.0), self.feasibility_tol)

        if spec["type"] == "upper_bound":
            raw = max(0.0, value - limit) / scale
        else:  # lower_bound
            raw = max(0.0, limit - value) / scale

        if spec.get("graded", True) is False:
            return 0.0 if raw <= 0.0 else 1.0

        lo, hi = spec.get("clip", [0.0, 1.0])
        return float(np.clip(raw, lo, hi))

    def _constraint_vals(self, design, metrics):
        """Graded violations: 0.0 = satisfied, positive = distance to feasible [2]."""
        return np.array(
            [self._eval_constraint(c, design, metrics) for c in self._constraints],
            dtype=float,
        )

    def evaluate(self, genes):
        """Return (objectives, is_constrained, constraint_vals, metrics).

        Matches the contract used across the optimizers [2][3].
        'is_constrained' is True when the design is INFEASIBLE.
        """
        design = self._ds.decode(np.asarray(genes, dtype=float))
        metrics = compute_all_metrics(design)

        if not self._keys_validated:
            self._validate_metric_keys(metrics)

        objectives = np.array([metrics.get(k, 0.0) for k in self.objective_keys],
                              dtype=float)

        constraint_vals = self._constraint_vals(design, metrics)
        total_violation = float(constraint_vals.sum())
        is_constrained = total_violation > self.feasibility_tol

        # Store the total violation so consumers (PPO reward, GA sort) can read it.
        metrics["total_violation"] = total_violation

        self.last_design = design
        return objectives, is_constrained, constraint_vals, metrics