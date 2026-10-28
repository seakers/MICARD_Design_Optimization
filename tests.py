"""Tests for the configurable ProblemConfig / RobotArmProblem plumbing.

Run with:  pytest test_problem_config.py -v

These tests avoid the real catalog and micard-metrics service by stubbing the
design space and metric computation, so they exercise the config logic in
isolation: JSON round-trip, validation, default parity, and the Option A
constraint split (structural -> yield + deflection).
"""
import json

import numpy as np
import pytest

from evaluation.problem import (
    RobotArmProblem,
    ProblemConfig,
    _DEFAULT_OBJECTIVES,
    _DEFAULT_CONSTRAINTS,
)


# --- Test doubles ------------------------------------------------------------

class FakeDesign:
    """Minimal stand-in for a decoded Design."""
    def __init__(self, dof=3):
        self.dof = dof
        self.metrics = {}


class FakeDesignSpace:
    """Decodes genes into a FakeDesign with a controllable set of metrics.

    `metrics` is what compute_all_metrics will be monkeypatched to return, so
    tests can drive feasibility directly without the real evaluators.
    """
    def __init__(self, dof=3):
        self._dof = dof

    def build_gene_space(self):
        return [{"low": 0.0, "high": 1.0}]

    def decode(self, genes):
        return FakeDesign(dof=self._dof)


def make_problem(monkeypatch, metrics, config=None, dof=3):
    """Build a RobotArmProblem whose evaluate() sees `metrics`."""
    import evaluation.problem as problem_mod
    monkeypatch.setattr(problem_mod, "compute_all_metrics",
                        lambda design: dict(metrics))
    ds = FakeDesignSpace(dof=dof)
    return RobotArmProblem(ds, config=config)


# A realistic, fully-feasible metrics dict (matches the default objective/
# constraint keys). Weight < 50, reach >= 2, margins fine, deflection < 1%.
FEASIBLE_METRICS = {
    "manipulability": 0.42,
    "cost": 1200.0,
    "weight": 30.0,
    "min_yield_margin": 0.5,
    "max_deflection_ratio": 0.001,
    "abs_reach_m": 2.5,
}


# --- Config object -----------------------------------------------------------

def test_default_config_matches_module_defaults():
    cfg = ProblemConfig()
    assert cfg.objectives == _DEFAULT_OBJECTIVES
    assert cfg.constraints == _DEFAULT_CONSTRAINTS
    assert cfg.feasibility_tol == 1e-9


def test_json_round_trip_dict():
    cfg = ProblemConfig()
    restored = ProblemConfig.from_json(cfg.as_dict())
    assert restored.objectives == cfg.objectives
    assert restored.constraints == cfg.constraints
    assert restored.feasibility_tol == cfg.feasibility_tol


def test_json_round_trip_string():
    cfg = ProblemConfig()
    restored = ProblemConfig.from_json(json.dumps(cfg.as_dict()))
    assert restored.as_dict() == cfg.as_dict()


def test_from_json_file(tmp_path):
    cfg = ProblemConfig()
    path = tmp_path / "problem_config.json"
    path.write_text(json.dumps(cfg.as_dict()), encoding="utf-8")
    restored = ProblemConfig.from_json(str(path))
    assert restored.as_dict() == cfg.as_dict()


def test_from_json_passthrough_of_config_object():
    cfg = ProblemConfig()
    assert ProblemConfig.from_json(cfg) is cfg


def test_partial_json_fills_defaults():
    # Only objectives supplied -> constraints fall back to defaults.
    cfg = ProblemConfig.from_json({"objectives": [{"metric": "cost", "direction": "min"}]})
    assert cfg.constraints == _DEFAULT_CONSTRAINTS
    assert cfg.objectives == [{"metric": "cost", "direction": "min"}]


# --- Construction-time validation --------------------------------------------

def test_empty_objectives_rejected():
    with pytest.raises(RuntimeError, match="at least one objective"):
        RobotArmProblem(FakeDesignSpace(), config={"objectives": []})


def test_bad_direction_rejected():
    bad = {"objectives": [{"metric": "cost", "direction": "sideways"}]}
    with pytest.raises(RuntimeError, match="invalid direction"):
        RobotArmProblem(FakeDesignSpace(), config=bad)


def test_bad_constraint_type_rejected():
    bad = {
        "objectives": [{"metric": "cost", "direction": "min"}],
        "constraints": [{"name": "x", "type": "equals", "metric": "weight"}],
    }
    with pytest.raises(RuntimeError, match="invalid type"):
        RobotArmProblem(FakeDesignSpace(), config=bad)


def test_duplicate_constraint_name_rejected():
    bad = {
        "objectives": [{"metric": "cost", "direction": "min"}],
        "constraints": [
            {"name": "dup", "type": "upper_bound", "metric": "weight", "limit": 1},
            {"name": "dup", "type": "lower_bound", "metric": "weight", "limit": 1},
        ],
    }
    with pytest.raises(RuntimeError, match="Duplicate constraint name"):
        RobotArmProblem(FakeDesignSpace(), config=bad)


# --- Lazy metric-key validation (Option 1) -----------------------------------

def test_unknown_metric_key_caught_on_first_evaluate(monkeypatch):
    bad = {"objectives": [{"metric": "not_a_real_metric", "direction": "max"}]}
    # Construction succeeds (fast-fail only checks direction/type/dupes)...
    problem = make_problem(monkeypatch, FEASIBLE_METRICS, config=bad)
    # ...but the first evaluate() flags the unproduced key.
    with pytest.raises(RuntimeError, match="no metric produces"):
        problem.evaluate([0.5])


def test_dof_attribute_is_allowed_metric_key(monkeypatch):
    # 'dof' is read off the design, not from the metrics dict, and must validate.
    problem = make_problem(monkeypatch, FEASIBLE_METRICS)
    objectives, is_constrained, cons, metrics = problem.evaluate([0.5])
    assert problem._keys_validated is True


# --- Objective wiring --------------------------------------------------------

def test_instance_objectives_from_config(monkeypatch):
    cfg = {
        "objectives": [
            {"metric": "abs_reach_m", "direction": "max"},
            {"metric": "weight", "direction": "min"},
        ],
        "constraints": [],
    }
    problem = make_problem(monkeypatch, FEASIBLE_METRICS, config=cfg)
    assert problem.objective_keys == ("abs_reach_m", "weight")
    assert problem.objective_min_max == ["max", "min"]
    assert problem.num_objectives == 2
    objectives, *_ = problem.evaluate([0.5])
    assert objectives[0] == pytest.approx(2.5)   # abs_reach_m
    assert objectives[1] == pytest.approx(30.0)  # weight


# --- Default parity + Option A constraint split ------------------------------

def test_default_constraint_keys_split_structural():
    problem = RobotArmProblem(FakeDesignSpace())
    assert problem.constraint_keys == ("mass", "yield", "deflection", "reach", "dof")


def test_feasible_design_has_no_violation(monkeypatch):
    problem = make_problem(monkeypatch, FEASIBLE_METRICS)
    objectives, is_constrained, cons, metrics = problem.evaluate([0.5])
    assert is_constrained is False
    assert cons.sum() == pytest.approx(0.0)
    assert metrics["total_violation"] == pytest.approx(0.0)


def test_mass_violation_is_graded(monkeypatch):
    metrics = dict(FEASIBLE_METRICS, weight=75.0)  # 25 kg over the 50 kg budget
    problem = make_problem(monkeypatch, metrics)
    _, is_constrained, cons, _ = problem.evaluate([0.5])
    mass_idx = problem.constraint_keys.index("mass")
    assert is_constrained is True
    assert cons[mass_idx] == pytest.approx(25.0 / 50.0)  # 0.5


def test_yield_and_deflection_are_independent_entries(monkeypatch):
    # Violate BOTH structural sub-constraints at once.
    metrics = dict(FEASIBLE_METRICS, min_yield_margin=-0.3, max_deflection_ratio=0.02)
    problem = make_problem(monkeypatch, metrics)
    _, is_constrained, cons, _ = problem.evaluate([0.5])
    yi = problem.constraint_keys.index("yield")
    di = problem.constraint_keys.index("deflection")
    assert cons[yi] == pytest.approx(0.3)          # -(-0.3) / 1.0
    assert cons[di] == pytest.approx(0.01)         # (0.02 - 0.01) / 1.0
    # Option A: two independent entries, so the pair can exceed the old cap of 1.0.
    assert cons[yi] + cons[di] == pytest.approx(0.31)
    assert is_constrained is True


def test_dof_is_flat_flag(monkeypatch):
    # dof == 0 -> full flag of 1.0, no gradient.
    problem = make_problem(monkeypatch, FEASIBLE_METRICS, dof=0)
    _, is_constrained, cons, _ = problem.evaluate([0.5])
    di = problem.constraint_keys.index("dof")
    assert cons[di] == pytest.approx(1.0)
    assert is_constrained is True


def test_reach_shortfall_is_graded(monkeypatch):
    metrics = dict(FEASIBLE_METRICS, abs_reach_m=1.0)  # 1 m short of 2 m
    problem = make_problem(monkeypatch, metrics)
    _, _, cons, _ = problem.evaluate([0.5])
    ri = problem.constraint_keys.index("reach")
    assert cons[ri] == pytest.approx(1.0 / 2.0)  # 0.5


def test_dropping_a_constraint(monkeypatch):
    # A caller can remove deflection entirely; it should no longer appear.
    cfg = ProblemConfig().as_dict()
    cfg["constraints"] = [c for c in cfg["constraints"] if c["name"] != "deflection"]
    metrics = dict(FEASIBLE_METRICS, max_deflection_ratio=0.9)  # would violate if present
    problem = make_problem(monkeypatch, metrics, config=cfg)
    _, is_constrained, cons, _ = problem.evaluate([0.5])
    assert "deflection" not in problem.constraint_keys
    assert is_constrained is False