from outputs.design_optimization_export import export_design, new_run_id

# One run id groups all designs from a single optimization run.
run_id = new_run_id()

# For each design produced by the optimizer (already decoded + evaluated so
# design.metrics is populated [10][11]):
result = export_design(design, run_id=run_id, method="Genetic Algorithm")
# -> {"design_id": "design_144613_323454", "run_id": "20260911_144613",
#     "artifact_dir": "tool_artifacts/design_optimization/.../designs/..."}