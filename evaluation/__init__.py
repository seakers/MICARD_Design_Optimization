"""Importing this package registers all metrics with the evaluator registry.

Each metric module uses @register_metric as an import side effect, so we import
them here to guarantee they are all registered before compute_all_metrics runs.
Registration order matters: conditioning is imported AFTER manipulability so the
real trac_ik 'w' overwrites the placeholder when the micard-metrics service is up,
and the placeholder stands when it is down (graceful degradation).
"""
from evaluation.base import (  # noqa: F401
    register_metric,
    compute_all_metrics,
    available_metrics,
)

# Placeholder metrics first.
from evaluation import cost           # noqa: F401
from evaluation import weight         # noqa: F401
from evaluation import manipulability # noqa: F401  (placeholder Yoshikawa)
from evaluation import reachability   # noqa: F401
from evaluation import structural     # noqa: F401
from evaluation import power          # noqa: F401

# Real conditioning metrics LAST so 'w' overwrites the placeholder when available.
from evaluation import conditioning   # noqa: F401