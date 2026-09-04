"""Pluggable evaluator registry.

Lets new metrics be added without touching the optimizers or the problem
class. Every metric fills the design.metrics dict; a subset becomes the
optimization objectives [MICARD 1.0].
"""

_METRICS = {}


def register_metric(name):
    """Decorator to register a metric function: fn(design) -> float or dict."""
    def wrapper(fn):
        _METRICS[name] = fn
        return fn
    return wrapper


def compute_all_metrics(design):
    """Run every registered metric and store results in design.metrics."""
    for name, fn in _METRICS.items():
        result = fn(design)
        if isinstance(result, dict):
            design.metrics.update(result)
        else:
            design.metrics[name] = result
    return design.metrics


def available_metrics():
    return list(_METRICS)