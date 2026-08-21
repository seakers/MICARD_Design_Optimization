"""Generic, schema-agnostic parts catalog loader.

Designed so an SME can edit the JSON without touching code [MICARD 2.0.3],
and so future catalog schema changes cause minimal breakage [MICARD 2.0.1].
"""
import json
from pathlib import Path


# Fields required for a component to be usable by the evaluators.
# Edit this list if the schema changes or new metrics need new fields.
REQUIRED_FIELDS = ("nominal_torque_nm", "weight_g", "cost_usd")


class Catalog:
    """Holds a list of component dicts and offers simple, generic access."""

    def __init__(self, components, source_path=None):
        self.components = components
        self.source_path = source_path

    def __len__(self):
        return len(self.components)

    def __getitem__(self, idx):
        return self.components[idx]

    def field_values(self, field):
        """All values present for a field (useful for building the design space)."""
        return [c[field] for c in self.components if c.get(field) is not None]

    def filter(self, predicate):
        """Return a new Catalog of components matching a predicate function."""
        return Catalog([c for c in self.components if predicate(c)], self.source_path)


def _is_complete(component, required_fields, defaults):
    for f in required_fields:
        if component.get(f) is None and f not in defaults:
            return False
    return True


def load_catalog(
    path,
    required_fields=REQUIRED_FIELDS,
    include_incomplete=False,
    defaults=None,
    flag_incomplete=True,
):
    """Load a JSON parts catalog generically.

    Args:
        path: path to a JSON file (a list of component objects).
        required_fields: fields a component must have to be usable.
        include_incomplete: if True, keep components missing required fields.
        defaults: optional dict of {field: default_value} filled in for missing
                  fields (applies whether or not include_incomplete is set).
        flag_incomplete: if True, tag kept-but-incomplete components with
                         '_incomplete': True for downstream transparency.

    Returns:
        Catalog
    """
    defaults = defaults or {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    if not isinstance(raw, list):
        raise ValueError("Expected the catalog JSON to be a list of components.")

    kept = []
    dropped = 0
    for comp in raw:
        comp = dict(comp)  # copy so we don't mutate source
        complete = _is_complete(comp, required_fields, defaults)

        # Fill defaults where provided.
        for field, value in defaults.items():
            if comp.get(field) is None:
                comp[field] = value

        still_missing = any(comp.get(f) is None for f in required_fields)

        if still_missing and not include_incomplete:
            dropped += 1
            continue

        if still_missing and flag_incomplete:
            comp["_incomplete"] = True

        kept.append(comp)

    print(f"Catalog: loaded {len(kept)} components "
          f"({dropped} dropped for missing {required_fields}).")
    return Catalog(kept, source_path=str(path))