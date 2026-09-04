"""Material properties and preset tube cross-sections for link design."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    density_kg_m3: float        # for weight
    elastic_modulus_pa: float   # for deflection
    yield_strength_pa: float    # for snapping


# Edit / extend this table freely. Values are typical engineering estimates.
MATERIALS = {
    "aluminum":     Material("aluminum",     2700.0,  69.0e9,  275.0e6),
    "steel":        Material("steel",        7850.0, 200.0e9,  250.0e6),
    "abs":          Material("abs",          1040.0,   2.3e9,   40.0e6),
    "carbon_fiber": Material("carbon_fiber", 1600.0,  70.0e9,  600.0e6),
}

# Preset tube cross-sections: (outer_diameter_m, wall_thickness_m)
TUBE_PRESETS = [
    (0.040, 0.002),
    (0.060, 0.003),
    (0.080, 0.003),
    (0.100, 0.004),
]

DEFAULT_MATERIAL = "aluminum"


def get_material(name: str) -> Material:
    key = name.lower()
    if key not in MATERIALS:
        raise KeyError(f"Unknown material '{name}'. Options: {list(MATERIALS)}")
    return MATERIALS[key]