"""Design representation: a variable-length serial chain of joints + links."""
from dataclasses import dataclass, field, asdict


# Orientation options. Easy to extend (e.g., add a passive elbow later) [user].
ORIENTATIONS = ("roll", "pitch", "yaw")

# Joint rotation axes in the local link frame (chain grows along +X).
ORIENTATION_AXES = {
    "roll":  (1.0, 0.0, 0.0),
    "pitch": (0.0, 1.0, 0.0),
    "yaw":   (0.0, 0.0, 1.0),
}


@dataclass
class Segment:
    motor: dict                 # a component dict from the Catalog
    orientation: str            # one of ORIENTATIONS
    link_length_m: float
    tube_outer_d_m: float
    tube_wall_m: float
    material: str = "aluminum"

    @property
    def axis(self):
        return ORIENTATION_AXES[self.orientation]


@dataclass
class Design:
    segments: list = field(default_factory=list)  # list[Segment]
    base_port: int = 0                             # which platform port [MICARD scenario]
    metrics: dict = field(default_factory=dict)    # ALL evaluated quantities

    @property
    def dof(self):
        return len(self.segments)

    def to_dict(self):
        return {
            "base_port": self.base_port,
            "segments": [asdict(s) for s in self.segments],
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, d):
        segments = [Segment(**s) for s in d.get("segments", [])]
        return cls(
            segments=segments,
            base_port=d.get("base_port", 0),
            metrics=d.get("metrics", {}),
        )