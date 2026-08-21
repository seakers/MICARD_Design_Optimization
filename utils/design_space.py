"""Variable-length design space with a fixed-length gene encoding.

Uses a fixed max-length + active-length + padding scheme so random search,
GA, and PPO can all operate on the same representation [design_repair 6].
The gene layout per slot is: [motor_idx, orientation_idx, link_len, tube_idx].
"""
import numpy as np

from utils.design import Design, Segment, ORIENTATIONS
from utils.materials import TUBE_PRESETS, DEFAULT_MATERIAL


class DesignSpace:
    def __init__(self, catalog, max_joints=7, min_joints=4,
                 link_len_range=(0.05, 1.0), n_ports=7,
                 material=DEFAULT_MATERIAL):
        self.catalog = catalog
        self.max_joints = max_joints
        self.min_joints = min_joints
        self.link_len_range = link_len_range
        self.n_ports = n_ports          # 7 mounting points on the platform [MICARD scenario]
        self.material = material
        self.n_orient = len(ORIENTATIONS)
        self.n_motors = len(catalog)
        self.n_tubes = len(TUBE_PRESETS)

    def build_gene_space(self):
        """Return a list of gene descriptors (drop-in for optimizer loops [random_search 5])."""
        space = [
            {"name": "base_port", "type": "discrete",
             "range": list(range(self.n_ports))},
            {"name": "active_length", "type": "discrete",
             "range": list(range(self.min_joints, self.max_joints + 1))},
        ]
        for i in range(self.max_joints):
            space.append({"name": f"motor_{i}", "type": "discrete",
                          "range": list(range(self.n_motors))})
            space.append({"name": f"orient_{i}", "type": "discrete",
                          "range": list(range(self.n_orient))})
            space.append({"name": f"link_{i}", "type": "continuous",
                          "range": list(self.link_len_range)})
            space.append({"name": f"tube_{i}", "type": "discrete",
                          "range": list(range(self.n_tubes))})
        return space

    def decode(self, genes):
        """Turn a gene vector into a Design, honoring active_length (padding ignored)."""
        base_port = int(genes[0])
        active_length = int(genes[1])
        segments = []
        for i in range(active_length):
            offset = 2 + i * 4
            motor_idx = int(genes[offset]) % self.n_motors
            orient_idx = int(genes[offset + 1]) % self.n_orient
            link_len = float(np.clip(genes[offset + 2], *self.link_len_range))
            tube_idx = int(genes[offset + 3]) % self.n_tubes
            outer_d, wall = TUBE_PRESETS[tube_idx]
            segments.append(Segment(
                motor=self.catalog[motor_idx],
                orientation=ORIENTATIONS[orient_idx],
                link_length_m=link_len,
                tube_outer_d_m=outer_d,
                tube_wall_m=wall,
                material=self.material,
            ))
        return Design(segments=segments, base_port=base_port)

    def random_genes(self, rng=None):
        """Sample one valid gene vector (used by random search / GA init)."""
        rng = rng or np.random.default_rng()
        genes = [rng.integers(0, self.n_ports),
                 rng.integers(self.min_joints, self.max_joints + 1)]
        lo, hi = self.link_len_range
        for _ in range(self.max_joints):
            genes.append(rng.integers(0, self.n_motors))
            genes.append(rng.integers(0, self.n_orient))
            genes.append(rng.uniform(lo, hi))
            genes.append(rng.integers(0, self.n_tubes))
        return np.array(genes, dtype=float)