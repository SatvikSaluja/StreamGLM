"""Faithful mock of hnn_core v0.6.1 neymotin_2020_model data structures."""
import itertools as it
import numpy as np

def _pyr_coords(n_x, n_y, z, s):
    return [pos for pos in it.product(np.arange(n_x)*s, np.arange(n_y)*s, [z])]

def _basket_coords(n_x, n_y, z, s, w):
    xzero = np.arange(0, n_x, 3)*s; xone = np.arange(1, n_x, 3)*s
    yeven = np.arange(0, n_y, 2)*s; yodd = np.arange(1, n_y, 2)*s
    coords = [p for p in it.product(xzero, yeven)] + [p for p in it.product(xone, yodd)]
    coords = sorted(coords, key=lambda p: p[1])
    return [(p[0], p[1], w*z) for p in coords]

META = {
    "L2_basket":   ("inhibitory", "2"), "L2_pyramidal": ("excitatory", "2"),
    "L5_basket":   ("inhibitory", "5"), "L5_pyramidal": ("excitatory", "5"),
}

# (src, target, loc, receptors, lamtha, A_weight, allow_autapses) exactly as in
# network_models.py. A_weight values are the real gbar_* defaults: they span 200x
# across classes and several classes SHARE a lamtha with different A_weight. A
# uniform placeholder here would hide pooling bugs, so these must stay realistic.
SPEC = [
    ("L2_pyramidal", "L2_pyramidal", "proximal", ["nmda", "ampa"],   3.0, 5e-4,  False),
    ("L5_pyramidal", "L5_pyramidal", "proximal", ["nmda", "ampa"],   3.0, 5e-4,  False),
    ("L2_basket",    "L2_pyramidal", "soma",     ["gabaa", "gabab"],50.0, 5e-2,  True),
    ("L5_basket",    "L5_pyramidal", "soma",     ["gabaa", "gabab"],70.0, 2.5e-2,True),
    ("L2_pyramidal", "L5_pyramidal", "proximal", ["ampa"],           3.0, 2.5e-4,True),
    ("L2_pyramidal", "L5_pyramidal", "distal",   ["ampa"],           3.0, 2.5e-4,True),
    ("L2_basket",    "L5_pyramidal", "distal",   ["gabaa"],         50.0, 1e-3,  True),
    ("L2_pyramidal", "L2_basket",    "soma",     ["ampa"],           3.0, 5e-4,  True),
    ("L2_basket",    "L2_basket",    "soma",     ["gabaa"],         20.0, 2e-2,  True),
    ("L5_basket",    "L5_basket",    "soma",     ["gabaa"],         20.0, 2e-2,  False),
    ("L5_pyramidal", "L5_basket",    "soma",     ["ampa"],           3.0, 5e-4,  True),
    ("L2_pyramidal", "L5_basket",    "soma",     ["ampa"],           3.0, 2.5e-4,True),
]

class MockNet:
    def __init__(self, mesh_shape=(5, 5), inplane_distance=1.0):
        nx, ny = mesh_shape
        s = inplane_distance
        self._inplane_distance = s
        self.pos_dict = {
            "L2_pyramidal": _pyr_coords(nx, ny, 0.0, s),
            "L5_pyramidal": _pyr_coords(nx, ny, -1307.4, s),
            "L2_basket":    _basket_coords(nx, ny, 0.0, s, 0.8),
            "L5_basket":    _basket_coords(nx, ny, -1307.4, s, 0.8),
        }
        self.cell_types = {
            k: {"cell_object": None,
                "cell_metadata": {"electro_type": META[k][0], "layer": META[k][1]}}
            for k in self.pos_dict
        }
        self.gid_ranges, start = {}, 0
        for k in ["L2_basket", "L2_pyramidal", "L5_basket", "L5_pyramidal"]:
            n = len(self.pos_dict[k]); self.gid_ranges[k] = range(start, start+n); start += n

        self.connectivity = []
        for src, tgt, loc, receptors, lam, a_weight, autapses in SPEC:
            for rec in receptors:
                src_gids = list(self.gid_ranges[src]); tgt_gids = list(self.gid_ranges[tgt])
                gid_pairs = {}
                for sg in src_gids:
                    tg = [g for g in tgt_gids if autapses or g != sg]
                    gid_pairs[sg] = tg
                self.connectivity.append({
                    "src_type": src, "target_type": tgt,
                    "src_gids": set(src_gids), "target_gids": set(tgt_gids),
                    "num_srcs": len(src_gids), "gid_pairs": gid_pairs,
                    "loc": loc, "receptor": rec,
                    "nc_dict": {"A_weight": a_weight, "A_delay": 1.0, "lamtha": lam,
                                "threshold": 0.0, "gain": 1.0},
                    "probability": 1.0, "allow_autapses": autapses,
                })
