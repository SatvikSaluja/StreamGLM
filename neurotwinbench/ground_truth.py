"""HNN-Core connectivity -> per-pair biological ground truth.

Resolves `net.connectivity` connection specifications into the effective
per-pair quantities NEURON actually uses, reproducing hnn_core v0.6.1
`cell._get_gaussian_connection`:

    scaled_lamtha = lamtha * inplane_distance
    g(d)          = exp(-(d_xy / scaled_lamtha)**2)
    weight        = A_weight * gain * g(d)
    delay         = A_delay / g(d)

Distance is in the xy plane only (cell.py: "Distance in xy plane is used for
gaussian decay"). `gain` is applied before the Gaussian, matching
network_builder `_connect_celltypes`; the builder deepcopies nc_dict first, so
stored connectivity is never gain-mutated and this is correct pre- or
post-simulation.

Two facts about the default model that shape every downstream metric, both
measurable with `scripts/ground_truth_audit.py`:

  * Within a connection class, weight * delay == A_weight * gain * A_delay, so
    Spearman rho(weight, delay) == -1 exactly. Weight and delay are perfectly
    rank-confounded and cannot be separated (PROJECT.md section 15).
  * lamtha is class-determined: 3.0 for every pyramidal-source connection,
    20-70 for basket-source. Inhibitory classes have almost no weight spread,
    so rank metrics on them are ranking numerical noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class GroundTruth:
    """Long-form table of every realised connection."""

    src_gid: np.ndarray
    target_gid: np.ndarray
    src_type: np.ndarray
    target_type: np.ndarray
    receptor: np.ndarray
    loc: np.ndarray
    weight: np.ndarray  # effective, distance-scaled
    delay_ms: np.ndarray  # effective, distance-scaled
    sign: np.ndarray  # +1 excitatory source, -1 inhibitory
    lamtha: np.ndarray

    def __len__(self) -> int:
        return len(self.src_gid)

    @property
    def connection_class(self) -> np.ndarray:
        """`src->target|receptor|loc` per row. The only valid grouping key.

        Grouping by `lamtha` is NOT equivalent and silently corrupts results:
        several classes share a lamtha with different `A_weight`. At 5x5,
        lamtha=50 covers L2bask->L2pyr (A_weight 5e-2) and L2bask->L5pyr
        (1e-3). Each has CV 0.0027 and rho = -1; pooled they read CV 0.686 and
        rho = -0.554, which is an artifact of mixing two weight scales, not a
        property of either class.
        """
        return np.array([
            f"{s}->{t}|{r}|{l}"
            for s, t, r, l in zip(self.src_type, self.target_type, self.receptor, self.loc)
        ])

    def by_class(self):
        """Yield (class_name, boolean mask) for each connection class."""
        classes = self.connection_class
        for name in sorted(set(classes)):
            yield name, classes == name

    def to_matrices(self, gids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project onto an observed subset, returning (weight, delay_ms, connected).

        `gids` is the observed population in the order its spike columns appear.
        Pairs involving unobserved cells are dropped -- that is what hiding a
        neuron means (PROJECT.md section 19).
        """
        index = {int(g): i for i, g in enumerate(gids)}
        n = len(gids)
        weight = np.zeros((n, n))
        delay = np.full((n, n), np.inf)
        connected = np.zeros((n, n), dtype=bool)

        for s, t, w, d in zip(self.src_gid, self.target_gid, self.weight, self.delay_ms):
            i, j = index.get(int(s)), index.get(int(t))
            if i is None or j is None:
                continue
            # A pair may carry several receptors; sum signed weight, keep fastest.
            weight[i, j] += w
            delay[i, j] = min(delay[i, j], d)
            connected[i, j] = True

        return weight, delay, connected


def extract_ground_truth(net, include_drives: bool = False) -> GroundTruth:
    """Resolve an hnn_core Network's connectivity into per-pair ground truth."""
    inplane = float(net._inplane_distance)
    positions = _gid_positions(net)
    cell_types = set(net.cell_types)

    rows: list[tuple] = []
    for conn in net.connectivity:
        src_type, target_type = conn["src_type"], conn["target_type"]
        if not include_drives and src_type not in cell_types:
            continue
        if src_type not in cell_types or target_type not in cell_types:
            continue

        nc = conn["nc_dict"]
        a_weight = float(nc["A_weight"]) * float(nc.get("gain", 1.0))
        a_delay = float(nc["A_delay"])
        lamtha = float(nc["lamtha"])
        scaled = lamtha * inplane
        sign = -1.0 if _is_inhibitory(net, src_type) else 1.0

        for src_gid, target_gids in conn["gid_pairs"].items():
            src_pos = positions.get(int(src_gid))
            if src_pos is None:
                continue
            for target_gid in target_gids:
                target_pos = positions.get(int(target_gid))
                if target_pos is None:
                    continue
                dist = math.hypot(target_pos[0] - src_pos[0], target_pos[1] - src_pos[1])
                g = math.exp(-((dist / scaled) ** 2))
                rows.append((
                    int(src_gid), int(target_gid), src_type, target_type,
                    conn["receptor"], conn["loc"],
                    sign * a_weight * g,
                    a_delay / g if g > 0 else math.inf,
                    sign, lamtha,
                ))

    if not rows:
        raise ValueError("no intrinsic connections found in network")

    cols = list(zip(*rows))
    return GroundTruth(
        src_gid=np.array(cols[0]), target_gid=np.array(cols[1]),
        src_type=np.array(cols[2]), target_type=np.array(cols[3]),
        receptor=np.array(cols[4]), loc=np.array(cols[5]),
        weight=np.array(cols[6]), delay_ms=np.array(cols[7]),
        sign=np.array(cols[8]), lamtha=np.array(cols[9]),
    )


def _gid_positions(net) -> dict[int, tuple[float, float, float]]:
    out: dict[int, tuple[float, float, float]] = {}
    for cell_type in net.cell_types:
        gids = list(net.gid_ranges[cell_type])
        pos = list(net.pos_dict[cell_type])
        if len(gids) != len(pos):
            raise RuntimeError(f"gid/position mismatch for {cell_type}")
        for gid, p in zip(gids, pos):
            out[int(gid)] = (float(p[0]), float(p[1]), float(p[2]))
    return out


def _is_inhibitory(net, cell_type: str) -> bool:
    meta = net.cell_types[cell_type]["cell_metadata"]
    return meta.get("electro_type") == "inhibitory"
