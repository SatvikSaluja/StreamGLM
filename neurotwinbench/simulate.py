"""HNN-Core simulation with sustained drive, plus a content-addressed cache.

Produces the two things the study needs from one run: observable spiking
activity, and the exact circuit that generated it. The second is hidden during
fitting.

Simulation is the binding resource (PROJECT.md constraint A), so every result is
cached by a hash of its complete scientific configuration and never recomputed.

Defaults follow section 3: `mesh_shape=(5,5)` (70 cells) with sustained Poisson
drive rather than a single evoked transient, because a coupling GLM needs many
spikes per neuron and an ERP-style simulation does not provide them. 5x5 is the
development environment, not merely the small condition -- at 10x10 only ~53% of
excitatory connections fall inside a 25 ms coupling window.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CACHE_DIR = Path("cache/simulations")


@dataclass
class SimulationConfig:
    """Everything that changes the result. The cache key is a hash of this."""

    mesh_shape: tuple[int, int] = (5, 5)
    tstop_ms: float = 1000.0
    n_trials: int = 1
    dt_ms: float = 0.025
    # Sustained drive. rate_constant is per drive cell, in Hz.
    drive_rate_hz: float = 10.0
    drive_weight_ampa: float = 2e-4
    drive_weight_nmda: float = 1e-4
    drive_location: str = "proximal"
    event_seed: int = 2
    conn_seed: int = 3
    legacy_mode: bool = False
    # "default" = neymotin_2020_model as shipped.
    # "location" = purpose-built network where each pair has exactly one
    #   dendritic location, which the default model cannot provide (section 14).
    network: str = "default"
    location_weight: float = 5e-4
    # Non-default connection probabilities, as {connection_index: probability}.
    # Sparse networks are the only way to get meaningful topology metrics
    # (section 2B) and independent circuit realisations (section 20).
    connection_probability: dict = field(default_factory=dict)

    def key(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class Simulation:
    spikes: np.ndarray  # (n_bins, n_neurons) counts, bin_s resolution
    gids: np.ndarray  # (n_neurons,) column order of `spikes`
    cell_types: np.ndarray  # (n_neurons,) cell type per column
    drive_spikes: np.ndarray  # (n_bins, n_drives) the section 12 condition-A covariate
    drive_names: list[str]
    bin_s: float
    config: SimulationConfig

    @property
    def rates_hz(self) -> np.ndarray:
        return self.spikes.mean(axis=0) / self.bin_s

    def observed(self, exclude_types: tuple[str, ...] = ()) -> np.ndarray:
        """Column mask after hiding cell types -- section 19's interneuron dropout.

            sim.observed(exclude_types=("L2_basket", "L5_basket"))
        """
        return ~np.isin(self.cell_types, exclude_types)


def build_network(config: SimulationConfig):
    """Construct the network and attach sustained drive. No simulation yet."""
    from hnn_core import neymotin_2020_model

    net = neymotin_2020_model(
        mesh_shape=tuple(config.mesh_shape), legacy_mode=config.legacy_mode
    )

    if config.network == "location":
        _rebuild_as_location_network(net, config)
    elif config.network != "default":
        raise ValueError(f"unknown network variant {config.network!r}")

    _thin_connections(net, config)

    weights_ampa = {
        cell: config.drive_weight_ampa
        for cell in ("L2_basket", "L2_pyramidal", "L5_basket", "L5_pyramidal")
    }
    weights_nmda = {cell: config.drive_weight_nmda for cell in weights_ampa}

    net.add_poisson_drive(
        name="sustained",
        tstart=0.0,
        tstop=config.tstop_ms,
        rate_constant=config.drive_rate_hz,
        location=config.drive_location,
        weights_ampa=weights_ampa,
        weights_nmda=weights_nmda,
        event_seed=config.event_seed,
        conn_seed=config.conn_seed,
    )
    return net


def run(
    config: SimulationConfig | None = None,
    bin_s: float = 0.001,
    n_procs: int | None = None,
    n_jobs: int = 1,
    cache_dir: Path = CACHE_DIR,
    use_cache: bool = True,
) -> Simulation:
    """Simulate, or return the cached result for this exact configuration.

    `n_procs` selects MPIBackend; None uses the serial JoblibBackend. MPI is the
    reference for any timing measurement (section 3), and a benchmark record is
    not reproducible without also recording process count and hardware.

    MPI requires `nrniv` on PATH -- hnn-core builds the literal command
    "mpiexec -np N nrniv -python -mpi ..." with a bare `nrniv`, so invoking
    python by absolute path without the environment's bin directory on PATH
    fails with return code 134 (SIGABRT), not a readable error. It also needs
    psutil and mpi4py, which are not hnn-core dependencies; without psutil the
    real failure is masked by a ModuleNotFoundError raised inside the cleanup
    handler. `_require_nrniv_on_path` turns all of that into one clear message.
    """
    config = config or SimulationConfig()
    cache_path = cache_dir / f"{config.key()}.npz"

    if use_cache and cache_path.exists():
        return _load(cache_path, config)

    from hnn_core import JoblibBackend, MPIBackend, simulate_dipole

    net = build_network(config)
    if n_procs:
        _require_nrniv_on_path()
        _require_no_concurrent_nrniv()
    # Two ways to use N cores, and which wins depends on n_trials. MPIBackend
    # splits ONE trial across ranks; JoblibBackend runs trials in parallel, one
    # core each. Trials are not a workaround: they are independent drive-event
    # realisations on fixed anatomy (the event_seed axis of section 20), and
    # _extract concatenates them.
    backend = MPIBackend(n_procs=n_procs) if n_procs else JoblibBackend(n_jobs=n_jobs)
    with backend:
        simulate_dipole(net, tstop=config.tstop_ms, dt=config.dt_ms,
                        n_trials=config.n_trials)

    sim = _extract(net, config, bin_s)
    if use_cache:
        _save(cache_path, sim)
    return sim


#: The controlled location contrast. Both sections accept AMPA, sit on the same
#: target population, and are reached by the same `sect_loc` groups in the
#: default model -- `basal_2` belongs to "proximal", `apical_tuft` to "distal".
LOCATION_SECTIONS = ("basal_2", "apical_tuft")


def _rebuild_as_location_network(net, config: SimulationConfig) -> None:
    """Purpose-built network where dendritic location IS identifiable.

    Section 14 retracted the default model's apparent location contrast: every
    L2pyr->L5pyr pair carries a proximal *and* a distal synapse, so the single
    coupling filter a GLM estimates per ordered pair is their sum and cannot be
    attributed to either. No amount of analysis recovers what was never
    separable.

    This network makes it separable by construction. The L2 pyramidal population
    is split in half by gid; one half projects onto `basal_2` and the other onto
    `apical_tuft` of the same L5 pyramidal targets, with identical receptor,
    weight and lamtha. Each ordered pair therefore has exactly one location, and
    the two groups differ in nothing else.

        L2pyr[half A] --AMPA--> L5pyr  basal_2      (proximal group)
        L2pyr[half B] --AMPA--> L5pyr  apical_tuft  (distal group)

    Feedforward inhibition is retained so the network stays in a firing regime
    rather than running away; the drive sweep showed the E/I balance is what
    keeps pyramidal cells alive (section 3).

    Requires `legacy_mode=False`: hnn-core only permits targeting a named
    section when legacy mode is off.
    """
    if config.legacy_mode:
        raise ValueError("the location network requires legacy_mode=False")

    src_gids = sorted(net.gid_ranges["L2_pyramidal"])
    target_gids = sorted(net.gid_ranges["L5_pyramidal"])
    half = len(src_gids) // 2
    if half == 0:
        raise ValueError("mesh too small to split the source population")

    # Surgical replacement, not a rebuild. Only the confounded L2pyr->L5pyr
    # projection is removed; every other connection keeps its default parameters.
    #
    # Clearing all connectivity and re-adding a chosen subset was tried first and
    # is wrong: it drops the recurrent pyramidal excitation that keeps the
    # network firing, leaving full-strength inhibition unopposed. The result was
    # 0.2 Hz mean with 46/70 cells silent -- an inhibition-dominated network that
    # no longer resembles the model being studied. Preserving the default E/I
    # balance is what keeps the condition interpretable.
    removed = [
        c for c in net.connectivity
        if c["src_type"] == "L2_pyramidal" and c["target_type"] == "L5_pyramidal"
    ]
    if not removed:
        raise RuntimeError("expected L2pyr->L5pyr connections to replace")
    net.connectivity = [c for c in net.connectivity if c not in removed]

    for section, sources in (
        (LOCATION_SECTIONS[0], src_gids[:half]),
        (LOCATION_SECTIONS[1], src_gids[half:]),
    ):
        net.add_connection(
            src_gids=list(sources), target_gids=list(target_gids),
            loc=section, receptor="ampa",
            weight=config.location_weight, delay=net.delay, lamtha=3.0,
        )


def _thin_connections(net, config: SimulationConfig) -> None:
    """Prune connections to `probability`, giving a sparse network.

    Sparse networks are the only route to two things the default model cannot
    provide: meaningful topology metrics, since at probability 1.0 there are no
    designed non-edges to detect (§2B), and a second uncertainty axis, since
    `conn_seed` does nothing when nothing is pruned (§20).

    Uses hnn-core's own `_connection_probability` rather than reimplementing the
    pruning, so the retained subset matches what the simulator would build. It
    is private API, hence the guarded import: if it moves, fail loudly here
    rather than silently diverge from upstream.

    `connection_probability` maps a class name substring to a probability, e.g.
    `{"L2_pyramidal->L2_pyramidal": 0.3}`. An entry matching nothing is an error
    -- a silent no-op would mean running a "sparse" condition that is dense.
    """
    if not config.connection_probability:
        return

    try:
        from hnn_core.network import _connection_probability
    except ImportError as exc:  # pragma: no cover - upstream moved it
        raise RuntimeError(
            "hnn_core.network._connection_probability is unavailable; sparse "
            "networks depend on it. Check the installed hnn-core version."
        ) from exc

    for pattern, probability in config.connection_probability.items():
        matched = 0
        for conn in net.connectivity:
            name = f"{conn['src_type']}->{conn['target_type']}"
            if pattern not in name and pattern not in f"{name}|{conn['receptor']}|{conn['loc']}":
                continue
            _connection_probability(conn, float(probability), config.conn_seed)
            matched += 1
        if matched == 0:
            raise ValueError(
                f"connection_probability pattern {pattern!r} matched no connection "
                f"class; the network would be dense while claiming to be sparse"
            )


def _require_no_concurrent_nrniv() -> None:
    """Refuse to start if another hnn-core MPI simulation is running.

    hnn-core's `MPIBackend.__exit__` calls `kill_proc_name("nrniv")` whenever
    n_procs > 1 -- "always kill nrniv processes for good measure". That helper
    matches processes by NAME across the whole machine, with no PID or process
    group scoping, so it kills every nrniv on the host, including workers
    belonging to unrelated simulations.

    Two hnn-core MPI runs therefore cannot coexist: whichever finishes first
    kills the other, which dies with return code 143 (SIGTERM) and a traceback
    pointing at MPI rather than at the real cause. A 300 s run was lost this way
    to a 6 s smoke test started alongside it.

    Consequence for experiment design: HNN simulations must be run
    **sequentially**, never in parallel, even though the machine has spare
    cores. Parallelism has to come from n_procs within one simulation.
    """
    try:
        from psutil import process_iter
    except ImportError:
        return  # already reported by _require_nrniv_on_path

    running = [
        p.pid for p in process_iter(attrs=["name"])
        if p.info["name"] == "nrniv"
    ]
    if running:
        raise RuntimeError(
            f"{len(running)} nrniv process(es) already running (pids {running[:5]}). "
            f"hnn-core kills nrniv by name across the whole machine when an "
            f"MPIBackend exits, so starting now would kill that simulation and "
            f"this one would likely be killed in turn. Run HNN simulations "
            f"sequentially, or pass n_procs=None to use the serial backend."
        )


def _require_nrniv_on_path() -> None:
    """Fail with a readable message instead of SIGABRT deep inside mpiexec."""
    import shutil
    import sys

    missing = [m for m in ("psutil", "mpi4py") if _import_fails(m)]
    if missing:
        raise RuntimeError(
            f"MPIBackend needs {' and '.join(missing)}, which hnn-core does not "
            f"install. Without psutil the real MPI error is swallowed by the "
            f"cleanup handler. Run: pip install {' '.join(missing)}"
        )
    if shutil.which("nrniv") is None:
        bindir = Path(sys.executable).parent
        raise RuntimeError(
            f"'nrniv' is not on PATH. hnn-core runs the bare command "
            f"'mpiexec -np N nrniv -python -mpi ...', so MPI aborts with return "
            f"code 134 rather than a useful error. Fix with:\n"
            f"    export PATH=\"{bindir}:$PATH\""
        )


def _import_fails(module: str) -> bool:
    from importlib.util import find_spec

    try:
        return find_spec(module) is None
    except (ImportError, ValueError):
        return True


def _extract(net, config: SimulationConfig, bin_s: float) -> Simulation:
    """Bin spike times into (n_bins, n_neurons), trials concatenated."""
    response = net.cell_response
    cell_types = list(net.cell_types)
    gids = np.concatenate([np.array(sorted(net.gid_ranges[t])) for t in cell_types])
    types = np.concatenate([
        np.full(len(net.gid_ranges[t]), t) for t in cell_types
    ])
    column = {int(g): i for i, g in enumerate(gids)}

    drive_names = sorted(net.external_drives)
    drive_column = {name: i for i, name in enumerate(drive_names)}
    drive_gid_to_name = {
        int(g): name for name in drive_names for g in net.gid_ranges[name]
    }

    per_trial = int(round(config.tstop_ms / 1000.0 / bin_s))
    n_bins = per_trial * config.n_trials
    spikes = np.zeros((n_bins, len(gids)), dtype=np.int16)
    drive_spikes = np.zeros((n_bins, len(drive_names)), dtype=np.int16)

    for trial, (times, spike_gids) in enumerate(
        zip(response.spike_times, response.spike_gids)
    ):
        offset = trial * per_trial
        for t_ms, gid in zip(times, spike_gids):
            b = offset + int(t_ms / 1000.0 / bin_s)
            if not (offset <= b < offset + per_trial):
                continue
            gid = int(gid)
            if gid in column:
                spikes[b, column[gid]] += 1
            elif gid in drive_gid_to_name:
                drive_spikes[b, drive_column[drive_gid_to_name[gid]]] += 1

    return Simulation(
        spikes=spikes, gids=gids, cell_types=types,
        drive_spikes=drive_spikes, drive_names=drive_names,
        bin_s=bin_s, config=config,
    )


def _save(path: Path, sim: Simulation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, spikes=sim.spikes, gids=sim.gids, cell_types=sim.cell_types,
        drive_spikes=sim.drive_spikes, drive_names=np.array(sim.drive_names),
        bin_s=sim.bin_s, config=json.dumps(sim.config.__dict__, default=str),
    )


def _load(path: Path, config: SimulationConfig) -> Simulation:
    z = np.load(path, allow_pickle=False)
    return Simulation(
        spikes=z["spikes"], gids=z["gids"], cell_types=z["cell_types"],
        drive_spikes=z["drive_spikes"], drive_names=list(z["drive_names"]),
        bin_s=float(z["bin_s"]), config=config,
    )
