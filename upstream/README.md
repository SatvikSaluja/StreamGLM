# Upstream contributions

Two proposed contributions to [hnn-core](https://github.com/jonescompneurolab/hnn-core), both arising from problems hit while building NeuroTwinBench rather than invented to have something to contribute (PROJECT.md §28).

---

## 1. `MPIBackend` preflight — small, high confidence

**Problem.** `MPIBackend` fails with `RuntimeError: MPI simulation failed. Return code: 134` (SIGABRT) when `nrniv` is not on `PATH`. The command built in `parallel_backends.py` is:

```
mpiexec [--use-hwthread-cpus] -np N nrniv -python -mpi -nobanner <python> <mpi_child.py>
```

`nrniv` is bare, so it must be resolvable on `PATH`. Invoking Python by absolute path — routine in a conda environment, a CI job, or any script that does not activate the env first — leaves the environment's `bin` off `PATH` and `mpiexec` aborts without a readable cause.

**Two traps compound it.**

`psutil` and `mpi4py` are not hnn-core dependencies, but `MPIBackend` imports `psutil` inside `kill_proc_name`, which runs in the cleanup path of both `run_subprocess` and `__exit__`. When it is missing, the genuine MPI error is **replaced** by a `ModuleNotFoundError` raised during cleanup — so the first traceback a user sees is unrelated to the actual problem. We lost time to exactly this.

**Proposed fix.** A preflight check in `MPIBackend.__init__`, before any simulation starts. Reference implementation: `neurotwinbench/simulate.py::_require_nrniv_on_path`.

```python
def _preflight(self):
    missing = [m for m in ("psutil", "mpi4py") if importlib.util.find_spec(m) is None]
    if missing:
        raise RuntimeError(
            f"MPIBackend requires {' and '.join(missing)}. Without psutil the "
            f"underlying MPI error is masked by an error raised during cleanup. "
            f"Install with: pip install {' '.join(missing)}"
        )
    if shutil.which("nrniv") is None:
        raise RuntimeError(
            f"'nrniv' is not on PATH, but MPIBackend runs "
            f"'mpiexec -np N nrniv ...'. MPI will abort with return code 134 "
            f"rather than a useful error. Add your environment's bin directory "
            f"to PATH, e.g.\n    export PATH=\"{Path(sys.executable).parent}:$PATH\""
        )
```

**Alternative worth considering:** resolve `nrniv` via `shutil.which("nrniv") or str(Path(sys.executable).parent / "nrniv")` and use the absolute path in the command. That fixes the common case outright rather than explaining it. The preflight is still wanted for the missing-dependency half.

**Suggested also:** move `psutil` out of the cleanup path, or guard its import, so a missing optional dependency cannot mask the real exception.

**Verified against** hnn-core 0.6.1, NEURON 8.2.7, OpenMPI 4.1.6 on Linux. With `PATH` set, 2/4/8 processes all run and produce byte-identical spike counts.

---

## 2. Per-pair ground-truth resolver — the substantial one

**Problem.** `net.connectivity` stores connection *specifications* — `A_weight`, `A_delay`, `lamtha`, `gid_pairs` — but not the per-pair values NEURON actually uses. Those are computed inside `cell._get_gaussian_connection` at build time and are not exposed.

Anyone who wants to know what a given synapse's weight and delay actually are must reimplement:

```python
scaled_lamtha = lamtha * inplane_distance
g(d)          = exp(-(d_xy / scaled_lamtha)**2)     # xy plane only
weight        = A_weight * gain * g(d)
delay         = A_delay / g(d)
```

Getting that wrong is easy and silent. Three details that are not obvious from the outside: distance is **xy-plane only** (z is ignored despite cells sitting in different layers); `lamtha` is scaled by `inplane_distance`; and `gain` is applied to `A_weight` *before* the Gaussian, via a `deepcopy` in `_connect_celltypes`, so stored connectivity is never gain-mutated.

**Who needs it.** Anyone validating an inference method against known connectivity, plotting an effective weight matrix, or comparing network variants quantitatively. It is the difference between HNN as a simulator and HNN as a source of ground truth.

**Proposed API.** A method on `Network`:

```python
truth = net.effective_connectivity()   # -> list of dict, or a DataFrame
# src_gid, target_gid, src_type, target_type, receptor, loc,
# weight, delay, distance, lamtha
```

Reference implementation: `neurotwinbench/ground_truth.py::extract_ground_truth`, validated against a hand-computed two-cell network and reproducing the audit's per-class statistics exactly on the real model.

**Two findings this exposes**, both properties of the default model that are invisible without it and that change how results should be analysed:

- Within a connection class `weight × delay = A_weight × gain × A_delay`, a constant, so **Spearman ρ(weight, delay) = −1 exactly**. Verified in all 16 classes at 5×5 and 10×10. Weight and delay cannot be separated by any rank-based method.
- `lamtha` is **not** a valid grouping key. Several classes share a `lamtha` with different `A_weight`; pooling them turns CV 0.0027 / ρ −1.0 into CV 0.686 / ρ −0.554, a pure artifact of mixing weight scales. Group by `(src_type, target_type, receptor, loc)`.

Shipping the resolver upstream means users get these for free instead of rediscovering them — or, more likely, not noticing them.

---

## 3. `kill_proc_name("nrniv")` is machine-wide, not job-scoped

**Problem.** `MPIBackend.__exit__` calls `kill_proc_name("nrniv")` whenever `n_procs > 1` — the comment reads *"always kill nrniv processes for good measure."* `_get_procs_running` matches by process **name** across every process on the host:

```python
for p in process_iter(attrs=["name", "exe", "cmdline"]):
    if proc_name == p.info["name"] or ...:
        process_list.append(p)
```

There is no PID, process-group, or session scoping. So a finishing simulation kills the MPI workers of **every other hnn-core simulation on the machine**.

**Consequence.** Two hnn-core MPI simulations cannot coexist. Whichever exits first terminates the other, which then dies with `RuntimeError: MPI simulation failed. Return code: 143` and a traceback pointing at MPI rather than at the real cause — an unrelated process. We lost a 300-second run to a 6-second smoke test started alongside it, and the failure mode gives no hint that concurrency was involved.

This is easy to hit: a multi-core machine invites running several conditions at once, and nothing in the API suggests that is unsafe.

**Proposed fix.** Track the PIDs this backend actually spawned and kill only those. `Popen` already returns the child; its process group can be captured with `start_new_session=True` and killed with `os.killpg`. Falling back to name-matching is reasonable only if no PIDs were recorded.

Failing that, the behaviour should at minimum be documented and opt-out-able, e.g. `MPIBackend(..., kill_stray_nrniv=False)`.

**Workaround in this project.** `simulate._require_no_concurrent_nrniv` refuses to start an MPI simulation while any `nrniv` is running, with a message naming the cause. HNN simulations are run sequentially; parallelism comes from `n_procs` within a single simulation.
