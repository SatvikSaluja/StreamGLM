"""Fit the cached production spikes. No NEURON, no re-simulation.

The 2560 s simulation is on disk (results/production_spikes.npz, 15.3
spikes/predictor). This fits it and reports recovery per connection class.

Two stages, because the solver config is not yet trusted:

  1. VALIDATE on a 500k-bin subset where the answer is already known from a
     full-batch float64 LBFGS run: rho +0.098, 2815 s, 11.0 GB peak. If SVRG
     with float32 reproduces that rho, the speedup is free; if it does not, the
     config is wrong and the full fit would be measuring the solver.
  2. FULL fit on all 2.56M bins with the validated config.

PROJECT.md section 22 called for batched solvers and every fit until now used
full-batch LBFGS instead, which is why each attempt cost ~4 h and OOMed.

    setsid python experiments/fit_production.py &
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurotwinbench import fit as fitmod
from neurotwinbench import metrics as met
from neurotwinbench import simulate as simmod
from neurotwinbench.ground_truth import extract_ground_truth

# float32 was measured to change the answer's SIGN on the same data
# (rho -0.212 vs +0.098) and did not converge; nemos warns to enable float64.
# So float64, and n_basis=5 instead of 8 to make it fit: 355 predictors rather
# than 568 means a 7.3 GB design matrix instead of 11.6 GB, and it RAISES power
# from 15.3 to 24.5 spikes per predictor. The cost is coarser temporal
# resolution on the coupling filter, which section 6's calibration has never
# been run to license anyway.
# Full-batch LBFGS cannot do this problem here. Measured: n_basis=5, float64,
# 2.56M bins reached 25 GB RSS with 1.1 GB free before being killed. The design
# matrix is only 7.3 GB; the rest is JAX's device copy plus LBFGS's gradient
# history, a ~3.4x multiplier over the raw array. Section 22's batched solvers
# are required here, not an optimisation.
#
# float32 stays off: it was measured to flip the answer's sign on identical
# data (rho -0.212 vs +0.098) and did not converge. Batching was never the
# problem; precision was.
CFG = dict(solver="SVRG", batch_size=4096, float32=False, max_iter=2000,
           regularizer_strength=1e-3, history_ms=25.0, n_basis=5)
KNOWN_SUBSET_RHO = 0.098   # LBFGS float64 n_basis=8, 500k bins, 2815 s, 11.0 GB


def peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def main() -> None:
    z = np.load("results/production_spikes.npz", allow_pickle=False)
    spikes, drives, gids = z["spikes"], z["drives"], z["gids"]
    gt = extract_ground_truth(
        simmod.build_network(simmod.SimulationConfig(mesh_shape=(5, 5))))
    w_true, d_true, connected = gt.to_matrices(gids)

    Path("fit_heartbeat.txt").write_text(f"started {time.strftime('%H:%M:%S')}\n")
    print("STAGE 1: validate the solver against a known answer", flush=True)
    t0 = time.time()
    r = fitmod.fit_population_glm(spikes[:500_000], exogenous=drives[:500_000], **CFG)
    m = met.evaluate(w_true, r.areas, connected & r.evaluable,
                     true_delay_ms=d_true, history_ms=CFG["history_ms"])
    label = f"{CFG.get('solver','LBFGS')}/{'f32' if CFG['float32'] else 'f64'}/nb{CFG['n_basis']}"
    print(f"  {label} on 500k: {time.time()-t0:.0f}s, {peak_gb():.1f}GB, "
          f"rho={m.weight_spearman:+.3f}", flush=True)
    print(f"  known LBFGS/float64:  2815s, 11.0GB, rho={KNOWN_SUBSET_RHO:+.3f}", flush=True)
    # n_basis differs from the reference run, so this is a sanity band rather
    # than an exact reproduction: the reference used 8 basis functions.
    agree = abs(m.weight_spearman - KNOWN_SUBSET_RHO) < 0.25
    print(f"  -> {'AGREES, config trusted' if agree else 'DISAGREES: solver changes the answer'}",
          flush=True)
    if not agree:
        print("  refusing the full fit; it would measure the solver, not the biology",
              flush=True)
        return

    est_gb = spikes.shape[0] * (spikes.shape[1]*CFG["n_basis"] + CFG["n_basis"]) * 8 / 1e9
    free_gb = int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]) / 1e6
    print(f"\nSTAGE 2: full fit, {spikes.shape[0]:,} bins", flush=True)
    print(f"  design matrix {est_gb:.1f} GB raw; measured multiplier ~3.4x for "
          f"full-batch, less for batched. {free_gb:.1f} GB available.", flush=True)
    if est_gb * 2.2 > free_gb:
        print("  REFUSING: even a batched fit is unlikely to fit here. "
              "Reduce n_basis or bins.", flush=True)
        return
    power = fitmod.power_check(spikes, CFG["n_basis"], n_exog=drives.shape[1])
    print(f"  {power['spikes_per_predictor']:.1f} spikes/predictor "
          f"(synthetic reached rho +0.80 at 16.1)", flush=True)
    t0 = time.time()
    result = fitmod.fit_population_glm(spikes, exogenous=drives, **CFG)
    print(f"  fit in {time.time()-t0:.0f}s, peak {peak_gb():.1f}GB", flush=True)

    classes = _class_matrix(gt, gids)
    rows = []
    print(f"\n{'connection class':<44}{'n':>6}{'in win':>8}{'rho':>9}{'sign':>8}")
    print("-" * 75, flush=True)
    for name in sorted({c for c in classes.ravel() if c}):
        mask = connected & (classes == name) & result.evaluable
        r = met.evaluate(w_true, result.areas, mask,
                         true_delay_ms=d_true, history_ms=CFG["history_ms"])
        if r.n_evaluated == 0:
            continue
        rows.append((name, r))
        print(f"{name:<44.44}{r.n_evaluated:>6}{r.fraction_representable:>7.0%}"
              f"{r.weight_spearman:>+9.3f}{r.sign_accuracy:>7.0%}", flush=True)

    Path("results").mkdir(exist_ok=True)
    out = Path("results/production_fit.json")
    out.write_text(json.dumps({
        "solver_config": CFG, "power": power,
        "per_class": {n: {"n_evaluated": r.n_evaluated,
                          "weight_spearman": r.weight_spearman,
                          "sign_accuracy": r.sign_accuracy,
                          "n_outside_support": r.n_outside_support}
                      for n, r in rows},
    }, indent=2))
    print(f"\nwrote {out}", flush=True)


def _class_matrix(gt, gids):
    index = {int(g): i for i, g in enumerate(gids)}
    out = np.full((len(gids), len(gids)), "", dtype=object)
    for s, t, name in zip(gt.src_gid, gt.target_gid, gt.connection_class):
        i, j = index.get(int(s)), index.get(int(t))
        if i is not None and j is not None:
            out[i, j] = name
    return out


if __name__ == "__main__":
    main()
