"""Milestone 2: the easiest valid HNN condition. Does anything recover at all?

PROJECT.md section 9's first HNN rung, deliberately favourable: all neurons
observed, true drive supplied, sustained recording, default physiological
coupling. Its purpose is not realism. It answers one question before the
experiment grid is built --

    is there ANY HNN regime where this estimator sees biophysical coupling?

If recovery is near zero even here, that is worth knowing in week four rather
than month four (section 10), and the project pivots to characterising why.

Metrics are per connection class, never pooled: excitatory and inhibitory weight
ranges are disjoint, so a pooled rank metric scores E-vs-I and nothing else
(section G), and classes sharing a lamtha differ in A_weight (section H).

    python experiments/hnn_first.py --tstop 30000 --n-procs 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurotwinbench import fit as fitmod
from neurotwinbench import metrics as met
from neurotwinbench import simulate as simmod
from neurotwinbench.ground_truth import extract_ground_truth


def class_matrix(gt, gids: np.ndarray) -> np.ndarray:
    """(n, n) connection-class label per pair, empty string where unconnected."""
    index = {int(g): i for i, g in enumerate(gids)}
    out = np.full((len(gids), len(gids)), "", dtype=object)
    for s, t, name in zip(gt.src_gid, gt.target_gid, gt.connection_class):
        i, j = index.get(int(s)), index.get(int(t))
        if i is not None and j is not None:
            out[i, j] = name
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tstop", type=float, default=30000.0, help="ms")
    p.add_argument("--mesh", nargs=2, type=int, default=(5, 5))
    p.add_argument("--history-ms", type=float, default=25.0)
    p.add_argument("--n-basis", type=int, default=8)
    p.add_argument("--n-procs", type=int, default=4)
    p.add_argument("--drive-rate", type=float, default=10.0)
    p.add_argument("--regularizer", type=float, default=1e-6)
    p.add_argument("--max-iter", type=int, default=2000)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()

    config = simmod.SimulationConfig(
        mesh_shape=tuple(args.mesh), tstop_ms=args.tstop,
        drive_rate_hz=args.drive_rate, event_seed=args.seed,
    )

    print(f"simulating {args.tstop/1000:g}s on {args.mesh} ...", flush=True)
    t0 = time.time()
    sim = simmod.run(config, n_procs=args.n_procs)
    sim_s = time.time() - t0
    rates = sim.rates_hz
    power = fitmod.power_check(sim.spikes, args.n_basis, n_exog=1)
    ratio = power["spikes_per_predictor"]
    if ratio < 16:
        print(f"  WARNING: {ratio:.1f} spikes per predictor. The synthetic run that "
              f"reached rho +0.80 had 16.1. Below that, a weak result cannot be "
              f"distinguished from insufficient data.", flush=True)
    print(f"  {sim_s:.0f}s wall ({sim_s/(args.tstop/1000):.1f} per simulated second) | "
          f"{sim.spikes.sum()} spikes | {rates.mean():.1f} Hz mean | "
          f"{int((rates == 0).sum())}/{len(rates)} silent | "
          f"{ratio:.1f} spikes/predictor", flush=True)

    # Ground truth comes from rebuilding the network, which does not simulate.
    gt = extract_ground_truth(simmod.build_network(config))
    w_true, d_true, connected = gt.to_matrices(sim.gids)
    classes = class_matrix(gt, sim.gids)

    print("fitting population GLM with true drive (condition A) ...", flush=True)
    t0 = time.time()
    result = fitmod.fit_population_glm(
        sim.spikes, bin_s=sim.bin_s, history_ms=args.history_ms,
        n_basis=args.n_basis, exogenous=sim.drive_spikes,
        regularizer_strength=args.regularizer, max_iter=args.max_iter,
    )
    print(f"  {time.time() - t0:.0f}s", flush=True)

    overall = met.evaluate(
        w_true, result.areas, connected & result.evaluable,
        true_delay_ms=d_true, inferred_latency_ms=result.latency_ms,
        history_ms=args.history_ms,
    )

    rows = []
    print()
    print(f"{'connection class':<44}{'n':>6}{'in win':>8}{'rho':>9}{'sign':>8}")
    print("-" * 75)
    for name in sorted({c for c in classes.ravel() if c}):
        mask = connected & (classes == name) & result.evaluable
        r = met.evaluate(
            w_true, result.areas, mask,
            true_delay_ms=d_true, history_ms=args.history_ms,
        )
        if r.n_evaluated == 0:
            continue
        rows.append((name, r))
        print(f"{name:<44.44}{r.n_evaluated:>6}{r.fraction_representable:>7.0%}"
              f"{r.weight_spearman:>+9.3f}{r.sign_accuracy:>7.0%}")

    print("-" * 75)
    print(f"{'POOLED (reported only as a warning)':<44}{overall.n_evaluated:>6}"
          f"{overall.fraction_representable:>7.0%}{overall.weight_spearman:>+9.3f}"
          f"{overall.sign_accuracy:>7.0%}")
    print("\nThe pooled row is NOT a result: E and I weight ranges are disjoint, so")
    print("it ranks excitatory against inhibitory rather than measuring recovery.")

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"hnn_first_{args.mesh[0]}x{args.mesh[1]}_t{args.tstop:g}.json"
    out.write_text(json.dumps({
        "config": vars(args) | {"out": str(args.out), "mesh": list(args.mesh)},
        "sim_wall_s": sim_s,
        "n_spikes": int(sim.spikes.sum()),
        "mean_rate_hz": float(rates.mean()),
        "n_silent": int((rates == 0).sum()),
        "per_class": {
            name: {
                "n_evaluated": r.n_evaluated,
                "n_outside_support": r.n_outside_support,
                "weight_spearman": r.weight_spearman,
                "sign_accuracy": r.sign_accuracy,
                "latency_error_ms": r.latency_error_ms,
            } for name, r in rows
        },
        "pooled_warning_only": {
            "weight_spearman": overall.weight_spearman,
            "sign_accuracy": overall.sign_accuracy,
            "n_evaluated": overall.n_evaluated,
        },
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
