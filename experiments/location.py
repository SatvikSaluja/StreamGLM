"""PROJECT.md section 14, rebuilt: is dendritic location recoverable from spikes?

The default model cannot answer this. Every L2pyr->L5pyr pair there carries a
proximal *and* a distal synapse, so the single coupling filter a GLM estimates
per ordered pair is their sum. That contrast was retracted.

This runs the purpose-built network instead (`network="location"`), where the
L2 pyramidal population is split in half: one half projects onto `basal_2`, the
other onto `apical_tuft` of the same L5 targets, with identical receptor, weight
and lamtha. Each pair has exactly one location and the groups differ in nothing
else, so a difference between them is a location effect.

Two questions, and the second is the interesting one:

  1. Does recovery differ between proximal and distal targets?
  2. Does *effective interaction latency* differ? Distal input is filtered by
     more dendrite before reaching the soma, so it should arrive later and more
     smeared than its physical delay implies -- exactly the transformation
     section 15 names as the object of study.

    python experiments/location.py --tstop 300000 --n-procs 4
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
from neurotwinbench.simulate import LOCATION_SECTIONS


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tstop", type=float, default=300000.0, help="ms")
    p.add_argument("--mesh", nargs=2, type=int, default=(5, 5))
    p.add_argument("--seeds", nargs="+", type=int, default=[2])
    p.add_argument("--history-ms", type=float, default=25.0)
    p.add_argument("--n-basis", type=int, default=8)
    p.add_argument("--n-procs", type=int, default=4)
    p.add_argument("--regularizer", type=float, default=1e-3)
    p.add_argument("--max-iter", type=int, default=2000)
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()

    records = []
    for seed in args.seeds:
        config = simmod.SimulationConfig(
            mesh_shape=tuple(args.mesh), tstop_ms=args.tstop,
            event_seed=seed, network="location",
        )
        print(f"\n=== seed {seed} ===", flush=True)
        t0 = time.time()
        sim = simmod.run(config, n_procs=args.n_procs)
        rates = sim.rates_hz
        power = fitmod.power_check(sim.spikes, args.n_basis, n_exog=1)
        ratio = power["spikes_per_predictor"]
        print(f"simulated in {time.time()-t0:.0f}s | {sim.spikes.sum()} spikes | "
              f"{rates.mean():.1f} Hz | {int((rates==0).sum())}/{len(rates)} silent "
              f"| {ratio:.1f} spikes/predictor", flush=True)
        if ratio < 16:
            print(f"  WARNING: {ratio:.1f} spikes/predictor vs 16.1 for the "
                  f"synthetic run that reached rho +0.80", flush=True)

        gt = extract_ground_truth(simmod.build_network(config))
        w_true, d_true, connected = gt.to_matrices(sim.gids)
        classes = _class_matrix(gt, sim.gids)

        result = fitmod.fit_population_glm(
            sim.spikes, bin_s=sim.bin_s, history_ms=args.history_ms,
            n_basis=args.n_basis, exogenous=sim.drive_spikes,
            regularizer_strength=args.regularizer, max_iter=args.max_iter,
        )

        print(f"\n{'location':<14}{'n':>6}{'rho':>9}{'sign':>8}"
              f"{'true delay':>12}{'eff latency':>13}{'shift':>9}")
        print("-" * 72)
        for section in LOCATION_SECTIONS:
            name = f"L2_pyramidal->L5_pyramidal|ampa|{section}"
            mask = connected & (classes == name) & result.evaluable
            if not mask.any():
                continue
            r = met.evaluate(
                w_true, result.areas, mask, true_delay_ms=d_true,
                inferred_latency_ms=result.latency_ms, history_ms=args.history_ms,
            )
            true_delay = float(np.median(d_true[mask]))
            eff = float(np.median(result.latency_ms[mask]))
            records.append({
                "seed": seed, "section": section, "n": r.n_evaluated,
                "weight_spearman": r.weight_spearman,
                "sign_accuracy": r.sign_accuracy,
                "median_true_delay_ms": true_delay,
                "median_effective_latency_ms": eff,
                "latency_shift_ms": eff - true_delay,
            })
            print(f"{section:<14}{r.n_evaluated:>6}{r.weight_spearman:>+9.3f}"
                  f"{r.sign_accuracy:>7.0%}{true_delay:>11.2f}ms{eff:>11.2f}ms"
                  f"{eff-true_delay:>+8.2f}ms")

    _report(records)
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"location_{args.mesh[0]}x{args.mesh[1]}_t{args.tstop:g}.json"
    out.write_text(json.dumps(
        {"config": vars(args) | {"out": str(args.out), "mesh": list(args.mesh)},
         "records": records}, indent=2))
    print(f"\nwrote {out}")


def _class_matrix(gt, gids: np.ndarray) -> np.ndarray:
    index = {int(g): i for i, g in enumerate(gids)}
    out = np.full((len(gids), len(gids)), "", dtype=object)
    for s, t, name in zip(gt.src_gid, gt.target_gid, gt.connection_class):
        i, j = index.get(int(s)), index.get(int(t))
        if i is not None and j is not None:
            out[i, j] = name
    return out


def _report(records: list[dict]) -> None:
    if not records:
        return
    print("\n" + "=" * 72)
    prox = [r for r in records if r["section"] == LOCATION_SECTIONS[0]]
    dist = [r for r in records if r["section"] == LOCATION_SECTIONS[1]]
    if not (prox and dist):
        return

    d_rho = np.mean([r["weight_spearman"] for r in dist]) - \
            np.mean([r["weight_spearman"] for r in prox])
    d_lat = np.mean([r["latency_shift_ms"] for r in dist]) - \
            np.mean([r["latency_shift_ms"] for r in prox])
    print(f"distal minus proximal:  rho {d_rho:+.3f}   latency shift {d_lat:+.2f} ms")
    print("\nBoth groups have identical weights, receptor and lamtha, and each pair")
    print("has exactly one location, so any difference is attributable to dendritic")
    print("placement rather than to connection class. With one seed these are")
    print("point estimates: run several seeds before reading them as an effect.")


if __name__ == "__main__":
    main()
