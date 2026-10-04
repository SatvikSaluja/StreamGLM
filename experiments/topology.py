"""PROJECT.md section 18: can real edges be ranked above non-edges?

Only answerable in a sparse network. The default model connects nearly every
eligible pair, so there are no designed non-edges and "edge detection" is not a
detection task at all -- `edge_detection` returns nan there rather than a
flattering number computed against an empty negative class.

Thinning also switches on the second uncertainty axis of section 20: with
`probability < 1` the `conn_seed` finally does something, so circuit
realisations become a source of variation distinct from drive events. This
script varies both, and reports them separately because pooling them would
present anatomical variability and trial variability as one error bar.

    python experiments/topology.py --probability 0.3 --conn-seeds 3 4 --n-procs 4
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

#: Thin only the excitatory recurrent projections. Inhibitory classes have
#: almost no weight spread (section F) and thinning them would remove the
#: inhibition that keeps the network firing at all (section 3's drive sweep).
THINNED = ("L2_pyramidal->L2_pyramidal", "L5_pyramidal->L5_pyramidal")


def candidate_mask(gt, gids: np.ndarray, thinned: tuple[str, ...]) -> np.ndarray:
    """Pairs some thinned class could have produced, connected or not.

    Without this the negative class is dominated by pairs the model never
    proposed -- L5->L2, cross-type combinations that no connection spec covers --
    and the estimator would be credited for rejecting connections that were
    never on the table.
    """
    index = {int(g): i for i, g in enumerate(gids)}
    types = np.empty(len(gids), dtype=object)
    types[:] = ""
    # Resolve from BOTH source and target rows. Using source rows alone leaves a
    # target-only cell typed "", dropping its columns from the candidate set and
    # silently shrinking the negative class -- which would inflate every
    # detection score. No cell is target-only in the default model, so this is
    # latent there, but purpose-built networks can and do have them.
    for gid_array, type_array in ((gt.src_gid, gt.src_type),
                                  (gt.target_gid, gt.target_type)):
        for gid, cell_type in zip(gid_array, type_array):
            i = index.get(int(gid))
            if i is not None and not types[i]:
                types[i] = cell_type
    if any(t == "" for t in types):
        raise ValueError(
            f"{sum(1 for t in types if not t)} observed cells appear in no "
            f"connection; their type cannot be resolved and the candidate mask "
            f"would be wrong"
        )

    mask = np.zeros((len(gids), len(gids)), dtype=bool)
    for spec in thinned:
        src_type, target_type = spec.split("->")
        rows = np.array([t == src_type for t in types])
        cols = np.array([t == target_type for t in types])
        mask |= rows[:, None] & cols[None, :]
    np.fill_diagonal(mask, False)
    return mask


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tstop", type=float, default=300000.0, help="ms")
    p.add_argument("--mesh", nargs=2, type=int, default=(5, 5))
    p.add_argument("--probability", type=float, default=0.3)
    p.add_argument("--conn-seeds", nargs="+", type=int, default=[3, 4],
                   help="circuit realisations (section 20)")
    p.add_argument("--event-seeds", nargs="+", type=int, default=[2],
                   help="drive-event realisations on each circuit")
    p.add_argument("--history-ms", type=float, default=25.0)
    p.add_argument("--n-basis", type=int, default=8)
    p.add_argument("--n-procs", type=int, default=4)
    p.add_argument("--regularizer", type=float, default=1e-3)
    p.add_argument("--max-iter", type=int, default=2000)
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()

    records = []
    for conn_seed in args.conn_seeds:
        for event_seed in args.event_seeds:
            config = simmod.SimulationConfig(
                mesh_shape=tuple(args.mesh), tstop_ms=args.tstop,
                event_seed=event_seed, conn_seed=conn_seed,
                connection_probability={s: args.probability for s in THINNED},
            )
            print(f"\n=== circuit {conn_seed}, events {event_seed} ===", flush=True)
            t0 = time.time()
            sim = simmod.run(config, n_procs=args.n_procs)
            rates = sim.rates_hz
            power = fitmod.power_check(sim.spikes, args.n_basis, n_exog=1,
                                       bin_s=sim.bin_s)
            ratio = power["spikes_per_predictor"]
            print(f"simulated in {time.time()-t0:.0f}s | {sim.spikes.sum()} spikes | "
                  f"{rates.mean():.1f} Hz | {int((rates==0).sum())}/{len(rates)} silent "
                  f"| {ratio:.1f} spikes/predictor", flush=True)
            if ratio < 16:
                print(f"  WARNING: {ratio:.1f} spikes/predictor vs 16.1 for the "
                      f"synthetic run that reached rho +0.80", flush=True)

            gt = extract_ground_truth(simmod.build_network(config))
            _, _, connected = gt.to_matrices(sim.gids)
            candidates = candidate_mask(gt, sim.gids, THINNED)

            result = fitmod.fit_population_glm(
                sim.spikes, bin_s=sim.bin_s, history_ms=args.history_ms,
                n_basis=args.n_basis, exogenous=sim.drive_spikes,
                regularizer_strength=args.regularizer, max_iter=args.max_iter,
            )
            scored = candidates & result.evaluable
            detection = met.edge_detection(connected, result.areas, candidate=scored)
            records.append({"conn_seed": conn_seed, "event_seed": event_seed,
                            "spikes_per_predictor": ratio, **detection})
            print(f"  ROC-AUC {detection['roc_auc']:.3f} | PR-AUC "
                  f"{detection['pr_auc']:.3f} (chance = prevalence "
                  f"{detection['prevalence']:.3f}) | {detection['n_edges']} edges, "
                  f"{detection['n_non_edges']} non-edges", flush=True)

    _report(records)
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"topology_p{args.probability:g}_t{args.tstop:g}.json"
    out.write_text(json.dumps(
        {"config": vars(args) | {"out": str(args.out), "mesh": list(args.mesh)},
         "thinned": list(THINNED), "records": records}, indent=2))
    print(f"\nwrote {out}")


def _report(records: list[dict]) -> None:
    if not records:
        return
    roc = [r["roc_auc"] for r in records if not np.isnan(r["roc_auc"])]
    if not roc:
        print("\nNo scoreable records: the network was not sparse enough.")
        return

    print("\n" + "=" * 62)
    print(f"ROC-AUC  {np.mean(roc):.3f} +- {np.std(roc):.3f}   (chance 0.5)")
    prevalence = np.mean([r["prevalence"] for r in records])
    pr = [r["pr_auc"] for r in records if not np.isnan(r["pr_auc"])]
    print(f"PR-AUC   {np.mean(pr):.3f} +- {np.std(pr):.3f}   "
          f"(chance = prevalence {prevalence:.3f})")

    # Variance decomposition only means something with several circuits.
    circuits = {r["conn_seed"] for r in records}
    if len(circuits) > 1:
        between = np.std([np.mean([r["roc_auc"] for r in records
                                   if r["conn_seed"] == c]) for c in circuits])
        print(f"\nbetween-circuit sd {between:.3f} -- anatomical variability, "
              f"distinct from drive variability (section 20)")


if __name__ == "__main__":
    main()
