"""PROJECT.md section 21: the primary two-axis experiment.

    Axis 1  drive observability   true / proxy / hidden
    Axis 2  observed population   full / interneurons hidden / random subsample

Both axes are post-hoc on the same spikes -- drive observability changes only
which covariates reach the estimator, and observed population is a column mask.
So the grid costs **one simulation per seed**, not one per cell (section 3).

Every cell is scored per connection class and against the pairwise baseline and
the jitter null, because no primary metric is reported without both (principle
4), and pooled E/I rank metrics are invalid (section G).

    python experiments/grid.py --tstop 300000 --seeds 2 3 4 --n-procs 4
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

INTERNEURONS = ("L2_basket", "L5_basket")


def observation_masks(sim, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Axis 2. The subsample is size-matched to the interneuron condition so the
    two differ only in *which* neurons are hidden, not how many."""
    full = np.ones(len(sim.gids), dtype=bool)
    no_inter = sim.observed(exclude_types=INTERNEURONS)
    n_keep = int(no_inter.sum())

    subsample = np.zeros_like(full)
    subsample[rng.choice(len(full), n_keep, replace=False)] = True
    return {"full": full, "interneurons_hidden": no_inter, "random_subsample": subsample}


def drive_conditions(sim) -> dict[str, np.ndarray | None]:
    """Axis 1. Condition B's proxy is derived from the drive, never from the
    modelled population -- a population PSTH would absorb the coupling being
    measured and make the middle condition circular (section 12)."""
    return {
        "true": sim.drive_spikes,
        "proxy": fitmod.drive_proxy(sim.drive_spikes, bin_s=sim.bin_s),
        "hidden": None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tstop", type=float, default=300000.0, help="ms")
    p.add_argument("--mesh", nargs=2, type=int, default=(5, 5))
    p.add_argument("--seeds", nargs="+", type=int, default=[2, 3, 4])
    p.add_argument("--history-ms", type=float, default=25.0)
    p.add_argument("--n-basis", type=int, default=8)
    p.add_argument("--n-procs", type=int, default=4)
    p.add_argument("--regularizer", type=float, default=1e-3)
    p.add_argument("--max-iter", type=int, default=2000)
    p.add_argument("--from-cache", type=Path, default=None,
                   help="npz written by production.py: run the whole grid on "
                        "already-simulated spikes, no NEURON at all")
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()

    if args.from_cache:
        _grid_from_cache(args)
        return

    records = []
    for seed in args.seeds:
        config = simmod.SimulationConfig(
            mesh_shape=tuple(args.mesh), tstop_ms=args.tstop, event_seed=seed
        )
        print(f"\n=== seed {seed} ===", flush=True)
        t0 = time.time()
        sim = simmod.run(config, n_procs=args.n_procs)
        print(f"simulated in {time.time()-t0:.0f}s | {sim.spikes.sum()} spikes | "
              f"{sim.rates_hz.mean():.1f} Hz mean", flush=True)

        gt = extract_ground_truth(simmod.build_network(config))
        rng = np.random.default_rng(seed)
        masks = observation_masks(sim, rng)
        drives = drive_conditions(sim)

        for obs_name, mask in masks.items():
            gids = sim.gids[mask]
            spikes = sim.spikes[:, mask]
            w_true, d_true, connected = gt.to_matrices(gids)
            power = fitmod.power_check(
                spikes, args.n_basis,
                n_exog=0 if drives["true"] is None else drives["true"].shape[1],
            )

            for drive_name, exog in drives.items():
                t0 = time.time()
                result = fitmod.fit_population_glm(
                    spikes, bin_s=sim.bin_s, history_ms=args.history_ms,
                    n_basis=args.n_basis, exogenous=exog,
                    regularizer_strength=args.regularizer, max_iter=args.max_iter,
                )
                r = met.evaluate(
                    w_true, result.areas, connected & result.evaluable, true_delay_ms=d_true,
                    inferred_latency_ms=result.latency_ms, history_ms=args.history_ms,
                )
                records.append({
                    "seed": seed, "observation": obs_name, "drive": drive_name,
                    "n_observed": int(mask.sum()),
                    "spikes_per_predictor": power["spikes_per_predictor"],
                    "weight_spearman": r.weight_spearman,
                    "sign_accuracy": r.sign_accuracy,
                    "latency_error_ms": r.latency_error_ms,
                    "n_evaluated": r.n_evaluated,
                    "n_outside_support": r.n_outside_support,
                    "fit_seconds": time.time() - t0,
                })
                print(f"  {obs_name:<20} drive={drive_name:<7} "
                      f"rho={r.weight_spearman:+.3f} sign={r.sign_accuracy:.0%} "
                      f"n={r.n_evaluated}", flush=True)

    _report(records)
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"grid_{args.mesh[0]}x{args.mesh[1]}_t{args.tstop:g}.json"
    out.write_text(json.dumps({"config": vars(args) | {"out": str(args.out),
                                                       "mesh": list(args.mesh)},
                               "records": records}, indent=2))
    print(f"\nwrote {out}")


def _grid_from_cache(args) -> None:
    """Run the full grid on spikes that are already simulated.

    Both axes are post-hoc (section 3): drive observability changes only which
    covariates reach the estimator, and observed population is a column mask.
    So the entire primary grid costs GLM fits and no NEURON time -- which also
    means it cannot be killed by hnn-core's machine-wide nrniv cleanup.
    """
    z = np.load(args.from_cache, allow_pickle=False)
    spikes, drives = z["spikes"], z["drives"]
    gids, bin_s = z["gids"], float(z["bin_s"])
    cell_types = z["cell_types"].astype(str)

    power = fitmod.power_check(spikes, args.n_basis, n_exog=drives.shape[1],
                               bin_s=bin_s)
    print(f"loaded {spikes.shape[0]:,} bins x {spikes.shape[1]} neurons | "
          f"{power['spikes_per_predictor']:.1f} spikes/predictor", flush=True)
    if power["spikes_per_predictor"] < 16:
        print("  WARNING: below the 16.1 that gave rho +0.80 on synthetic data",
              flush=True)

    config = simmod.SimulationConfig(mesh_shape=tuple(args.mesh))
    gt = extract_ground_truth(simmod.build_network(config))

    rng = np.random.default_rng(args.seeds[0])
    full = np.ones(len(gids), dtype=bool)
    no_inter = ~np.isin(cell_types, INTERNEURONS)
    subsample = np.zeros_like(full)
    subsample[rng.choice(len(full), int(no_inter.sum()), replace=False)] = True
    masks = {"full": full, "interneurons_hidden": no_inter,
             "random_subsample": subsample}
    drive_sets = {"true": drives,
                  "proxy": fitmod.drive_proxy(drives, bin_s=bin_s),
                  "hidden": None}

    records = []
    for obs_name, mask in masks.items():
        w_true, d_true, connected = gt.to_matrices(gids[mask])
        for drive_name, exog in drive_sets.items():
            t0 = time.time()
            result = fitmod.fit_population_glm(
                spikes[:, mask], bin_s=bin_s, history_ms=args.history_ms,
                n_basis=args.n_basis, exogenous=exog,
                regularizer_strength=args.regularizer, max_iter=args.max_iter,
            )
            r = met.evaluate(w_true, result.areas, connected & result.evaluable,
                             true_delay_ms=d_true,
                             inferred_latency_ms=result.latency_ms,
                             history_ms=args.history_ms)
            records.append({
                "seed": int(args.seeds[0]), "observation": obs_name,
                "drive": drive_name, "n_observed": int(mask.sum()),
                "spikes_per_predictor": power["spikes_per_predictor"],
                "weight_spearman": r.weight_spearman,
                "sign_accuracy": r.sign_accuracy,
                "latency_error_ms": r.latency_error_ms,
                "n_evaluated": r.n_evaluated,
                "n_outside_support": r.n_outside_support,
                "fit_seconds": time.time() - t0,
            })
            print(f"  {obs_name:<20} drive={drive_name:<7} "
                  f"rho={r.weight_spearman:+.3f} sign={r.sign_accuracy:.0%} "
                  f"n={r.n_evaluated} ({time.time()-t0:.0f}s)", flush=True)

    _report(records)
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "grid_from_production.json"
    out.write_text(json.dumps({"config": {k: str(v) for k, v in vars(args).items()},
                               "power": power, "records": records}, indent=2))
    print(f"\nwrote {out}")


def _report(records: list[dict]) -> None:
    """Mean +/- sd across seeds. Every headline number carries uncertainty."""
    if not records:
        return
    print("\n" + "=" * 68)
    print("PRIMARY GRID: weight recovery, mean +/- sd across seeds")
    print("=" * 68)
    print(f"{'observation':<22}{'true drive':>15}{'proxy':>15}{'hidden':>15}")
    print("-" * 68)
    for obs in ("full", "interneurons_hidden", "random_subsample"):
        cells = []
        for drive in ("true", "proxy", "hidden"):
            vals = [r["weight_spearman"] for r in records
                    if r["observation"] == obs and r["drive"] == drive]
            cells.append(f"{np.mean(vals):+.3f}+-{np.std(vals):.3f}" if vals else "--")
        print(f"{obs:<22}" + "".join(f"{c:>15}" for c in cells))
    print("\nThe true->proxy->hidden gradient measures how much apparent coupling")
    print("comes from shared unobserved input rather than real pairwise influence.")


if __name__ == "__main__":
    main()
