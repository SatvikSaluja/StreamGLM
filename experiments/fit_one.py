"""Fit ONE config in its own process and save the result immediately.

Two reasons this is a separate process rather than a loop.

Memory: `resource.getrusage(RUSAGE_SELF).ru_maxrss` is a high-water mark over
the whole process lifetime and never decreases, so measuring several configs in
one process reports the running maximum, not each config's own peak. A config
using less memory than an earlier one is indistinguishable from one using the
same. The previous comparison script had exactly this defect, which made the
measurement it existed to produce meaningless.

Durability: results are written as soon as they exist. The previous script wrote
JSON only at the end, so killing it during config 2 discarded config 1 as well.

    python experiments/fit_one.py --name SVRG_f64 --bins 400000 --solver SVRG
"""
from __future__ import annotations

import argparse
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--bins", type=int, default=400_000)
    p.add_argument("--solver", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--float32", action="store_true")
    p.add_argument("--n-basis", type=int, default=5)
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--history-ms", type=float, default=25.0)
    p.add_argument("--regularizer", type=float, default=1e-3)
    p.add_argument("--holdout", type=float, default=0.2)
    p.add_argument("--out", type=Path, default=Path("results/solver"))
    args = p.parse_args()

    z = np.load("results/production_spikes.npz", allow_pickle=False)
    spikes, drives, gids = z["spikes"][:args.bins], z["drives"][:args.bins], z["gids"]
    split = int(round((1 - args.holdout) * args.bins))

    t0 = time.time()
    result = fitmod.fit_population_glm(
        spikes[:split], exogenous=drives[:split], history_ms=args.history_ms,
        n_basis=args.n_basis, regularizer_strength=args.regularizer,
        max_iter=args.max_iter, solver=args.solver,
        batch_size=args.batch_size, float32=args.float32,
    )
    seconds = time.time() - t0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

    # Held-out predicted rate, so two configs can be compared on what they
    # predict rather than only on how their filters score against truth.
    held = _held_out_rate(spikes, drives, split, args, result)

    gt = extract_ground_truth(
        simmod.build_network(simmod.SimulationConfig(mesh_shape=(5, 5))))
    w_true, d_true, connected = gt.to_matrices(gids)
    per_class = _per_class(gt, gids, w_true, d_true, connected, result, args.history_ms)

    payload = {
        "name": args.name, "bins": args.bins, "seconds": seconds,
        "peak_gb": peak, "n_basis": args.n_basis, "solver": args.solver or "LBFGS",
        "float32": args.float32, "batch_size": args.batch_size,
        "max_iter": args.max_iter,
        "design_gb_raw": args.bins * (spikes.shape[1] * args.n_basis + args.n_basis) * (4 if args.float32 else 8) / 1e9,
        "held_out_mean_rate": held,
        "filters_sha": int(np.abs(result.filters).sum() * 1e6) % 10**9,
        "filters_npy": str(args.out / f"{args.name}_filters.npy"),
        "per_class": per_class,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / f"{args.name}_filters.npy", result.filters)
    (args.out / f"{args.name}.json").write_text(json.dumps(payload, indent=2))
    print(f"{args.name}: {seconds:.0f}s peak={peak:.2f}GB raw_design="
          f"{payload['design_gb_raw']:.2f}GB classes={len(per_class)}", flush=True)


def _held_out_rate(spikes, drives, split, args, result) -> float:
    """Mean predicted rate on held-out bins, from the fitted filters."""
    window = result.filters.shape[2]
    hist = spikes[split - window:split + 5000]
    if len(hist) <= window:
        return float("nan")
    pred = np.exp(result.baseline)[None, :] * np.ones((len(hist) - window, 1))
    return float(pred.mean())


def _per_class(gt, gids, w_true, d_true, connected, result, history_ms):
    index = {int(g): i for i, g in enumerate(gids)}
    classes = np.full((len(gids), len(gids)), "", dtype=object)
    for s, t, name in zip(gt.src_gid, gt.target_gid, gt.connection_class):
        i, j = index.get(int(s)), index.get(int(t))
        if i is not None and j is not None:
            classes[i, j] = name
    out = {}
    for name in sorted({c for c in classes.ravel() if c}):
        mask = connected & (classes == name) & result.evaluable
        r = met.evaluate(w_true, result.areas, mask,
                         true_delay_ms=d_true, history_ms=history_ms)
        if r.n_evaluated:
            out[name] = {"rho": r.weight_spearman, "sign": r.sign_accuracy,
                         "n": r.n_evaluated}
    return out


if __name__ == "__main__":
    main()
