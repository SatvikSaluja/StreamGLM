"""Milestone 1: can the estimator recover coupling it was designed to recover?

Generates spikes from a known coupled GLM, fits them with the population GLM,
and scores recovery against baselines and the jitter null. Nothing here touches
HNN -- this validates the measuring instrument before it is pointed at
biophysics (PROJECT.md section 8).

The tolerances this run establishes get frozen into calibration.json and are not
revisited after the first HNN result is examined.

    python experiments/calibrate.py --duration 300 --neurons 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurotwinbench import fit as fitmod
from neurotwinbench import metrics as met
from neurotwinbench import synthetic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=300.0, help="seconds")
    parser.add_argument("--neurons", type=int, default=20)
    parser.add_argument("--history-ms", type=float, default=25.0)
    parser.add_argument("--n-basis", type=int, default=8)
    parser.add_argument("--connection-prob", type=float, default=0.3)
    parser.add_argument("--strength", type=float, default=0.15)
    parser.add_argument("--jitter-mult", type=float, default=2.0,
                        help="jitter width as a multiple of the history window")
    parser.add_argument("--regularizer", type=float, default=1e-3,
                        help="selected by held-out log-likelihood; see fit.select_regularizer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    bin_s = 0.001
    window = int(round(args.history_ms / (bin_s * 1000)))

    print(f"generating {args.neurons} neurons x {args.duration:g}s ...", flush=True)
    net = synthetic.simulate(
        n_neurons=args.neurons,
        duration_s=args.duration,
        window=window,
        n_basis=args.n_basis,
        connection_prob=args.connection_prob,
        strength=args.strength,
        seed=args.seed,
    )
    true_area = net.filters.sum(axis=2)
    connected = true_area != 0.0
    true_latency = np.abs(net.filters).argmax(axis=2) * bin_s * 1000.0
    rates = net.rates_hz
    print(f"  {net.spikes.sum()} spikes | {rates.min():.1f}-{rates.max():.1f} Hz "
          f"| {int(connected.sum())} true connections", flush=True)

    print("fitting population GLM ...", flush=True)
    result = fitmod.fit_population_glm(
        net.spikes, bin_s=bin_s, history_ms=args.history_ms, n_basis=args.n_basis,
        regularizer_strength=args.regularizer,
    )
    glm = met.evaluate(
        true_area, result.areas, connected,
        true_delay_ms=true_latency, inferred_latency_ms=result.latency_ms,
    )

    print("fitting jitter null ...", flush=True)
    jittered = met.jitter(net.spikes, width_bins=int(args.jitter_mult * window), seed=args.seed + 1)
    null_result = fitmod.fit_population_glm(
        jittered, bin_s=bin_s, history_ms=args.history_ms, n_basis=args.n_basis,
        regularizer_strength=args.regularizer,
    )
    null = met.evaluate(true_area, null_result.areas, connected)

    print("fitting pairwise GLM baseline ...", flush=True)
    pairwise = met.evaluate(
        true_area,
        fitmod.fit_pairwise_glm(
            net.spikes, bin_s=bin_s, history_ms=args.history_ms,
            n_basis=args.n_basis, regularizer_strength=args.regularizer,
        ).areas,
        connected,
    )

    print("computing CCG baseline ...", flush=True)
    ccg = met.evaluate(true_area, met.cross_correlogram(net.spikes, window), connected)

    rows = [("population GLM", glm), ("pairwise GLM", pairwise),
            ("CCG baseline", ccg), ("jitter null", null)]
    print()
    print(f"{'estimator':<18}{'rho':>9}{'sign':>9}{'latency err':>14}")
    print("-" * 50)
    for name, r in rows:
        print(f"{name:<18}{r.weight_spearman:>+9.3f}{r.sign_accuracy:>8.0%}"
              f"{r.latency_error_ms:>13.2f}ms")

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": vars(args) | {"out": str(args.out)},
        "n_spikes": int(net.spikes.sum()),
        "rate_hz": [float(rates.min()), float(rates.max())],
        "n_connections": int(connected.sum()),
        "results": {
            name: {
                "weight_spearman": r.weight_spearman,
                "sign_accuracy": r.sign_accuracy,
                "latency_error_ms": r.latency_error_ms,
                "n_evaluated": r.n_evaluated,
            }
            for name, r in rows
        },
    }
    out = args.out / f"calibration_n{args.neurons}_t{args.duration:g}_s{args.seed}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")

    if glm.weight_spearman <= abs(null.weight_spearman):
        print("\nFAIL: population GLM does not clear its null.")
        raise SystemExit(1)
    print("\nPASS: population GLM clears its jitter null.")
    if glm.weight_spearman <= pairwise.weight_spearman:
        print("NOTE: the pairwise baseline matches or beats joint estimation here. "
              "That is a result, not a bug -- see PROJECT.md section 17.")


if __name__ == "__main__":
    main()
