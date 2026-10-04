"""PROJECT.md section 6: what latency differences can this pipeline resolve?

Generates synthetic coupling at known lags, pushes them through the identical
binning, history window, basis, fit and latency estimator used on HNN data, and
reports the smallest between-condition difference that survives.

This gates every latency claim in the project. Until it runs, reporting an
"effective interaction latency" to 0.1 ms is unlicensed -- and hnn_first.py was
already printing latency numbers before this existed.

    python experiments/latency_calibration.py --duration 300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurotwinbench import fit as fitmod
from neurotwinbench import synthetic


def coupling_at_lag(
    n_neurons: int, n_basis: int, kernels: np.ndarray, lag_bins: int,
    strength: float, rng: np.random.Generator, connection_prob: float,
) -> np.ndarray:
    """Coupling whose filters peak at `lag_bins`, via the basis nearest that lag."""
    peaks = kernels.argmax(axis=0)
    which = int(np.argmin(np.abs(peaks - lag_bins)))

    mask = rng.random((n_neurons, n_neurons)) < connection_prob
    np.fill_diagonal(mask, False)
    coupling = np.zeros((n_neurons, n_neurons, n_basis))
    coupling[:, :, which] = strength * mask
    return coupling


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--duration", type=float, default=300.0)
    p.add_argument("--neurons", type=int, default=20)
    p.add_argument("--history-ms", type=float, default=25.0)
    p.add_argument("--n-basis", type=int, default=8)
    p.add_argument("--strength", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()

    bin_s = 0.001
    window = int(round(args.history_ms / (bin_s * 1000)))
    kernels = synthetic.raised_cosine_log(args.n_basis, window)

    # Only lags a basis function actually peaks at are testable: the basis is
    # log-spaced, so resolution is finer early and coarser late by construction.
    peaks = sorted(set(kernels.argmax(axis=0).tolist()))
    print(f"basis peaks at lags (ms): {peaks}")
    print("log spacing means resolution is finer at short lags by design\n")

    rows = []
    for lag in peaks:
        rng = np.random.default_rng(args.seed)
        coupling = coupling_at_lag(
            args.neurons, args.n_basis, kernels, lag, args.strength, rng, 0.3
        )
        is_inhibitory = np.zeros(args.neurons, dtype=bool)
        net = synthetic.SyntheticNetwork(
            spikes=_simulate_with(coupling, kernels, args, bin_s, window),
            coupling=coupling, baseline=np.full(args.neurons, np.log(8.0 * bin_s)),
            kernels=kernels, is_inhibitory=is_inhibitory, bin_s=bin_s,
        )
        connected = net.filters.sum(axis=2) != 0.0
        result = fitmod.fit_population_glm(
            net.spikes, bin_s=bin_s, history_ms=args.history_ms,
            n_basis=args.n_basis, regularizer_strength=1e-3,
        )
        recovered = result.latency_ms[connected]
        rows.append((lag, float(np.median(recovered)), float(np.std(recovered))))
        print(f"true lag {lag:>3} ms -> recovered median {rows[-1][1]:>6.2f} ms "
              f"(sd {rows[-1][2]:>5.2f}, n={int(connected.sum())})")

    print()
    print("resolvability between adjacent testable lags:")
    resolvable = []
    for (l0, m0, s0), (l1, m1, s1) in zip(rows, rows[1:]):
        separation = abs(m1 - m0)
        noise = np.hypot(s0, s1)
        ok = separation > noise
        resolvable.append((l1 - l0, ok))
        print(f"  {l0:>3} vs {l1:>3} ms (true gap {l1-l0:>2} ms): "
              f"recovered gap {separation:>5.2f} ms vs noise {noise:>5.2f} ms "
              f"-> {'RESOLVED' if ok else 'NOT resolved'}")

    smallest = min((gap for gap, ok in resolvable if ok), default=None)
    print()
    if smallest is None:
        print("VERDICT: no adjacent lag pair is resolvable. Do not report latency "
              "differences at all at this configuration.")
    else:
        print(f"VERDICT: smallest resolved separation is {smallest} ms. Latency "
              f"differences below this must not be claimed.")

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"latency_calibration_n{args.neurons}_t{args.duration:g}.json"
    out.write_text(json.dumps({
        "config": vars(args) | {"out": str(args.out)},
        "basis_peak_lags_ms": peaks,
        "per_lag": [
            {"true_lag_ms": l, "recovered_median_ms": m, "recovered_sd_ms": s}
            for l, m, s in rows
        ],
        "smallest_resolved_separation_ms": smallest,
    }, indent=2))
    print(f"\nwrote {out}")


def _simulate_with(coupling, kernels, args, bin_s, window) -> np.ndarray:
    """Run the generator with a supplied coupling matrix."""
    rng = np.random.default_rng(args.seed + 1)
    n_bins = int(round(args.duration / bin_s))
    baseline = np.full(args.neurons, np.log(8.0 * bin_s))
    spikes = np.zeros((n_bins, args.neurons), dtype=np.int16)
    ceiling = np.log(200.0 * bin_s)

    for t in range(1, n_bins):
        lo = max(0, t - window)
        history = spikes[lo:t][::-1]
        features = kernels[: history.shape[0]].T @ history
        log_rate = baseline + np.einsum("ki,ijk->j", features, coupling)
        spikes[t] = rng.poisson(np.exp(np.minimum(log_rate, ceiling)))
    return spikes


if __name__ == "__main__":
    main()
