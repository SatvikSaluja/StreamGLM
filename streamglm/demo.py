"""Run a bounded synthetic fit: python -m streamglm.demo --out artifacts/demo.json."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import jax
import numpy as np
from .data import Recording
from .model import StreamingGLM
from .reference import materialize_weights
from .synthetic import fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=int, default=12000)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--out", type=Path, default=Path("artifacts/demo.json"))
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    counts, basis, truth = fixture(args.bins)
    split = int(.75 * len(counts))
    train, test = Recording(counts[:split]), Recording(counts[split:])
    model = StreamingGLM(basis, chunk_size=512, rank=2, ridge=.001)
    initial = model.initialize(counts.shape[1], mean_count=counts[:split].mean(axis=0))
    fit = model.fit(train, initial, max_iter=args.max_iter, tolerance=1e-8)
    evaluation = StreamingGLM(basis, chunk_size=512, rank=2)
    null = {key: np.array(value) for key, value in initial.items()}
    null["U"][:] = 0.
    fitted_nll, _ = evaluation.value_and_grad(fit.params, test)
    null_nll, _ = evaluation.value_and_grad(null, test)
    true_nll, _ = evaluation.value_and_grad(truth, test)
    true_w, fitted_w = materialize_weights(truth), materialize_weights(fit.params)
    result = asdict(fit)
    result.pop("params")
    result.update({"n_bins": args.bins, "neurons": counts.shape[1], "rank": 2,
        "dtype": "float64", "seed": 1,
        "heldout_nll_per_bin": fitted_nll, "intercept_only_nll_per_bin": null_nll,
        "true_model_nll_per_bin": true_nll,
        "heldout_gain_nats_per_bin": null_nll - fitted_nll,
        "weight_relative_frobenius_error": float(np.linalg.norm(fitted_w-true_w)/np.linalg.norm(true_w)),
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "platform": platform.platform(),
        "versions": {p: importlib.metadata.version(p) for p in ("jax", "numpy", "scipy")},
        "interpretation": "Single small synthetic fit; no general recovery or laptop-scale claim."})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    np.savez(args.out.with_suffix(".npz"), basis=basis, **fit.params)
    print(json.dumps({k: v for k, v in result.items() if k != "trace"}, indent=2))


if __name__ == "__main__":
    main()
