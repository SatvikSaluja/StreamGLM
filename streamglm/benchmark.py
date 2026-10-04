"""One disk-backed objective/gradient pass, not time to fit.

Run each configuration in a fresh process so peak RSS is meaningful.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import tempfile
import time
import jax
import jax.numpy as jnp
import numpy as np
from .data import Recording
from .model import StreamingGLM
from .pipeline import exponential_basis
from .reference import features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=int, default=8192)
    parser.add_argument("--neurons", type=int, default=128)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--n-basis", type=int, default=3)
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--implementation", choices=["streamed", "materialized"], default="streamed")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    basis = exponential_basis(args.history, args.n_basis)
    design_bytes = args.bins * args.neurons * args.n_basis * 8
    if args.implementation == "materialized" and design_bytes > 512*2**20:
        raise ValueError("materialized benchmark capped at 512 MiB raw design; use smaller data")
    model = StreamingGLM(basis, args.chunk, args.rank)
    params = model.initialize(args.neurons)
    with tempfile.TemporaryDirectory(prefix="streamglm-bench-") as folder:
        path = Path(folder) / "counts.npy"
        data = np.lib.format.open_memmap(path, mode="w+", dtype=np.int16,
                                        shape=(args.bins, args.neurons))
        rng = np.random.default_rng(12)
        for start in range(0, args.bins, args.chunk):
            stop = min(start + args.chunk, args.bins)
            data[start:stop] = rng.poisson(.05, (stop-start, args.neurons))
        data.flush(); del data
        recording = Recording.from_npy(path)
        t = time.perf_counter()
        if args.implementation == "streamed":
            evaluate = lambda: model.value_and_grad(params, recording)
        else:
            x, valid = features(recording.counts, basis)
            design = jnp.asarray(x[valid]); target = jnp.asarray(recording.counts[valid])
            def loss(p, x, y):
                projected = jnp.einsum("tki,kir->tkr", x, p["U"])
                eta = jnp.einsum("tkr,kjr->tj", projected, p["V"])+p["b"]
                return jnp.sum(jnp.exp(eta)-y*eta)/len(y)
            compiled = jax.jit(jax.value_and_grad(loss))
            def evaluate():
                value, grad = compiled(params, design, target)
                return float(value), jax.tree.map(np.asarray, grad)
        value, grad = evaluate()
        cold = time.perf_counter() - t
        t = time.perf_counter()
        value, grad = evaluate()
        warm = time.perf_counter() - t
    result = vars(args) | {"out": str(args.out), "dtype": "float64",
        "cold_pass_seconds": cold, "warm_pass_seconds": warm,
        "objective": value, "gradient_inf_norm": max(float(np.max(np.abs(g))) for g in grad.values()),
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "hypothetical_design_gib": design_bytes / 2**30,
        "parameter_count": sum(p.size for p in params.values()),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "platform": platform.platform(), "jax_version": importlib.metadata.version("jax"),
        "note": "Whole-process Linux peak RSS, including JAX, compilation and mapped file pages. Two exact passes; not a fit."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
