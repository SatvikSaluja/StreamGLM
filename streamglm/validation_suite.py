"""Reproducible multiseed synthetic selection and optional cached HNN demo."""
import argparse
import json
from pathlib import Path
import jax
import numpy as np
from .adapters import from_hnn_cache, load_recording
from .data import Recording
from .persistence import load_bundle, write_json
from .pipeline import run_pipeline, exponential_basis
from .reference import materialize_weights
from .synthetic import fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/v2/validation"))
    parser.add_argument("--hnn", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    records = []
    for seed in args.seeds:
        y, basis, truth = fixture(20000, 8, seed=seed)
        out = args.out/f"synthetic_{seed}"
        if (out/"report.json").exists():
            report = json.loads((out/"report.json").read_text())
        else:
            report = run_pipeline(Recording(y, bin_s=.01), out, basis,
                ranks=(0, 1, 2), seeds=(0, 1), max_iter=800, chunk_size=1024, ridge=.001)
        _, fitted, _, _ = load_bundle(out/"model.npz")
        expected = materialize_weights(truth)
        estimated = materialize_weights(fitted)
        error = float(np.linalg.norm(expected-estimated)/np.linalg.norm(expected))
        record = {"seed": seed, "selected_rank": report["selection"]["selected"]["rank"],
            "weight_relative_error": error, "test": report["test"],
            "gain_over_self_history": report["test_gain_over_self_history_nats_per_bin"]}
        records.append(record)
        write_json(args.out/"synthetic_summary.json", {"records": records,
            "note": "Three small known inhibitory low-rank fixtures; not a universal power or anatomical recovery claim."})
        print(json.dumps(record), flush=True)
    if args.hnn:
        directory = Path("cache/hnn_streamglm")
        if directory.exists():
            meta = json.loads((directory/"recording.json").read_text())
            if meta["provenance"]["source"] != str(args.hnn.resolve()):
                raise ValueError("HNN conversion cache belongs to a different source")
            data = load_recording(directory)
        else:
            data = from_hnn_cache(args.hnn, directory)
        # A bounded predictive demonstration, not a redo of anatomical inference.
        part = data.select([(0, min(120000, data.epoch_lengths[0]))])
        out = args.out/"hnn_predictive"
        if not out.exists():
            run_pipeline(part, out, exponential_basis(25, 3), ranks=(0, 1, 2),
                         seeds=(0, 1), max_iter=800, chunk_size=1024, ridge=.01)


if __name__ == "__main__":
    main()
