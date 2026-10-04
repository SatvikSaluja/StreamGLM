"""Command-line entry point; use `streamglm --help`."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import jax
import numpy as np
from .adapters import load_recording, from_nwb, from_hnn_cache
from .evaluation import score
from .model import StreamingGLM
from .optim import fit_adam
from .persistence import load_bundle, save_bundle, write_json
from .pipeline import run_pipeline, exponential_basis


def _model_args(parser):
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--n-basis", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--ridge", type=float, default=.01)


def _aligned_data(path, meta):
    data = load_recording(path)
    ids = meta.get("unit_ids")
    if not ids or meta.get("bin_s") != data.bin_s:
        raise ValueError("model lacks unit metadata or bin width differs")
    if not all(key in data.unit_ids for key in ids):
        raise ValueError("recording is missing model units")
    return data.select([(0, len(data.counts))], [data.unit_ids.index(key) for key in ids])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Memory-bounded neural population GLMs")
    sub = parser.add_subparsers(dest="command", required=True)
    nwb = sub.add_parser("import-nwb", help="bin an NWB Units table into a recording directory")
    nwb.add_argument("path", type=Path); nwb.add_argument("--out", type=Path, required=True)
    nwb.add_argument("--epoch", type=float, nargs=2, action="append", required=True, metavar=("START", "STOP"))
    nwb.add_argument("--bin-s", type=float, default=.005)
    nwb.add_argument("--unit-ids", type=int, nargs="+")
    hnn = sub.add_parser("import-hnn", help="extract a cached NPZ, preserving segment boundaries")
    hnn.add_argument("path", type=Path); hnn.add_argument("--out", type=Path, required=True)
    fit = sub.add_parser("fit", help="fit a recording; save model, diagnostics and checkpoint")
    fit.add_argument("data", type=Path); fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--solver", choices=["lbfgs", "adam"], default="lbfgs")
    fit.add_argument("--rank", type=int, default=5, help="0=self-history only; -1=dense")
    fit.add_argument("--max-iter", type=int, default=300)
    fit.add_argument("--epochs", type=int, default=10)
    fit.add_argument("--learning-rate", type=float, default=.01)
    fit.add_argument("--resume", action="store_true")
    _model_args(fit)
    evaluate = sub.add_parser("evaluate", help="score saved model on an explicitly chosen recording")
    evaluate.add_argument("model", type=Path); evaluate.add_argument("data", type=Path)
    evaluate.add_argument("--out", type=Path, required=True)
    predict = sub.add_parser("predict", help="write predictions incrementally to NPY")
    predict.add_argument("model", type=Path); predict.add_argument("data", type=Path)
    predict.add_argument("--out", type=Path, required=True)
    select = sub.add_parser("pipeline", help="train/validation/test, rank selection and baseline")
    select.add_argument("data", type=Path); select.add_argument("--out", type=Path, required=True)
    select.add_argument("--ranks", type=int, nargs="+", default=[0, 1, 2])
    select.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    select.add_argument("--max-iter", type=int, default=500)
    select.add_argument("--min-rate-hz", type=float, default=.1)
    _model_args(select)
    args = parser.parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    if args.command == "import-nwb":
        data = from_nwb(args.path, args.epoch, args.bin_s, args.out, args.unit_ids)
        print(json.dumps({"bins": len(data.counts), "neurons": data.n_neurons}))
    elif args.command == "import-hnn":
        data = from_hnn_cache(args.path, args.out)
        print(json.dumps({"bins": len(data.counts), "neurons": data.n_neurons}))
    elif args.command in ("evaluate", "predict"):
        model, params, meta, _ = load_bundle(args.model)
        data = _aligned_data(args.data, meta)
        if args.command == "evaluate":
            result = score(model, params, data)
            write_json(args.out, result); print(json.dumps(result))
        else:
            import os
            import tempfile
            if args.out.exists():
                raise FileExistsError(args.out)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=args.out.parent, delete=False) as f:
                temporary = Path(f.name)
            try:
                output = np.lib.format.open_memmap(temporary, mode="w+", dtype=model.basis.dtype,
                                                   shape=(len(data.counts), data.n_neurons))
                for start, rates, valid in model.predict_chunks(params, data):
                    output[start:start+len(rates)] = rates
                output.flush(); del output
                os.replace(temporary, args.out)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
            write_json(args.out.with_suffix(".json"), {"unit_ids": data.unit_ids,
                "bin_s": data.bin_s, "epoch_lengths": data.epoch_lengths,
                "values": "expected counts per bin; invalid history rows are NaN"})
            print(str(args.out))
    elif args.command == "pipeline":
        result = run_pipeline(load_recording(args.data), args.out,
            exponential_basis(args.history, args.n_basis), args.ranks, args.seeds,
            args.max_iter, args.chunk, args.ridge, args.min_rate_hz)
        print(json.dumps({"status": result["status"], "test": result["test"]}, indent=2))
    else:
        if args.out.exists() and not args.resume:
            raise FileExistsError("output exists; use --resume to continue its checkpoint")
        args.out.mkdir(parents=True, exist_ok=True)
        write_json(args.out/"status.json", {"status": "RUNNING"})
        try:
            data = load_recording(args.data)
            model = StreamingGLM(exponential_basis(args.history, args.n_basis), args.chunk,
                None if args.rank == -1 else args.rank, args.ridge, self_history=True)
            options = {"checkpoint": args.out/"checkpoint.npz", "resume": args.resume}
            if args.solver == "adam":
                result = fit_adam(model, data, epochs=args.epochs, learning_rate=args.learning_rate, **options)
            else:
                result = model.fit(data, max_iter=args.max_iter, **options)
            save_bundle(args.out/"model.npz", model, result.params,
                        {"unit_ids": data.unit_ids, "bin_s": data.bin_s})
            report = asdict(result); report.pop("params")
            write_json(args.out/"fit.json", report)
            write_json(args.out/"status.json", {"status": "DONE", "converged": result.converged})
            print(json.dumps({k: v for k, v in report.items() if k != "trace"}, indent=2))
        except BaseException as exc:
            write_json(args.out/"status.json", {"status": "FAILED", "error": str(exc)})
            raise


if __name__ == "__main__":
    main()
