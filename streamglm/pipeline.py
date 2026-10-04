"""Complete train/validation/test workflow with explicit convergence gates."""
from dataclasses import asdict
import importlib.metadata
from pathlib import Path
import platform
import time
import numpy as np
from .evaluation import split_recording, select_rank, score
from .model import StreamingGLM
from .persistence import save_bundle, write_json


def exponential_basis(history=25, n_basis=3):
    if history < 1 or n_basis < 1 or n_basis > history:
        raise ValueError("need history >= n_basis >= 1")
    times = np.geomspace(1., max(2., history/2), n_basis)
    basis = np.exp(-np.arange(1, history+1)[:, None]/times)
    return basis/basis.sum(axis=0)


def run_pipeline(recording, out, basis=None, ranks=(0, 1, 2), seeds=(0, 1),
                 max_iter=500, chunk_size=1024, ridge=.01, min_rate_hz=.1):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"output already exists: {out}")
    out.mkdir(parents=True)
    started = time.perf_counter()
    write_json(out/"status.json", {"status": "RUNNING", "step": "split"})
    try:
        basis = exponential_basis() if basis is None else np.asarray(basis)
        train, validation, test = split_recording(recording)
        means = train.mean_counts(len(basis))
        keep = np.flatnonzero((means > 0) & (means/recording.bin_s >= min_rate_hz))
        if len(keep) == 0:
            raise ValueError("no units pass training-only firing-rate filter")
        train, validation, test = [part.select([(0, len(part.counts))], keep)
                                   for part in (train, validation, test)]
        means = np.maximum(train.mean_counts(len(basis)), 1e-8)
        if max(ranks) > len(keep):
            raise ValueError("requested rank exceeds retained neurons")
        write_json(out/"status.json", {"status": "RUNNING", "step": "rank_selection", "retained_neurons": len(keep)})
        def progress(rows):
            write_json(out/"selection_progress.json", rows)
            write_json(out/"status.json", {"status": "RUNNING", "step": "rank_selection",
                "completed_candidates": len(rows), "seconds": time.perf_counter()-started})
        model, fitted, selection = select_rank(train, validation, basis, tuple(ranks), tuple(seeds),
                                              chunk_size, ridge, max_iter, callback=progress)
        write_json(out/"selection.json", selection)
        # Independently report the self-history baseline on exactly the same test rows.
        baseline = StreamingGLM(basis, chunk_size, rank=0, ridge=ridge, self_history=True)
        basefit = baseline.fit(train, max_iter=max_iter)
        if not basefit.converged:
            raise RuntimeError("self-history baseline did not converge")
        test_score = score(model, fitted.params, test, means)
        baseline_score = score(baseline, basefit.params, test, means)
        save_bundle(out/"model.npz", model, fitted.params,
                    {"unit_ids": train.unit_ids, "bin_s": train.bin_s, "selection": selection["selected"]})
        save_bundle(out/"self_history.npz", baseline, basefit.params,
                    {"unit_ids": train.unit_ids, "bin_s": train.bin_s})
        result = asdict(fitted); result.pop("params")
        write_json(out/"fit.json", result)
        report = {"status": "DONE", "seconds": time.perf_counter()-started,
            "original_neurons": recording.n_neurons, "retained_neurons": len(keep),
            "unit_ids": train.unit_ids, "min_training_rate_hz": min_rate_hz,
            "split": "chronological 60/20/20 within each epoch; histories reset",
            "selection": selection, "test": test_score, "self_history_test": baseline_score,
            "test_gain_over_self_history_nats_per_bin": test_score["log_likelihood_per_bin"]-baseline_score["log_likelihood_per_bin"],
            "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "jax")},
            "platform": platform.platform(),
            "interpretation": "Predictive benchmark. Coupling does not establish anatomical or causal connectivity. No stimulus/behavior covariates in this version."}
        write_json(out/"report.json", report)
        lines = ["# StreamGLM predictive evaluation", "",
            f"Selected rank: {selection['selected']['rank']}; retained neurons: {len(keep)}.", "",
            "Selection used validation data; test data were scored only after selection.", "",
            f"Test gain over training-rate intercept: {test_score['gain_nats_per_bin']:.6g} nats/bin.",
            f"Test gain over self-history: {report['test_gain_over_self_history_nats_per_bin']:.6g} nats/bin.", "",
            "| Rank | Seed | Converged | Validation log likelihood/bin |",
            "|---:|---:|:---:|---:|"]
        for row in selection["candidates"]:
            lines.append(f"| {row['rank']} | {row['seed']} | {row['converged']} | {row['validation']['log_likelihood_per_bin']:.6g} |")
        lines.extend(["", report["interpretation"], ""])
        (out/"report.md").write_text("\n".join(lines))
        write_json(out/"status.json", {"status": "DONE", "seconds": report["seconds"]})
        return report
    except BaseException as exc:
        write_json(out/"status.json", {"status": "FAILED", "error": str(exc), "seconds": time.perf_counter()-started})
        raise
