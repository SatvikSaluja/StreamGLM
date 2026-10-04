"""Complete low-rank CPU fit on a disk-backed independent-Poisson stress case.

This deliberately easy null is a computational test, not a recovery calibration
or evidence that correlated 1,000-unit Neuropixels data can be fit equally fast.
"""
import argparse
from dataclasses import asdict
import importlib.metadata
import os
from pathlib import Path
import resource
import time

import jax
import numpy as np

from .data import Recording
from .evaluation import score, split_recording
from .model import StreamingGLM
from .persistence import write_json, save_bundle
from .pipeline import exponential_basis


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--counts', type=Path, required=True)
    p.add_argument('--neurons', type=int, default=1000)
    p.add_argument('--seconds', type=float, default=3600)
    p.add_argument('--bin-s', type=float, default=.01)
    p.add_argument('--max-iter', type=int, default=100)
    p.add_argument('--seed', type=int, default=732)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    def status(**fields):
        write_json(args.out/'status.json', {'pid': os.getpid(),
                   'updated_unix': time.time(), 'elapsed_s': time.perf_counter()-started,
                   'seed': args.seed, **fields})
    status(status='RUNNING', step='generate')
    jax.config.update('jax_enable_x64', True)
    try:
        if args.counts.exists():
            raise FileExistsError(args.counts)
        args.counts.parent.mkdir(parents=True, exist_ok=True)
        n = int(round(args.seconds/args.bin_s))
        rng = np.random.default_rng(args.seed)
        data = np.lib.format.open_memmap(args.counts, mode='w+', dtype=np.int16, shape=(n, args.neurons))
        for start in range(0, n, 2048):
            stop = min(n, start+2048)
            data[start:stop] = rng.poisson(4*args.bin_s, (stop-start, args.neurons))
        data.flush()
        del data
        recording = Recording.from_npy(args.counts, bin_s=args.bin_s)
        train, _, test = split_recording(recording)
        basis = exponential_basis(10, 3)
        mean = train.mean_counts(len(basis))
        fitted_models = {}
        for rank in (0, 5):
            status(status='RUNNING', step='fit', rank=rank)
            model = StreamingGLM(basis, chunk_size=2048, rank=rank, ridge=.01, self_history=True)
            def progress(row):
                status(status='RUNNING', step='fit', rank=rank, **row)
                print(rank, row, flush=True)
            fit = model.fit(train, max_iter=args.max_iter, tolerance=1e-7,
                            checkpoint=args.out/f'checkpoint_rank{rank}.npz', callback=progress)
            save_bundle(args.out/f'model_rank{rank}.npz', model, fit.params,
                        {'unit_ids': recording.unit_ids, 'bin_s': args.bin_s})
            diagnostics = asdict(fit)
            diagnostics.pop('params')
            diagnostics['test'] = score(model, fit.params, test, mean)
            fitted_models[str(rank)] = diagnostics
            write_json(args.out/f'fit_rank{rank}.json', diagnostics)
        write_json(args.out/'report.json', {
            'status': 'DONE', 'generator': f'independent Poisson, 4 Hz/unit, seed {args.seed}',
            'seed': args.seed,
            'neurons': args.neurons, 'recording_seconds': args.seconds, 'bins': n,
            'bin_s': args.bin_s, 'history_bins': 10, 'n_basis': 3,
            'split': '60/20/20 chronological; validation portion unused; no rank selection',
            'counts_fingerprint': recording.fingerprint(), 'fits': fitted_models,
            'gain_over_self_history_nats_per_bin': fitted_models['5']['test']['log_likelihood_per_bin']-fitted_models['0']['test']['log_likelihood_per_bin'],
            'hypothetical_full_design_gib': n*args.neurons*3*8/2**30,
            'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            'seconds': time.perf_counter()-started,
            'versions': {k: importlib.metadata.version(k) for k in ('jax', 'numpy', 'scipy')},
            'note': 'Synthetic independent null at 10 ms bins, not 1 ms Neuropixels or biophysical recovery. Whole-process RSS includes mapped input pages. Shared host; convergence diagnostics retained.'})
        status(status='DONE', all_optimizer_converged=all(f['converged'] for f in fitted_models.values()))
    except BaseException as exc:
        status(status='FAILED', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
