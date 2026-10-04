"""Fresh-process complete-fit benchmark, including selection and test scoring.

Use --data for an imported recording, or a small synthetic fixture by default.
Run each configuration in a separate process. This is not a solver comparison.
"""
import argparse
import os
from pathlib import Path
import resource
import time

import jax

from .adapters import load_recording
from .data import Recording
from .persistence import write_json
from .pipeline import exponential_basis, run_pipeline
from .synthetic import fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--bins', type=int, default=20000)
    parser.add_argument('--neurons', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--ranks', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--fit-seeds', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--max-iter', type=int, default=500)
    parser.add_argument('--chunk', type=int, default=512)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    # Never overwrite a previous benchmark, including an unsuccessful one.
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    jax.config.update('jax_enable_x64', True)
    metadata = {
        'arguments': vars(args) | {'data': str(args.data) if args.data else None,
                                   'out': str(args.out)},
        'dtype': 'float64',
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'note': 'Whole-process Linux peak RSS includes input, JAX compilation, '
                'selection, baseline fitting and scoring. Cold workflow; no warmup. '
                'Synthetic neuron-count scaling also changes firing statistics. '
                'No comparison against NeMoS or anatomical recovery claim.',
    }
    write_json(args.out/'benchmark.json', metadata | {'status': 'RUNNING'})
    try:
        if args.data:
            recording = load_recording(args.data)
            basis = exponential_basis()
        else:
            counts, basis, _ = fixture(args.bins, args.neurons, args.seed)
            recording = Recording(counts)
        metadata.update({'recording_fingerprint': recording.fingerprint(),
                         'bins': len(recording.counts), 'neurons': recording.n_neurons,
                         'n_basis': basis.shape[1], 'history': len(basis)})
        report = run_pipeline(recording, args.out/'workflow', basis=basis,
                              ranks=args.ranks, seeds=args.fit_seeds,
                              max_iter=args.max_iter, chunk_size=args.chunk,
                              min_rate_hz=0.)
        candidates = report['selection']['candidates']
        metadata.update({'status': 'DONE',
                         'all_candidates_converged': all(c['converged'] for c in candidates),
                         'selection': report['selection'],
                         'test': report['test'],
                         'test_gain_over_self_history_nats_per_bin':
                         report['test_gain_over_self_history_nats_per_bin'],
                         'versions': report['versions'], 'platform': report['platform']})
    except BaseException as exc:
        metadata.update({'status': 'FAILED', 'error': str(exc)})
        raise
    finally:
        metadata.update({'total_seconds': time.perf_counter()-started,
                         'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20})
        write_json(args.out/'benchmark.json', metadata)


if __name__ == '__main__':
    main()
