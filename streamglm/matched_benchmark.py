"""Matched dense Poisson fits: streamed, NeMoS/common optimizer, NeMoS/native.

The common SciPy driver isolates objective evaluation; it is not NeMoS.fit.
The native arm uses NeMoS.fit and is reported separately. No low-rank speed claim.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

from .persistence import write_json


def worker(data, out, backend, max_iter=1000):
    started = time.perf_counter()
    result = {'backend': backend, 'status': 'RUNNING', 'max_iter': max_iter}
    write_json(out.with_suffix('.json'), result)
    try:
        import jax
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        from jax.flatten_util import ravel_pytree
        from scipy.optimize import minimize
        from scipy.special import gammaln
        from .data import Recording
        from .model import StreamingGLM
        with np.load(data) as f:
            counts, basis = f['counts'], f['basis']
        split = int(.8*len(counts))
        train, test = counts[:split], counts[split:]
        h, k = basis.shape
        n = counts.shape[1]
        initial = {'W': jnp.zeros((k, n, n)),
                   'b': jnp.asarray(np.log(train[h:].mean(axis=0)))}
        flat, unravel = ravel_pytree(initial)
        fit_started = time.perf_counter()
        if backend == 'streamed':
            model = StreamingGLM(basis, chunk_size=512)
            recording = Recording(train)
            def objective(vector):
                value, grad = model.value_and_grad(unravel(jnp.asarray(vector)), recording)
                return value, np.asarray(ravel_pytree(grad)[0])
        else:
            import nemos as nmo
            kwargs = {'regularizer': 'UnRegularized', 'solver_name': 'LBFGS',
                      'solver_kwargs': {'maxiter': max_iter, 'tol': 1e-7}}
            if 'inverse_link_function' in inspect.signature(nmo.glm.PopulationGLM).parameters:
                kwargs.update(observation_model='Poisson', inverse_link_function=jnp.exp)
            else:
                kwargs['observation_model'] = nmo.observation_models.PoissonObservations(inverse_link_function=jnp.exp)
            glm = nmo.glm.PopulationGLM(**kwargs)
            def design(y):
                x = nmo.convolve.create_convolutional_predictor(basis, y, shift=True)
                return jnp.asarray(x[h:].reshape(len(y)-h, n*k))
            x = design(train)
            # Integer inputs make JAX gammaln use float32 even with x64 enabled.
            y = jnp.asarray(train[h:], dtype=jnp.float64)
            factorial = jnp.sum(jnp.mean(jax.scipy.special.gammaln(y+1), axis=0))
            def loss(vector):
                p = unravel(vector)
                coef = p['W'].transpose(1, 0, 2).reshape(n*k, n)
                rate = jnp.exp(x @ coef+p['b'])
                return -glm.observation_model.log_likelihood(y, rate,
                    aggregate_sample_scores=lambda z: jnp.sum(jnp.mean(z, axis=0)))-factorial
            compiled = jax.jit(jax.value_and_grad(loss))
            def objective(vector):
                value, grad = compiled(jnp.asarray(vector))
                return float(value), np.asarray(grad)
            result.update({'nemos_version': importlib.metadata.version('nemos'),
                'nemos_source_hashes': {str(Path(inspect.getfile(cls)).name):
                    hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
                    for cls in (nmo.glm.PopulationGLM, type(glm.observation_model))},
                'has_stochastic_fit': hasattr(glm, 'stochastic_fit')})
        initial_value, initial_grad = objective(np.asarray(flat))
        if backend == 'nemos_native':
            glm.fit(x, y, init_params=(jnp.zeros((n*k, n)), initial['b']))
            params = {'W': np.asarray(glm.coef_).reshape(n, k, n).transpose(1, 0, 2),
                      'b': np.asarray(glm.intercept_)}
            vector = np.asarray(ravel_pytree(params)[0])
            info = getattr(glm, 'optim_info_', None)
            optimizer_success = bool(info.converged) if hasattr(info, 'converged') else None
            message = str(info)
            iterations = int(info.iter_num) if hasattr(info, 'iter_num') else None
        else:
            fit = minimize(objective, np.asarray(flat), method='L-BFGS-B', jac=True,
                options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 10})
            vector = fit.x
            params = {key: np.asarray(value) for key, value in unravel(jnp.asarray(vector)).items()}
            optimizer_success, message, iterations = bool(fit.success), str(fit.message), int(fit.nit)
        fit_seconds = time.perf_counter()-fit_started
        value, gradient = objective(vector)
        if backend == 'streamed':
            prediction = np.concatenate([p for _, p, _ in model.predict_chunks(params, Recording(test))])[h:]
        else:
            glm.coef_ = jnp.asarray(params['W'].transpose(1, 0, 2).reshape(n*k, n))
            glm.intercept_ = jnp.asarray(params['b'])
            glm.scale_ = 1.
            prediction = np.asarray(glm.predict(design(test)))
        if not np.all(np.isfinite(prediction)) or np.any(prediction <= 0):
            raise FloatingPointError('invalid held-out predictions')
        target = test[h:]
        test_ll = float(np.sum(target*np.log(prediction)-prediction-gammaln(target+1))/len(target))
        gradient_norm = float(np.max(np.abs(gradient)))
        np.savez_compressed(out.with_suffix('.npz'), **params, prediction=prediction,
                            initial_gradient=initial_grad)
        result.update({'status': 'DONE', 'optimizer_success': optimizer_success,
            'message': message, 'iterations': iterations, 'objective': value,
            'initial_objective': initial_value, 'gradient_inf_norm': gradient_norm,
            'accuracy_gate_passed': gradient_norm <= 1e-5,
            'heldout_log_likelihood_per_bin': test_ll, 'test_valid_bins': len(target),
            'fit_seconds_including_preparation': fit_seconds,
            'bins': len(counts), 'neurons': n, 'n_basis': k, 'dtype': 'float64',
            'regularizer': 'none', 'cpu_affinity': sorted(os.sched_getaffinity(0)),
            'versions': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'jax')}})
    except BaseException as exc:
        result.update({'status': 'FAILED', 'error': repr(exc)})
        raise
    finally:
        result.update({'total_seconds': time.perf_counter()-started,
                       'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20})
        write_json(out.with_suffix('.json'), result)


def suite(args):
    from .synthetic import fixture
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out/'provenance.json', {
        'command': sys.argv,
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in Path(__file__).parent.glob('*.py')},
        'environment': {key: os.environ.get(key) for key in
                        ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'JAX_PLATFORMS')},
        'cpu_affinity': sorted(os.sched_getaffinity(0))})
    rows = []
    for bins in args.bins:
        for seed in args.seeds:
            stem = f't{bins}_n{args.neurons}_s{seed}'
            counts, basis, _ = fixture(bins, args.neurons, seed)
            data = args.out/f'{stem}_input.npz'
            np.savez_compressed(data, counts=counts, basis=basis)
            results = {}
            for backend in ('streamed', 'nemos_common', 'nemos_native'):
                out = args.out/f'{stem}_{backend}'
                command = [sys.executable, '-m', 'streamglm.matched_benchmark', '--worker',
                           backend, '--data', str(data), '--out', str(out)]
                print(f'START {stem} {backend}', flush=True)
                try:
                    with out.with_suffix('.log').open('w') as log:
                        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=180)
                    record = json.loads(out.with_suffix('.json').read_text()) if out.with_suffix('.json').exists() else {}
                    if completed.returncode:
                        record.update(status='FAILED', returncode=completed.returncode)
                except subprocess.TimeoutExpired:
                    record = {'status': 'FAILED', 'error': '180-second timeout'}
                record.update(backend=backend, bins=bins, seed=seed, neurons=args.neurons)
                write_json(out.with_suffix('.json'), record)
                results[backend] = record
            comparisons = {}
            if results['streamed']['status'] == 'DONE':
                with np.load(args.out/f'{stem}_streamed.npz') as ref:
                    for backend in ('nemos_common', 'nemos_native'):
                        if results[backend]['status'] != 'DONE':
                            continue
                        with np.load(args.out/f'{stem}_{backend}.npz') as other:
                            comparisons[backend] = {
                                'max_prediction_abs_difference': float(np.max(np.abs(ref['prediction']-other['prediction']))),
                                'max_initial_gradient_abs_difference': float(np.max(np.abs(ref['initial_gradient']-other['initial_gradient']))),
                                'initial_objective_abs_difference': abs(results['streamed']['initial_objective']-results[backend]['initial_objective']),
                                'objective_abs_difference': abs(results['streamed']['objective']-results[backend]['objective']),
                                'heldout_ll_abs_difference': abs(results['streamed']['heldout_log_likelihood_per_bin']-results[backend]['heldout_log_likelihood_per_bin'])}
                            comparison = comparisons[backend]
                            comparison['agreement_passed'] = (
                                results['streamed']['accuracy_gate_passed']
                                and results[backend]['accuracy_gate_passed']
                                and comparison['max_initial_gradient_abs_difference'] <= 1e-10
                                and comparison['initial_objective_abs_difference'] <= 1e-10
                                and comparison['objective_abs_difference'] <= 1e-6
                                and comparison['heldout_ll_abs_difference'] <= 1e-6
                                and comparison['max_prediction_abs_difference'] <= 1e-3)
            rows.append({'bins': bins, 'seed': seed, 'results': results, 'comparisons': comparisons,
                         'input_sha256': hashlib.sha256(data.read_bytes()).hexdigest()})
            write_json(args.out/'summary.json', {'rows': rows, 'status': 'RUNNING'})
    write_json(args.out/'summary.json', {'rows': rows, 'status': 'DONE',
        'note': 'Dense unregularized exp-Poisson; identical zero weights and training-rate intercepts; '
                '80/20 chronological split with reset history; native solver differs from common SciPy driver. '
                'Accuracy gate is gradient infinity norm <=1e-5. Shared host; no upstream streaming comparison.'})
    report(args.out, rows)


def report(out, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = {'streamed': 'StreamGLM / SciPy', 'nemos_common': 'NeMoS objective / SciPy',
              'nemos_native': 'NeMoS native fit'}
    colors = {'streamed': 'tab:blue', 'nemos_common': 'tab:orange', 'nemos_native': 'tab:green'}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    lines = ['# Matched dense fits', '',
             'Fresh processes, float64, unregularized, identical initialization and splits. '
             'Each row is a synthetic dataset; seeds are data replicates, not timing repeats. '
             'The HNN simulation shared this host. NeMoS native uses a different solver.', '',
             '| Bins | Seed | Backend | Seconds* | Peak GiB | Gradient norm | Accuracy gate |',
             '|---:|---:|---|---:|---:|---:|:---:|']
    for backend, label in labels.items():
        for axis, metric in zip(axes[:2], ['fit_seconds_including_preparation', 'peak_rss_gib']):
            points = [(r['bins'], r['results'][backend].get(metric)) for r in rows
                      if r['results'][backend]['status'] == 'DONE']
            if points:
                axis.scatter(*zip(*points), label=label, color=colors[backend], alpha=.7)
        if backend != 'streamed':
            points = [(r['bins'], r['comparisons'][backend]['heldout_ll_abs_difference'])
                      for r in rows if backend in r['comparisons']]
            if points:
                axes[2].scatter(*zip(*points), label=label, color=colors[backend], alpha=.7)
    for r in rows:
        for backend, d in r['results'].items():
            if d['status'] != 'DONE':
                lines.append(f"| {r['bins']} | {r['seed']} | {backend} FAILED | — | — | — | no |")
            else:
                lines.append(f"| {r['bins']} | {r['seed']} | {backend} | {d['fit_seconds_including_preparation']:.3f} | {d['peak_rss_gib']:.3f} | {d['gradient_inf_norm']:.2g} | {d['accuracy_gate_passed']} |")
    for axis, title in zip(axes, ['Preparation + fit seconds', 'Whole-process peak GiB', 'Held-out LL difference (nats/bin)']):
        axis.set(xlabel='Recording bins', title=title, xscale='log')
        ticks = sorted({row['bins'] for row in rows})
        axis.set_xticks(ticks, [f'{value:,}' for value in ticks])
        axis.minorticks_off()
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    axes[2].axhline(1e-6, color='gray', linestyle='--', label='LL agreement threshold')
    axes[2].legend(fontsize=7)
    axes[2].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    fig.savefig(out/'comparison.png', dpi=160)
    fig.savefig(out/'comparison.svg')
    plt.close(fig)
    lines += ['', '*Includes feature construction, initial objective evaluation, compilation and optimization. '
              'Peak RSS covers the whole worker, including held-out scoring. '
              'Common-driver fits use SciPy L-BFGS-B (ftol=1e-14, gtol=1e-7, maxcor=10). '
              'Native NeMoS uses LBFGS tol=1e-7. All have a 1,000-iteration budget. '
              'The separate accuracy gate requires gradient infinity norm <=1e-5. '
              'No low-rank, real-data, large-scale speedup or current upstream batching claim.', '']
    lines += ['Native timing also includes the independent objective-audit compilation; '
              'use the common-driver arms for the primary implementation comparison.', '',
              '## Agreement with StreamGLM', '',
              '| Bins | Seed | Reference | Objective difference | Held-out LL difference | Max prediction difference | All gates pass |',
              '|---:|---:|---|---:|---:|---:|:---:|']
    for row in rows:
        for backend, comparison in row['comparisons'].items():
            lines.append(f"| {row['bins']} | {row['seed']} | {backend} | "
                         f"{comparison['objective_abs_difference']:.3g} | "
                         f"{comparison['heldout_ll_abs_difference']:.3g} | "
                         f"{comparison['max_prediction_abs_difference']:.3g} | "
                         f"{comparison.get('agreement_passed', 'not recorded')} |")
    lines += ['', 'Agreement requires gradient norms <=1e-5, initial objective/gradient '
              'differences <=1e-10, final objective and held-out LL differences <=1e-6, '
              'and maximum prediction difference <=1e-3. No failed gate is omitted.', '']
    (out/'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['streamed', 'nemos_common', 'nemos_native'])
    parser.add_argument('--data', type=Path)
    parser.add_argument('--bins', type=int, nargs='+', default=[4000, 16000, 64000])
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 11, 19])
    parser.add_argument('--neurons', type=int, default=16)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.worker:
        worker(args.data, args.out, args.worker)
    else:
        suite(args)


if __name__ == '__main__':
    main()
