"""Exercise current NeMoS stochastic fitting through an epoch-safe loader.

Run in the separately pinned upstream environment. This is an interoperability
and accuracy experiment, not a memory/speed comparison between optimizers.
"""
import argparse
import importlib.metadata
from pathlib import Path
import resource
import time

import jax
import jax.numpy as jnp
import numpy as np

from .data import Recording
from .evaluation import score
from .integration import NeMoSRecordingLoader
from .model import StreamingGLM
from .persistence import write_json, save_bundle
from .synthetic import fixture


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--passes', type=int, default=300)
    p.add_argument('--stepsize', type=float, default=.5)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    jax.config.update('jax_enable_x64', True)
    import nemos as nmo
    started = time.perf_counter()
    write_json(args.out/'status.json', {'status': 'RUNNING'})
    try:
        counts, basis, _ = fixture(6000, 6, seed=23)
        train = Recording(counts[:4800], [2400, 2400])
        test = Recording(counts[4800:])
        loader = NeMoSRecordingLoader(train, basis, 512)
        reference = StreamingGLM(basis, chunk_size=512)
        mean = train.mean_counts(len(basis))
        fitted = reference.fit(train, reference.initialize(6, mean_count=mean),
                               max_iter=1500, tolerance=1e-12)
        glm = nmo.glm.PopulationGLM(observation_model='Poisson', inverse_link_function=jnp.exp,
            regularizer='UnRegularized', solver_name='SVRG',
            solver_kwargs={'stepsize': args.stepsize, 'tol': 1e-7})
        def convert(coef, intercept):
            return {'W': np.asarray(coef).reshape(6, 2, 6).transpose(1, 0, 2),
                    'b': np.asarray(intercept)}
        trace = []
        class Accuracy(nmo.callbacks.Callback):
            def on_pass_end(self, ctx):
                if (ctx.pass_idx+1) % 10:
                    return
                value, grad = reference.value_and_grad(convert(ctx.params.coef, ctx.params.intercept), train)
                norm = max(float(np.max(np.abs(g))) for g in grad.values())
                trace.append({'pass': ctx.pass_idx+1, 'objective': value, 'gradient_inf_norm': norm})
                write_json(args.out/'progress.json', trace)
                print(trace[-1], flush=True)
                if norm <= 1e-5:
                    ctx.request_stop('independent full-gradient threshold reached')
        glm.stochastic_fit(loader, n_passes=args.passes,
            init_params=(jnp.zeros((12, 6)), jnp.log(jnp.asarray(mean))), callbacks=Accuracy())
        params = convert(glm.coef_, glm.intercept_)
        value, grad = reference.value_and_grad(params, train)
        norm = max(float(np.max(np.abs(g))) for g in grad.values())
        stochastic_score = score(reference, params, test, mean)
        reference_score = score(reference, fitted.params, test, mean)
        result = {'status': 'DONE', 'nemos_version': importlib.metadata.version('nemos'),
            'versions': {k: importlib.metadata.version(k) for k in ('jax', 'numpy', 'scipy', 'pynapple')},
            'upstream_commit': '81c7200a0e66686e98fd1907e7fa11e111a0e66a',
            'train_valid_bins': loader.n_samples, 'train_epoch_lengths': list(train.epoch_lengths),
            'stochastic_summary': str(glm.stochastic_fit_summary_),
            'stochastic_objective': value, 'stochastic_gradient_inf_norm': norm,
            'stochastic_accuracy_passed': norm <= 1e-5,
            'reference_converged': fitted.converged, 'reference_gradient_inf_norm': fitted.gradient_inf_norm,
            'objective_difference': abs(value-fitted.objective),
            'stochastic_test': stochastic_score, 'reference_test': reference_score,
            'seconds': time.perf_counter()-started,
            'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            'passes_budget': args.passes, 'stepsize': args.stepsize,
            'note': 'Both solvers share this process; timings/RSS are not comparative. Current upstream already supports streamed convolution via custom loaders. No native low-rank model is used in this check.'}
        write_json(args.out/'report.json', result)
        save_bundle(args.out/'nemos_model.npz', reference, params)
        write_json(args.out/'status.json', {'status': 'DONE', 'accuracy_passed': norm <= 1e-5})
    except BaseException as exc:
        write_json(args.out/'status.json', {'status': 'FAILED', 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
