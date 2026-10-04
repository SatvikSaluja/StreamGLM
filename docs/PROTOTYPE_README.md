# StreamGLM 0.1 prototype (historical)

Memory-bounded population GLMs for long neural recordings, implemented in JAX.
This is an early working prototype: exact streamed evaluation first, low-rank
coupling as a separately validated model restriction. No claim of brain-wide
anatomical reconstruction or hours-long, 1,000-neuron laptop fitting is made.

## Implemented

- Replayable count batches from arrays or read-only `.npy` memory maps.
- Causal lag-one history, overlap between chunks, independent epoch boundaries,
  and exclusion of initial bins without complete history.
- Dense coupling and rank-r coupling for each temporal basis coefficient.
- Projection before convolution in the factorized path: `(X * b) U = (X U) * b`.
- Exact chunk-wise Poisson objective and autodifferentiated gradients, accumulated
  across a recording; no recording-sized design matrix or gradient tape.
- L-BFGS fitting, convergence diagnostics, evaluation trace and chunked prediction.
- NeMoS checks of convolution, prediction, likelihood and gradients at identical
  parameters on a 20-neuron problem. Factorized fits are not asserted to equal
  unrestricted NeMoS optima.
- A synthetic low-rank fit and fresh-process, disk-backed pass benchmarks.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[validation]'
python -m unittest discover -s tests_streamglm -v
python -m streamglm.demo --max-iter 800 --out artifacts/my_demo.json
python -m streamglm.benchmark --bins 65536 --neurons 128 --out artifacts/my_benchmark.json
```

The current workspace already has a local `.venv` with an editable install.
It uses the existing scientific environment's packages via system-site-packages;
no packages were installed into the main experiment's environment.
On this Linux machine, checks were restricted to CPU cores 6 and 7:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu \
  taskset -c 6,7 .venv/bin/python -m unittest discover -s tests_streamglm -v
```

Choose CPU IDs available on your own machine. Tests and command-line examples
enable float64 explicitly; importing the library does not change JAX configuration.

## Minimal usage

```python
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from streamglm import Recording, StreamingGLM

counts = Recording.from_npy("counts.npy", epoch_lengths=[100_000, 100_000])
lag = np.arange(1, 26)[:, None]
basis = np.exp(-lag / np.array([2., 5., 10.])[None, :])
basis /= basis.sum(axis=0)
model = StreamingGLM(basis, chunk_size=1024, rank=5, ridge=0.001)
result = model.fit(counts, max_iter=100)
print(result.converged, result.message, result.gradient_inf_norm)
for start, predicted_counts, valid in model.predict_chunks(result.params, counts):
    # Consume or write each block; concatenating them defeats bounded output memory.
    pass
```

`counts.npy` contains nonnegative integer counts with shape `(time, neuron)`.
Epoch lengths must partition it. Basis shape is `(history_bins, n_basis)`;
row zero represents lag one. Predictions are expected counts per bin. Divide
by bin duration to obtain Hz. Every neuron can predict every target, including
its own future bins. Dedicated refractory/self-history terms are not implemented.

The objective is the mean over valid time bins of the summed-neuron Poisson
negative log likelihood, omitting log-factorials, plus `ridge/2 * ||W||²`.
The penalty is on reconstructed W for both parameterizations, evaluated via
rank-sized Gram matrices for low rank. Intercepts are not penalized.
`value_and_grad` performs one complete data pass, not one minibatch update.
L-BFGS may make several passes per iteration during line search.

## First measured results

The six unittest methods pass, including dense and low-rank variants, multiple
chunk sizes, a short epoch, tail padding, disk input, and a fitted dense model.
Equivalence tolerances are roughly `1e-10` to `1e-11` in float64; these are
test tolerances, not a claim of bitwise equality.

In the eight-neuron synthetic demo, one rank-2 fit reached the optimizer's
relative-objective convergence criterion after 205 iterations (225 evaluations).
Held-out improvement over a training-rate intercept model was 0.04557 nats/bin,
summed over neurons. Gradient infinity norm was 7.15e-5. Reconstructed-weight
relative error was 0.489: useful predictive performance is not exact recovery.
This is one small fixture and one initialization, not a scientific calibration.
The earlier 150-iteration result is preserved and correctly reports nonconvergence.

Fresh-process benchmark, two CPU cores, float64, K=3, rank=5, H=25, chunk=512:

| Bins | Neurons | Warm objective + gradient pass | Whole-process peak RSS |
|---:|---:|---:|---:|
| 8,192 | 32 | 0.026 s | 0.277 GiB |
| 8,192 | 128 | 0.075 s | 0.288 GiB |
| 8,192 | 512 | 0.178 s | 0.313 GiB |
| 65,536 | 128 | 0.505 s | 0.299 GiB |

Cold passes took 1.07–1.34 s including compilation. These are single measurements
on synthetic disk-backed counts, while another experiment was running. They
measure two passes at fixed parameters, not time to convergence. Peak RSS includes
the Python/JAX runtime, compilation and mapped file pages, and is measured using
Linux `ru_maxrss`. The OS may retain mapped pages; total RSS is not guaranteed
constant with recording length. Larger repeated benchmarks are still required.
Raw records and traces are in [artifacts](artifacts/).

## Limits and next steps

1. Compare against NeMoS's existing batching implementation at pinned revisions,
   before proposing an upstream change. This prototype is not a NeMoS extension yet.
2. Expand convergence and recovery validation across ranks, initializations,
   firing rates and deliberately misspecified ground truth. Low-rank fitting
   is nonconvex and its factors are not individually identifiable.
3. Measure time-to-fit, memory and held-out prediction on longer disk-backed data.
   Dense parameters still scale as K*N²; low-rank parameters scale as 2*K*N*r.
   Backpropagation workspace is measured, not inferred from raw array sizes.
4. Add tested Pynapple/NWB adapters and one quality-filtered real session.
5. Reuse audited HNN recordings as an additional validation case. Correct the
   scientific controls before making anatomical recovery claims.

No Pynapple/NWB adapter, held-out rank selection, stochastic optimizer, resume
checkpoint, custom target mask or real Neuropixels analysis is implemented yet.
Inputs with missing values must be split into valid epochs; silently filling
gaps with zero counts would change the model.

## Workspace provenance

This worktree is `/home/satvik/side_project`, branch
`side-project/streaming-glm`, forked from NeuroTwinBench commit `9dca3ef`.
The inherited `neurotwinbench/`, `experiments/`, `PROJECT.md` and
`CODEX_HANDOFF.md` describe the older project and are not part of this package.
Its previous README is preserved at [docs/NEUROTWINBENCH_README.md](docs/NEUROTWINBENCH_README.md).
The primary experiment runs in a separate worktree and was not stopped or edited.
