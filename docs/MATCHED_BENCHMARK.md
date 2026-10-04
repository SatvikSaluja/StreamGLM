# Matched dense-model benchmark

This benchmark separates implementation comparison from model compression.
All three arms fit unrestricted recurrent population GLMs, including diagonal
self-history weights, with an exponential Poisson link and no regularization.
There is no low-rank constraint or rank selection in this experiment.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu \
taskset -c 6,7 .venv/bin/python -m streamglm.matched_benchmark \
  --out artifacts/my_matched_benchmark
```

Use a new output directory. Defaults are 16 neurons, two temporal basis
functions spanning eight bins, 4,000/16,000/64,000 bins, and data seeds 7/11/19.
Each dataset is generated once and shared by all three fresh worker processes.
SHA-256 hashes of the input files, local source, and installed NeMoS GLM and
observation-model source are saved. Version strings alone are insufficient for
the locally installed NeMoS build.

The three arms are:

1. StreamGLM's exact chunked objective, with SciPy L-BFGS-B.
2. NeMoS's materialized causal features and public Poisson likelihood, with
   exactly the same SciPy driver and flattened parameter ordering.
3. Native `NeMoS.PopulationGLM.fit`, using its own L-BFGS implementation.

Arm 2 isolates the feature/objective implementation. It is deliberately **not**
a claim to benchmark NeMoS's native optimizer. Arm 3 supplies that additional
reference. None benchmarks newer upstream streaming or stochastic fitting.

## Matching and acceptance

All arms use float64, zero initial coupling, training-rate intercepts, and an
80/20 chronological split. Features are recomputed separately for train and
test; the first eight bins of each split are excluded. No hyperparameters are
selected using held-out data. The counts are cast to float64 before NeMoS
likelihood evaluation to avoid float32 evaluation of integer-input log-factorials.

The common driver uses `ftol=1e-14`, `gtol=1e-7`, `maxcor=10`; native NeMoS
uses `tol=1e-7`. These native stopping rules are not asserted equivalent.
Every arm has a 1,000-iteration budget and 180-second process timeout.
An independent final gradient infinity-norm threshold of `1e-5` checks accuracy.
Optimizer success, stopping message, and gradient are retained separately.

Agreement requires both fits to pass that gradient check, initial objective and
gradient differences at most `1e-10`, final objective and held-out likelihood
differences at most `1e-6`, and maximum held-out predicted-count difference at
most `1e-3`. Absolute likelihood differences are in nats per valid time bin,
summed over neurons. Failed and timed-out jobs remain in the output table.

## Timing and memory interpretation

Preparation-plus-fit time includes feature construction, initial objective
evaluation, compilation and optimization. Total worker time is also saved.
Linux `ru_maxrss` records whole-process peak memory, including imports, data,
optimizer state and held-out scoring. It does not isolate feature memory.
The native NeMoS arm includes the independent objective-audit compilation as
well as its own optimizer compilation; it is not a clean fit-only timing of
`NeMoS.fit`. Use the common-driver arms for the primary implementation tradeoff.

The runs use CPU cores 6 and 7 while the independent HNN simulation shares the
host. Three seeds are distinct datasets, not repeated timing trials on identical
data. Backend order is fixed. Do not interpret these measurements as a controlled
hardware speedup, an asymptotic memory proof, or evidence for a 1,000-neuron fit.

The initial `matched_smoke` artifact predates the float64 log-factorial fix and
is retained as diagnostic history. Use `matched_dense` for the final comparison.
