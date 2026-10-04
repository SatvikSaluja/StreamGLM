# StreamGLM

[Results report](RESULTS_REPORT.md) · [Simulation commands](SIMULATIONS.md) · [HTML results page](results_page/index.html) · [Page source reports](results_page/sources)

The GitHub publication is a compact source/evidence snapshot. Download the `results_page` folder and open `index.html` in a browser; GitHub's file view displays HTML source rather than rendering the page. Large recordings and caches remain local. Actions workflows are not enabled in this publication snapshot.

**Memory-bounded population GLMs for long neural recordings.**

StreamGLM 0.3 is a working scientific Python package for causal dense and low-rank
Poisson population models. It reads count data in chunks, computes features on
demand, and provides a complete import → fit → select → evaluate workflow.

Validation includes five minutes of real Neuropixels data (185 retained VISp
units; held-out gain over self-history 0.196 nats/bin) and a complete 1,000-unit,
one-hour **independent-Poisson synthetic** workflow at **10 ms bins** (27.1 minutes,
1.24 GiB peak RSS). An hour-long 1,000-neuron real Neuropixels fit is not demonstrated.

Current NeMoS already supports streaming fitting. Our epoch-aware loader is
checked against pinned upstream NeMoS. See the [measured report](artifacts/v4/report.md)
and [protocol and limitations](docs/RELEASE_03.md).

![Measured validation](artifacts/v4/overview.png)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[validation,io,plots]'
streamglm --help
python -m unittest discover -s tests_streamglm -v
```

Python >=3.10 is supported by the package metadata; all 23 tests passed on Python 3.11 and Python 3.12 with pinned current NeMoS.
The existing local `.venv` is already installed and shares the original scientific
environment's dependencies read-only. It does not modify the main experiment.
For the exact tested numerical versions, see [constraints-validated.txt](constraints-validated.txt).

## What is implemented

- Read-only memory-mapped count recordings with explicit epoch lengths and unit IDs.
- NWB Units, Pynapple TsGroup and cached HNN NPZ importers.
- Strictly causal history, overlap at chunk boundaries, and resets between epochs.
- Dense coupling or `W_k = U_k V_k.T`, with projection before convolution.
- Optional independent self-history with the coupling diagonal removed.
- Exact full-data gradients accumulated one chunk at a time, with JAX recomputation.
- L-BFGS and sequential minibatch Adam, convergence diagnostics and progress traces.
- Atomic parameter files and checkpoints; no pickle loading.
- Chronological train/validation/test splitting, multiple initializations and
  validation-based rank selection, including a self-history-only baseline.
- Training-only firing-rate filtering and test likelihood/bits-per-spike reporting.
- CLI fitting, resume, evaluation and incremental prediction output.
- Explicit NeMoS prediction/scoring export with a dense-allocation size guard.
- Numerical and integration tests, CI configuration, packaged wheel, reproducible
  examples and measured benchmark artifacts.

## Use your recording

For NWB Units, provide explicit analysis intervals and bin width in seconds:

```bash
streamglm import-nwb session.nwb --epoch 10 310 --bin-s 0.005 --out data/session
streamglm pipeline data/session --ranks 0 1 2 --seeds 0 1 \
  --max-iter 800 --out runs/session
```

Repeat `--epoch START STOP` for multiple disjoint intervals. Intervals are
half-open `[start, stop)`; durations must contain an integer number of bins.
Use `--unit-ids` to select known good NWB units explicitly. The importer does
not invent dataset-specific spike-sorting quality thresholds.

The pipeline splits each epoch chronologically 60/20/20 and resets history at
every boundary. Unit filtering uses training data only. Rank and initialization
are selected on validation likelihood among optimizer-converged candidates.
The test set is used only for the selected model and the self-history baseline.
If no candidate converges, the pipeline fails explicitly and retains its progress.

For direct fitting and continuation:

```bash
streamglm fit data/session --rank 2 --solver adam --epochs 10 --out runs/adam
streamglm fit data/session --rank 2 --solver adam --epochs 10 --out runs/adam --resume
streamglm predict runs/adam/model.npz data/session --out predictions.npy
streamglm evaluate runs/adam/model.npz data/test --out test_score.json
```

Reuse the same model/solver arguments when resuming. Adam stores moments, step
count and parameters: continuation at a completed epoch boundary is tested
against an uninterrupted run. L-BFGS resumes accepted parameters but restarts
its optimizer history. Both verify model configuration and a streaming hash of
recording content before resume. A checkpoint is saved after each accepted
L-BFGS iteration or completed Adam epoch.

Direct `evaluate` scores exactly the supplied recording; it does not create a
holdout automatically. Use `pipeline` for automatic held-out evaluation. Saved
unit order is enforced, and prediction output marks incomplete-history bins NaN.
`DONE` means the requested computation finished; inspect `converged` separately.

The Python API also accepts an existing count matrix or memory map:

```python
import jax
jax.config.update("jax_enable_x64", True)
from streamglm import Recording, StreamingGLM
from streamglm.pipeline import exponential_basis

data = Recording.from_npy("counts.npy", epoch_lengths=[100_000, 100_000], bin_s=.001)
model = StreamingGLM(exponential_basis(25, 3), rank=2,
                     chunk_size=1024, ridge=.01, self_history=True)
result = model.fit(data, max_iter=300, checkpoint="checkpoint.npz")
print(result.converged, result.gradient_inf_norm)
for start, expected_counts, valid in model.predict_chunks(result.params, data):
    pass  # consume/write each chunk without concatenating the whole prediction
```

To import Pynapple, call `streamglm.adapters.from_pynapple(units, epochs, bin_s, out)`.
For HNN, use `streamglm import-hnn production_spikes.npz --out data/hnn`.
HNN caches must contain explicit `segment_lengths`; the importer will not guess
the boundaries of concatenated simulations.

## Measured evidence

All results below were run locally on CPU cores 6 and 7 while the original
experiment continued. The timing values are single measurements, not controlled
performance estimates. JSON records include platform and version details.

**Correctness:** 19 test methods cover dense/factorized gradients, NeMoS causal
feature agreement, gaps, short epochs, partial batches, disk input, adapters,
float32 persistence, exact Adam resume, L-BFGS restart and the CLI lifecycle.
Float64 fixed-parameter equivalence tolerances are approximately 1e-10–1e-11.

A small fitted unrestricted GLM independently converged with NeMoS and StreamGLM:
objective difference **3.11e-11**, maximum predicted-count difference **1.12e-5**.
The local NeMoS installation reports 0.2.8 and has no batching module or
`stochastic_fit`; its source hash is recorded. This is not a speed comparison
against newer NeMoS streaming support.

**Predictive evaluation:**

| Recording | Retained neurons | Selected rank | Test gain over self-history, nats/bin |
|---|---:|---:|---:|
| Mouse32: 300 s wake interval, 5 ms bins | 44 | 2 | 0.06400 |
| HNN: first 120 s of cached seed 200, 1 ms bins | 70 | 2 | 0.03079 |

All five candidates converged in each of these two runs. The public-data example
completed in about 66 seconds, including five candidate fits and a baseline fit.
These gains are summed over neurons. They demonstrate prediction under this
split and configuration, not anatomical or causal connectivity recovery.
The HNN result uses a bounded subset, not the full original production dataset.

Three independent eight-neuron synthetic fixtures gave positive held-out gains
over self-history (0.0323–0.0378 nats/bin). Reconstructed-weight relative errors
were 0.378–0.491; predictive success is not exact weight recovery. Selected ranks
were 1, 1 and 2 for rank-2 generators, showing the finite-data selection tradeoff.

**Memory and pass time:** float64, K=3, rank=5, H=25, chunk=512.

| Bins × neurons | Evaluation | Warm pass | Peak process RSS |
|---|---|---:|---:|
| 32,768 × 128 | streamed | 0.275 s | 0.291 GiB |
| 32,768 × 128 | materialized reference | 0.386 s | 0.708 GiB |
| 262,144 × 128 | streamed | 1.623 s | 0.347 GiB |
| 32,768 × 1,000 | streamed | 1.332 s | 0.411 GiB |

The matched materialized/streamed objectives agree to floating-point precision.
Benchmarks evaluate fixed-parameter objective/gradient passes, **not time to fit**.
Peak RSS includes Python/JAX, compilation and mapped file pages. The OS can retain
mapped pages; process RSS is not guaranteed constant with recording length.

Raw artifacts: [real data](artifacts/v2/real_mouse32/report.md),
[HNN](artifacts/v2/validation/hnn_predictive/report.md),
[synthetic summary](artifacts/v2/validation/synthetic_summary.json),
[NeMoS comparison](artifacts/v2/upstream_comparison.json).

## Reproduce

```bash
python -m streamglm.upstream_compare
python -m streamglm.validation_suite
python -m streamglm.real_example
python -m streamglm.benchmark --bins 32768 --neurons 1000 \
  --out artifacts/v2/benchmark_32768_1000_streamed.json
python -m streamglm.plots
```

For fresh end-to-end runs, pass a new `--out` directory; existing reports are
preserved. `validation_suite` can resume completed fixtures; do not reuse its
output directory after changing the experiment configuration. The real example
downloads a 36.6 MB checksum-pinned NWB file only when explicitly requested.
It uses the [Pynapple head-direction tutorial dataset](https://pynapple.org/examples/tutorial_HD_dataset.html)
(Peyrache-2015 Mouse32-140822), **not Neuropixels**. Dataset provenance and exact
analysis interval are saved separately. Raw data stay in ignored `cache/`.

On this machine, prefix commands with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu taskset -c 6,7
```

Choose available CPU IDs on other machines. See [docs/METHODS.md](docs/METHODS.md)
for model conventions and [docs/VALIDATION.md](docs/VALIDATION.md) for limitations.

## Scope and integration

Low rank is a modeling assumption, not an exact replacement for an unrestricted
GLM. Its factors are not identifiable individually, and optimization is nonconvex.
Separate self-history avoids forcing refractory dynamics into shared factors.
No stimulus/behavior covariates, synaptic sign constraints or anatomical edge
inference are provided in this release.

NeMoS already documents disk-backed stochastic fitting in its
[development documentation](https://nemos-neuro.org/en/latest/how_to_guide/batching/stochastic_fit.html).
StreamGLM is currently a standalone prototype, not an upstream extension. The
next integration step is a pinned comparison with that implementation, followed
by a narrowly scoped contribution if maintainers consider it useful.

The worktree is `/home/satvik/side_project`, branch `side-project/streaming-glm`.
Inherited NeuroTwinBench files remain for provenance but are excluded from this
Python distribution. `PROJECT.md` and `CODEX_HANDOFF.md` describe that older
project; use [HANDOFF_STREAMGLM.md](HANDOFF_STREAMGLM.md) for this one.

### Complete-fit benchmarks

The separate `fit_benchmark` command measures a cold train/validation/test
workflow, including rank selection, baseline fitting, and peak process RSS:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu \
taskset -c 6,7 .venv/bin/python -m streamglm.fit_benchmark \
  --bins 20000 --neurons 8 --out artifacts/my_fit_benchmark
```

Use a new output directory for every run and a fresh process for each size.
`--data PATH` instead benchmarks a recording imported with StreamGLM's adapters.
The default uses synthetic data; its time bins are not a Neuropixels recording.
`benchmark.json` records failures as well as successful workflows. Candidate
convergence flags reflect the optimizer's stopping rules; inspect the saved
messages and gradient norms, since relative objective convergence does not
guarantee the gradient tolerance was reached. Timings on a shared host are
observations, not controlled speedup estimates.

For a matched unrestricted-model comparison, run
`python -m streamglm.matched_benchmark --out artifacts/my_matched_benchmark`
in a fresh process with the same CPU/thread settings. This compares StreamGLM
and NeMoS objectives under a common SciPy optimizer, and separately runs native
NeMoS fitting. See [the benchmark protocol](docs/MATCHED_BENCHMARK.md) for
accuracy gates, timing boundaries, provenance, and limitations.
