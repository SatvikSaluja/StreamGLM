# Validation record and limits

## Tested behavior

- Independent NumPy/JAX materialized reference for likelihood, gradient and rates.
- Independent NeMoS causal convolution, likelihood and fixed-parameter gradients.
- Independently converged small dense fits against the installed NeMoS version.
- Histories shorter/longer than chunks, short epochs and partial final batches.
- No current-bin or cross-epoch spike leakage.
- Read-only memory maps, lazy temporal/column selection and split boundaries.
- Half-open spike binning, large absolute timestamp edges, NWB and Pynapple input.
- HNN NPZ extraction preserving known segmentation.
- Dense and factorized separate self-history, including rank zero.
- Pickle-free model round trips and explicit float32 persistence under x64 mode.
- Adam resume identical to uninterrupted fitting; mismatched data rejected.
- L-BFGS parameter restart and the CLI fit/resume/predict/evaluate lifecycle.

The integration tests create tiny NWB files locally; they do not require network
access. The real-data example is explicit and separate from unit tests.

## Experiments run

1. Three 20,000-bin, eight-neuron inhibitory low-rank synthetic fixtures.
2. A 120,000-bin, 70-neuron prefix from the cached HNN seed-200 recording.
3. A fixed 300-second wake interval (8813–9113 s), 5 ms bins, from public Mouse32
   head-direction data. All 49 units are imported; 44 meet the training-rate
   threshold of 0.1 Hz. No additional spike-sorting quality criterion is implied.
4. Fresh-process streamed/materialized fixed-parameter memory comparisons and
   a 1,000-neuron streamed pass benchmark.

Each predictive pipeline has a chronological 60/20/20 train/validation/test
split, separate self-history coefficients, fixed ridge, and rank candidates
0,1,2. Rank 1 and 2 each use two initializations. Hyperparameter grids were
fixed before examining test scores. Results are single-session examples,
not population-level statistical significance or independent animal replication.

## Complete-workflow measurements (2026-09-26)

`artifacts/v3/` adds fresh-process float64 synthetic runs with chronological
60/20/20 splits, ranks 0/1/2, two initializations for nonzero ranks, and a
self-history baseline. Generation, fitting, selection and scoring are included
in total time. Peak RSS includes the entire process. These runs shared the host
with HNN simulation; the second also overlapped the test suite briefly.

| Bins | Neurons | Total seconds | Peak GiB | Test gain over self-history (nats/bin) |
|---:|---:|---:|---:|---:|
| 20,000 | 8 | 16.34 | 0.357 | 0.02863 |
| 40,000 | 32 | 36.50 | 0.367 | 0.04350 |

All five candidates in each run reported optimizer convergence. This includes
relative-objective stopping, not necessarily satisfaction of gradient tolerance;
individual messages and gradient norms are retained. These are single-seed
workflow checks, not a controlled scaling law, speedup, or power calibration.
Changing neuron count also changes the synthetic generator's firing statistics.

## Matched unrestricted fits (2026-09-26)

Protocol: [MATCHED_BENCHMARK.md](MATCHED_BENCHMARK.md). Complete table, numerical
agreement gates, and figure: `artifacts/v3/matched_dense/`. Nine synthetic
datasets (16 neurons; three recording lengths and three data seeds) were each
fit in three fresh processes, totaling 27 fits. All fit gradient infinity norms
passed the independent 1e-5 accuracy threshold.

Median measurements across three data seeds, using the **same SciPy optimizer**:

| Bins | StreamGLM seconds | NeMoS objective seconds | StreamGLM peak GiB | NeMoS objective peak GiB |
|---:|---:|---:|---:|---:|
| 4,000 | 2.76 | 3.39 | 0.284 | 0.514 |
| 16,000 | 7.74 | 5.96 | 0.292 | 0.586 |
| 64,000 | 25.15 | 14.97 | 0.299 | 0.867 |

These measurements show a memory/runtime tradeoff in the tested configurations:
lower process memory for StreamGLM, but slower execution on the two longer
recordings under the common driver. Whole-process memory includes the libraries'
different import overheads. The native NeMoS reference is reported separately;
its measurement includes an additional objective-audit compilation.

16 of 18 pairwise agreement checks passed all thresholds. Both misses involved
the 4,000-bin, seed-11 dataset: held-out likelihood differences were 1.16e-6 and
1.27e-6 nats/bin, slightly exceeding the predeclared 1e-6 threshold. Their fitted
objective differences were below 5e-11 and maximum predicted-count differences
below 9e-6. These misses remain failures in the report; there is no claim that
every fit matches to machine precision. No failures were removed or thresholds
relaxed. Tests additionally cover integer-input likelihood precision and
preservation of iteration-limit/failure diagnostics.

## Not established

- No complete hour-long, 1,000-neuron fitted real-data model.
- No new anatomical-connectivity recovery claim on HNN or real data.
- No statistical certification that rank 2 is the true biological rank.
- No controlled performance comparison against current NeMoS `stochastic_fit`.
  Release 0.3 includes a pinned-upstream loader and numerical accuracy check.
- No stimulus/behavior-adjusted or causally identified coupling estimate.
- No promise that total process RSS stays constant: mapped input pages can
  remain resident even when feature and differentiation buffers are bounded.
- No controlled speedup estimate: another experiment shared the host, and
  timing measurements were not repeated across dedicated machines.
- CI configuration is included, but remote CI has not run because no branch
  has been pushed. Local wheel building and tests are separate verified checks.

The public dataset is linked by the Pynapple tutorial and downloaded with its
published SHA-256. Raw data are excluded from Git. See `dataset.json` in the
real-data artifact directory for provenance and exact analysis window.

## Release 0.3

See [measured results](../artifacts/v4/report.md) and [protocol](RELEASE_03.md).
Both overlapping five-minute VISp subsets improved held-out likelihood over
self-history. The 1,000-unit synthetic hour at 10 ms finished at 1.24 GiB peak
RSS in 27.1 minutes. This is not a real-data 1,000-unit fit or anatomical recovery.
Relative-objective optimizer convergence does not imply every fit passed a
strict gradient gate. The pinned NeMoS SVRG check met its 1e-5 gradient gate;
the earlier 300-pass miss is retained. All 23 tests passed in both environments.
