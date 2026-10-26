# StreamGLM 0.3 validation protocol

The release adds reproducible Neuropixels predictive workflows, a complete
synthetic scale test, and a loader for current NeMoS stochastic fitting.
Measured results are generated in `artifacts/v4/report.md` and
`artifacts/v4/release_report.json`; individual failed or budget-limited runs
remain available beside successful follow-ups.

## Real data

Data come from the [CCN February 2026 Visual Coding workshop](https://flatironinstitute.github.io/neurorse-workshops/workshops/feb-2026/branch/main/users/group_projects/05_visual_coding-users.html).
Its [download registry](https://github.com/flatironinstitute/ccn-software-feb-2026/blob/main/src/workshop_utils/fetch.py)
provides the URL and SHA-256 used by `streamglm.neuropixels`.
The archive contains 243 VISp units, including 195 labelled good, and flash
stimulus intervals. It is an extract of Allen Visual Coding Neuropixels data,
not a new recording, brain-wide dataset, or complete NWB session.

We take the first 100 and all 195 good VISp units in ascending ID order. Selection
does not use held-out activity, except that the provider's quality labels were
computed for the original recording. No waveform metrics are present for an
independent quality reanalysis. The analysis window is [1285, 1585) seconds,
covering five minutes around the flash stimulus block; bin width is 5 ms.

The workflow splits the window chronologically 60/20/20, resets history at
boundaries, and removes units with no training spikes. Three exponential basis
functions cover 125 ms. Candidate ranks are 0/2/5, ridge is .01, and nonzero
ranks each get initializations 0 and 1. The original pipeline's optimizer
relative-objective tolerance is 1e-7 with a 500-iteration budget. Rank selection
uses validation likelihood among optimizer-converged candidates; the test set
is used afterward. Gradient norms are reported separately. Neither positive
predictive gain nor optimizer success establishes biological connectivity.

```bash
python -m streamglm.neuropixels --neurons 100 --out cache/allen_visp_100_300s
python -m streamglm.fit_benchmark --data cache/allen_visp_100_300s \
  --ranks 0 2 5 --out artifacts/v4/allen_visp_100_300s
python -m streamglm.neuropixels --neurons 195 --out cache/allen_visp_195_300s
python -m streamglm.fit_benchmark --data cache/allen_visp_195_300s \
  --ranks 0 2 5 --out artifacts/v4/allen_visp_195_300s
```

Use fresh output directories. Runs here set `OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1`, `JAX_PLATFORMS=cpu`, and CPU affinity 6,7. Those particular
core numbers are machine-specific. The HNN simulation and validation jobs share
the host, so timing is descriptive, not a controlled speed comparison.

## Synthetic scale test

```bash
python -m streamglm.large_fit --out artifacts/v4/null_1000_1h \
  --counts cache/null_1000_1h.npy
```

Defaults generate 1,000 independent 4-Hz Poisson units for one hour at **10 ms**
resolution (360,000 bins), written as an int16 memory map. This is an easy null,
deliberately separate from biological data or coupled-GLM recovery calibration.
There is no true cross-neuron coupling. The experiment uses chronological
60/20/20 partitions, leaves the middle partition unused, and independently fits
rank 0 and rank 5 on the training partition. It performs no hyperparameter
selection and reports both held-out scores. Thus an hour of recording is
processed, but only its 60% training partition is used for optimization.

Models have separate self-history, 3 basis functions spanning 100 ms, ridge .01,
chunk size 2,048, float64, and a 100-iteration L-BFGS budget. The generator, fit,
gradient diagnostics, test scores, checkpoints and whole-process peak RSS are
saved. An optimizer stop is not a declaration of a global optimum. RSS includes
mapped input pages and compilation. The hypothetical dense design size refers
to the full recording and is arithmetic, not a measured OOM comparison.

## Current upstream compatibility

Pinned source: NeMoS commit `81c7200a0e66686e98fd1907e7fa11e111a0e66a`.
It requires Python >=3.12, so it was installed in `cache/upstream/venv`, leaving
the HNN environment and the original StreamGLM environment unchanged. The shallow
checkout reports version `0.0.1.dev1`; the commit and saved dependency lock are
the meaningful provenance. See `artifacts/v4/upstream_environment.txt`.

Current NeMoS [already supports stochastic fitting](https://nemos-neuro.org/en/latest/how_to_guide/batching/stochastic_fit.html)
and [custom convolutional loaders with recording gaps](https://nemos-neuro.org/en/latest/how_to_guide/batching/custom_dataloader.html).
Do not claim streaming as an upstream capability gap. `NeMoSRecordingLoader`
bridges StreamGLM recordings into this interface; it preserves short tails and
excludes exactly the incomplete-history rows of each epoch. A regression test
compares every emitted row against independent whole-epoch NeMoS convolution.

```bash
cache/upstream/venv/bin/python -m streamglm.upstream_streaming \
  --passes 1500 --out artifacts/v4/upstream_streaming_1500
```

This uses an unrestricted six-unit synthetic model, two training epochs, and
SVRG with stepsize .5. An independent full-gradient check every ten passes stops
at norm <=1e-5. The initial 300-pass budget did not pass; the unchanged longer
configuration reached the threshold at pass 1,300. Objective difference from
the StreamGLM L-BFGS reference was 9.25e-7. Held-out likelihoods differ by about
6.51e-5 nats/bin, so this is not a claim of machine-precision fitted predictions.
This shared-process experiment is not a solver timing or memory comparison.

## Packaging and remaining scope

The active CI config tests supported workflows, builds a wheel, and separately
tests the pinned upstream environment. The inherited NeuroTwinBench CI file is
archived as `docs/NEUROTWINBENCH_CI.yml`. Local checks and wheel installation are
distinct from remote CI: no remote CI or publication has been performed.

Still outside this release: 1,000-neuron hour-long real Neuropixels fitting,
1 ms scaling at that size, stimulus/behavior-adjusted interpretation, biological
replication, and accepted upstream contributions. The source and measurements
provide a usable research-software artifact without those larger claims.
