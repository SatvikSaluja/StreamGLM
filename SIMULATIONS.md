# StreamGLM: experiments and command guide

Repository: https://github.com/SatvikSaluja/StreamGLM . Run commands from its root. Results and limitations: [RESULTS_REPORT.md](RESULTS_REPORT.md). Commands below are individual tasks, not an instruction to launch them all simultaneously.

## Setup

Local validated interpreter: `cache/upstream/venv/bin/python`. Activate it before running:

```bash
source cache/upstream/venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pip install -e .
```

For a fresh checkout create a Python 3.11 virtual environment, install the project and the dependencies in pyproject.toml. Historical streaming comparison pins NeMoS commit `81c7200a0e66686e98fd1907e7fa11e111a0e66a`; package names alone do not reproduce that API. Check [constraints-validated.txt](constraints-validated.txt) and each saved protocol. Several older scripts assume the original local interpreter/cache layout; review before moving machines. Use CPU cores 6–7 if sharing the original laptop with HNN; do not benchmark runtime on a competing host.

## Main pending work: coupled 1,000-neuron benchmark

Executable: [streamglm/coupled_scale.py](streamglm/coupled_scale.py). Two generated seeds, 360,000 bins at 10 ms (one hour), inhibitory block rank-4 truth, two basis functions, eight-bin causal history. This structured generator is not arbitrary brain-wide connectivity.

```bash
python -u -m streamglm.coupled_scale --seed 941 --neurons 1000 --bins 360000 --out artifacts/v8/coupled1000_seed941
python -u -m streamglm.coupled_scale --seed 942 --neurons 1000 --bins 360000 --out artifacts/v8/coupled1000_seed942
```

Run sequentially. Seed 941: candidate0–5 JSON/NPZ completed, checkpoint6 exists. Seed 942: generated data exist; no completed fit report at snapshot. Each seed fits self-history and ranks 4/8 × ridge .001/.01 × two initializations. Selection uses validation and gradient norm <=1e-5, then evaluates test likelihood, oracle comparison, off-diagonal truth recovery, edge detection, 20 target-label permutations and held-out time shift. A shifted-data score is not a surrogate refit.

Same command/output path resumes completed candidates and saved parameters. L-BFGS history restarts. Generated data and checkpoints must be retained locally; a fresh clone does not contain them. `status.json` is a checkpoint, not a liveness check. Look for `report.json` and inspect convergence before claiming success. Previous candidates took 2–9 hours each; no five-hour completion promise is justified.

## Pending streaming-versus-streaming comparison

```bash
python -u -m streamglm.streaming_benchmark --out artifacts/v8/streaming_comparison
```

Run on an otherwise idle machine after HNN finishes. The protocol covers 64 neurons; seeds 31/47; 24k/96k/384k bins; two generating families; StreamGLM exact chunked optimization versus NeMoS loader SVRG at two step sizes. 36 worker outcomes, all failures retained, independent gradient gate. Counts are resident; neither materializes the full design. It is not fully out-of-core input loading or a universal library benchmark.

## Completed tuned synthetic benchmark and new confirmation

```bash
# Existing results: artifacts/v6/tuned; --resume retains completed work.
python -m streamglm.tuned_comparison --neurons 64 128 --seeds 31 47 --out artifacts/v6/tuned --resume
# New seeds, unchanged selection protocol. Dry-run first:
python -m streamglm.pending_runs confirmation
python -m streamglm.pending_runs confirmation --run
```

Confirmation uses seeds 59/71 and a fresh output directory. It is supplied as code in [streamglm/pending_runs.py](streamglm/pending_runs.py); no confirmation result is claimed. Rank/penalty choices must be validation-selected; do not tune against final test/truth scores.

## Other experiments and outputs

| Command / module | Purpose | Existing evidence |
|---|---|---|
| `python -m streamglm.validation_suite` | Small synthetic fixtures, optional HNN predictive input via --hnn | artifacts/v2/validation |
| `python -m streamglm.demo` | Small demonstration | artifacts/demo.json |
| `python -m streamglm.matched_benchmark --out artifacts/reproduce/matched` | Shared-optimizer and native NeMoS comparisons | artifacts/v3/matched_dense |
| `python -m streamglm.three_way --out artifacts/reproduce/three_way` | Dense / low rank / NeMoS before tuned study | artifacts/v5/three_way |
| `python -m streamglm.benchmark --bins 32768 --neurons 1000 --out artifacts/reproduce/memory.json` | Evaluation memory/pass-time, not full recovery | artifacts/v2/benchmark_* |
| `python -m streamglm.upstream_streaming --passes 1500 --out artifacts/reproduce/upstream` | Loader integration with pinned upstream | artifacts/v4/upstream_streaming_1500 |
| `python -m streamglm.large_fit --counts artifacts/reproduce/null_counts.npy --out artifacts/reproduce/null1000 --seed 732` | Independent Poisson null, 1,000 neurons | artifacts/v4/null_1000_1h (seed733 replicate exists) |
| `python -m streamglm.neuropixels --neurons 100 --duration 300 --out artifacts/reproduce/allen100` | Public Allen data import and fitting | artifacts/v4/allen_visp_100_300s |
| `python -m streamglm.neuropixels --neurons 195 --duration 300 --out artifacts/reproduce/allen195` | Second cohort from same recording | artifacts/v4/allen_visp_195_300s |
| `python -m streamglm.coupled_scale --neurons 12 --bins 2000 --seed 941 --out artifacts/reproduce/coupled_smoke` | Small code-path smoke only | artifacts/v8/smoke_coupled |

All modules have command-line help where arguments are supported. Public data commands need network access and storage. Real session replication requires selecting a separate suitable NWB session; importing another cohort from the same session is not independent replication. Use the documented `streamglm import-nwb` and `streamglm pipeline` commands in README for user-supplied sessions.

Historical scripts `run_phase2` and `finalize_pending` orchestrate broad queues and use local paths/status files. Prefer explicit modules above; the historical HNN wait condition can observe a stale runner status. Low-rank-plus-sparse coupling is a proposed architecture, not an implemented simulation recipe. Separate self-history already exists and should not be advertised as a new contribution.

## Publishing and saved evidence

Commit code, protocols and compact reports. Never stage whole cache/artifacts folders without review. Count arrays, NPZ models/checkpoints, downloaded NWB files and logs are deliberately excluded from new publication bundles. The linked reports in results_page/sources let readers inspect all webpage numbers without hundreds of GB of arrays.
