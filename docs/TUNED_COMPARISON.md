# Validation-selected regularization, rank and NeMoS step size

This follows the unregularized v5 experiment; it is a new experiment, not a
replacement of inconvenient results. Fresh generator seeds 31 and 47, neuron
counts 64 and 128, both rank-2 and full-rank permutation inhibitory ground truths.
Each dataset has 12,000 bins with chronological 60/20/20 train/validation/test
partitions. History resets at each partition; bins have no assigned physical time.

Each of the eight datasets has 32 candidates:

- StreamGLM dense: ridge 0, 0.001, 0.01, 0.1 (4 fits).
- StreamGLM factorized: the same ridge grid crossed with ranks 2, 4, 8, 16 (16 fits).
- NeMoS streamed SVRG: the same ridge grid crossed with stepsizes 0.1, 0.5, 1 (12 fits).

Use float64 and the same two history basis functions. Ridge penalizes the
materialized coupling weights by 0.5 * ridge * sum(W**2), not factor norms;
intercepts are unpenalized. Both losses average over time and sum over neurons.
Every NeMoS worker checks a nonzero-parameter penalized value and gradient
against an explicit calculation before fitting. Factorized/dense objective
penalty equivalence and validation-only selection have unit tests.

Every candidate uses its own fresh process. L-BFGS budget is 1,500 iterations;
SVRG budget is 1,500 passes; neither is an equal-compute budget. A 900-second
worker limit applies to both. L-BFGS tolerance is 1e-12. NeMoS checks the full
penalized gradient every ten passes and stops at infinity norm <= 1e-5. All
candidates get a final full-gradient check. Low-rank starts use seed 0; there
are no multiple restarts in this grid. A small gradient does not establish a
low-rank global optimum and is coordinate dependent.

Within each backend, select the largest validation log-likelihood among
candidates passing the gradient gate. No eligible candidate means no reported
winner for that backend/dataset. Freeze selection to JSON before evaluating
its held-out test and known coupling filters. Candidate workers do not read
truth files or score test data. Selected fits are not refitted. Oracle likelihood,
filter correlation and relative error are reported only after selection.

Report constant-rate predictive baseline, selected test likelihood/bits per
spike, filter recovery, final gradient, chosen hyperparameters, peak process
RSS, selected fit runtime AND total tuning runtime. Different search sizes
must not be concealed. All failed candidates, timeouts and misses are retained.
Both synthetic families differ in structure/strength; compare methods within
each dataset. Compare against v5 cautiously: fresh seeds and different split.

This is a limited grid, not globally optimal tuning of any library. Two seeds
are exploratory evidence, not a strong significance claim. If winners reach
grid boundaries, report that and plan a new confirmation experiment; do not
adapt the grid to test results. No claims of current-NeMoS supremacy or
1,000-neuron performance follow from this experiment alone.

Run in cache/upstream/venv (NeMoS commit
81c7200a0e66686e98fd1907e7fa11e111a0e66a), CPU cores 6/7, one BLAS thread:

    python -m streamglm.tuned_comparison --out artifacts/v6/tuned

Progress: status.json; all tuning outcomes: candidates.json; selected test
results: selected.json. Protocol includes source hashes. The 256 candidate
limits permit about 64 hours plus generation/evaluation overhead in the worst
case. Actual runtime should be measured, not predicted from iteration counts.
