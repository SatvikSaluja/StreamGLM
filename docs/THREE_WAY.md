# Three-way synthetic comparison

Run `streamglm.three_way` in `cache/upstream/venv`, which pins NeMoS commit
81c7200a0e66686e98fd1907e7fa11e111a0e66a. All arms share that numerical environment.

The initial grid is 16/64/128 neurons, 12,000 bins, seeds 7/11/19, and two
recurrent Poisson ground truths: rank-2 inhibitory coupling and full-rank
permutation inhibitory coupling. Both have bounded conditional rates without
clipping the exponential link. Bins have no assigned physical duration.

Each dataset is shared across StreamGLM dense L-BFGS, StreamGLM rank-2 L-BFGS,
and current NeMoS streamed SVRG using the tested epoch-safe custom loader.
All use two identical basis functions, float64, no regularization, and a
chronological 80/20 train/test split with history reset. Dense starts are zero;
factorized starts use fixed seed 0 (zero factors would have zero gradients).
Rank 2 is specified in advance, including for the full-rank family.

Each arm runs in a fresh process, restricted to cores 6/7. Record whole-worker
peak RSS, preparation/fit/scoring runtime, held-out gain over training constant
rates, oracle likelihood, filter correlation and relative error. The parent
also records elapsed time and preserves timeouts/failures. Model outputs and
per-evaluation/pass progress are saved. Suite status has PID and heartbeat.

Budget: 1,500 L-BFGS iterations or SVRG passes (not equal compute); a 900-second
wall limit per worker. Dense/low-rank L-BFGS uses tolerance 1e-12, NeMoS SVRG
uses fixed stepsize 0.5 and a full-gradient check every ten passes. Final gradient
infinity norms are recorded, with a 1e-5 diagnostic gate. Low-rank gradient norms
are in different coordinates and do not certify a global optimum. There is
one low-rank initialization; this is an exploratory baseline, not exhaustive
tuning of any method. Rank knowledge favors low-rank fits in that family.

Do not count an unconverged or timed-out competitor as evidence of superior
speed. Compare prediction and convergence alongside memory/runtime. Family
strengths differ, so compare methods within each dataset. HNN shares the host;
these are not dedicated-machine timing measurements. This grid is a first
comparison, not a 1,000-neuron test. Preserve every failed run.
