## Active restart — 2026-09-28

User authorized all pending runs. HNN runner PID 6742, topology c3 fit child
6788; StreamGLM runner PID 5533, report watcher 7607. Verify command lines and
heartbeat before acting; PIDs here are snapshots.

HNN now uses float64 DISK_LBFGS: disk feature maps and chunked analytic gradients.
Validation matched NeMoS loss/gradient and fitted filters (max difference
1.746e-6). Optimization RSS approximately 0.86 GiB. Validation evidence:
results/handoff/disk_fit_validation.json. The 26-task pending_manifest.json
queues topology c3/c4, location refit, two design pilots, six topology jitter
fits, nine proxy refits, three five-basis calibrations and three latency runs.
Existing simulations are reused. Completed commands do not prove convergence.
Shared-drive and swapped-location runs are 20-second feasibility pilots;
full corrected-design scientific experiments remain follow-up work.
Status: results/handoff/status.json. Log: results/handoff/pending_runner.log.
Never start a second HNN runner or MPI simulation concurrently.

StreamGLM resumed the v6 comparison, preserving existing candidate outcomes.
133/256 were recorded at this snapshot, including failures. Status:
/home/satvik/side_project/artifacts/v6/tuned/status.json. Report watcher writes
report.md on completion. Upstream-env suite: 25 tests passed. Wheel rebuilt
locally. No publication or remote CI performed.

Earlier process snapshots below are historical and superseded.

# StreamGLM handoff

Work here: `/home/satvik/side_project`, branch `side-project/streaming-glm`.
Read README.md for current scope, commands, results and limitations. The older
PROJECT.md and CODEX_HANDOFF.md belong to the inherited NeuroTwinBench project.

The user authorized starting this separate project, including local worktree
creation and implementation. Do not publish or contact maintainers without an
explicit request. Do not stop the main runner as a side effect of this work.

## Latest: release 0.3, 2026-09-27

Validation runs complete: artifacts/v4/report.md and docs/RELEASE_03.md.
Real VISp: 98/185 retained units in overlapping 300-second subsets; gains over
self-history 0.088/0.196 nats/bin. Synthetic 1,000-unit hour at 10 ms: 1.24 GiB,
27.1 minutes, no true coupling. Real 1,000-unit hour-long fit remains a stretch goal.
Current NeMoS already has streaming support; our loader and numerical check use
commit 81c7200a0e66686e98fd1907e7fa11e111a0e66a. SVRG met its gradient gate at
pass 1300; the initial 300-pass miss is retained. All 23 tests passed in .venv
(Python 3.11) and cache/upstream/venv (Python 3.12). A new independent-seed replication was launched on 2026-09-27: PID 94931,
artifacts/v4/null_1000_1h_seed733/status.json. It repeats the 1,000-unit,
one-hour 10 ms null with seed 733 on cores 6,7; verify PID and status before acting. Remote CI and upstream acceptance have not happened.
Earlier release notes below are historical.

## Completed release 0.2

- `streamglm/data.py`: replayable arrays/memory maps with epoch-local history.
- `streamglm/model.py`: dense and low-rank Poisson models, exact streamed
  value/gradient, chunked predictions and bounded-data L-BFGS objective.
- `tests_streamglm/`: independent materialized and NeMoS oracle checks.
- `streamglm/demo.py`: known low-rank recurrent simulation, fitting and held-out
  prediction with explicit convergence/parameter-error reporting.
- `streamglm/benchmark.py`: fresh-process disk-backed objective/gradient benchmark.
- Local editable install in `.venv`, basic CI workflow, measured JSON artifacts.
- NWB/Pynapple/HNN import, lazy splits and column selection, separate self-history.
- Exact epoch-boundary Adam resume, parameter-only L-BFGS restart, content hashes.
- CLI import/fit/resume/evaluate/predict/pipeline, rank selection with multiple
  initializations and held-out test reports.
- Three synthetic seeds, cached HNN predictive demo and public Mouse32 NWB demo.
- Fitted NeMoS comparison, larger fresh-process benchmarks and exported figures.

Current docs: README.md, docs/METHODS.md, docs/VALIDATION.md. Raw input is in
ignored cache/; measured results and small fitted models are in artifacts/v2/.
The original prototype README is preserved in docs/PROTOTYPE_README.md.

## Environment and resource budget

Main experiment uses `/home/satvik/hnn_neuro_bridge` and conda env `neurotwin`.
Local `.venv` shares installed dependencies read-only, with its own editable
package installation. Use cores 6,7 and small data while the main queue is active:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu \
taskset -c 6,7 .venv/bin/python -m unittest discover -s tests_streamglm -v
```

Validated numerical environment: Python 3.11, JAX 0.10.2, NumPy 2.4.6,
SciPy 1.17.1. Local NeMoS reports 0.2.8 but has a newer API signature;
the oracle detects the inverse-link argument location rather than trusting
version metadata. Clean released-version CI remains to be verified.

## Scope beyond this release

2026-09-26 follow-up: `streamglm.fit_benchmark` now records cold complete-workflow
runtime, peak RSS, selection diagnostics and held-out scores, with failure records
and overwrite protection. Two small synthetic runs are saved in `artifacts/v3/`;
see docs/VALIDATION.md. These are not Neuropixels or matched NeMoS performance
comparisons. A regression test covers failed-run reporting and preservation.

The subsequent `streamglm.matched_benchmark` experiment compares unrestricted
StreamGLM and NeMoS objectives with a common SciPy optimizer, plus native NeMoS
fits in separate processes. Protocol: docs/MATCHED_BENCHMARK.md. Final data:
artifacts/v3/matched_dense/summary.json and report.md. Consult both optimizer
status and agreement gates; a finished suite does not mean every gate passed.
The initial matched_smoke output predates a likelihood-constant precision fix.

Matched suite completed: 27 fits, all passed gradient accuracy; 16/18 pairwise
agreement gates passed. The two misses are the same 4,000-bin seed (11), with
held-out LL differences just above 1e-6; retained without changing thresholds.
64,000-bin median common-driver results: streamed 25.15s/0.299 GiB, NeMoS
objective 14.97s/0.867 GiB. No universal speedup claim. Full local test suite:
22 tests passed. Figures/report were enhanced after computation; separate
report_provenance.json records that source hash without changing fit provenance.

1. Pin and compare upstream NeMoS batching, including handling of epoch gaps.
2. Validate a quality-filtered Neuropixels session (current public demo is
   head-direction data, not Neuropixels).
3. Extend fresh-process memory benchmarks over T, N and K; distinguish memory
   mapping residency, compiled workspace and parameter/optimizer memory.
4. Add external stimulus/behavior covariates and more statistical controls.
5. Broad synthetic calibration and repeated time-to-convergence benchmarks.

No 1,000-neuron multi-hour fit, anatomical recovery result, or benchmark against
upstream's newer streaming code has been completed. The 1,000-neuron benchmark
measures fixed-parameter objective/gradient passes only.

## Three-way comparison queued 2026-09-27

User requested the actual NeMoS-streaming / StreamGLM-dense / StreamGLM-lowrank
comparison. Protocol: docs/THREE_WAY.md; implementation: streamglm/three_way.py.
Six tiny smoke workers completed without errors (10-step budget, deliberately
not convergence certification); generator rank/causality/determinism checks passed.
Detached queue PID 100846 waits for null replicate PID 94931 to exit, then runs
54 workers on cores 6/7 using pinned upstream environment. Verify live PIDs and
artifacts/v5/three_way_queue.json; suite heartbeat is artifacts/v5/three_way/status.json.
Do not launch a duplicate. Results accumulate in summary.json, including failures.
Budget 15 minutes per worker, at most about 13.5 hours of worker time for the grid;
actual runtime unknown. Fixed hyperparameters, single low-rank start, no claims
of superiority from unconverged fits. No 1,000-neuron comparative fit yet.

## Tuned comparison launched 2026-09-28

v5 completed 54/54 workers; 41 passed gradient gate (dense18, lowrank17, NeMoS6).
User requested regularization + rank selection at 64/128 with tuned NeMoS.
New module streamglm/tuned_comparison.py; protocol docs/TUNED_COMPARISON.md.
Detached PID 265412 (verify live PID and heartbeat). Output artifacts/v6/tuned,
log artifacts/v6/tuned.log, launch manifest artifacts/v6/launch.json. Do not duplicate.
256 candidates: 2 fresh seeds (31/47), 2 populations (64/128), 2 wiring families,
4 ridges; dense4 + lowrank16 + NeMoS12 per dataset. Validation-only selection
among gradient-gate passes, then selected-model test evaluation; no refit.
All candidates/errors retained. Worst-case worker limits sum to ~64 hours.
Two new unit tests and six fixed tiny execution checks passed; selected-model
held-out evaluator also tested. Initial smoke API failure is retained in
artifacts/v6/smoke; corrected run in smoke_fixed. Current upstream penalty
normalization is checked independently inside each NeMoS worker.

NeuroTwinBench separately: topology_c3 simulation completed; fit failed its
memory guard (21.26 GiB RSS, 1.27 GiB available). Main queue stopped before c4.
Do not repeat the unchanged heavy fit. Cached simulations should be preserved.

Failure audit completed: artifacts/v6/failure_audit/README.md. All 21 NeMoS failures are nonfinite objective/gradient at large tested stepsizes. The single timeout was retried identically and finished in 127s but failed the convergence gate; original results and selections preserved. See detailed audit for accounting.

## Phase 2 started 2026-09-29 (supersedes earlier process snapshots)

HNN runner82671: results/phase2/status.json plus results/handoff/status.json heartbeat.14 tasks include conditional uncertainty,6 new topology circuit/drive recordings,3 corrected shared-input contrasts,3 paired location contrasts,and final aggregation. No concurrent HNN MPI. StreamGLM runner93641: artifacts/v8/queue_status.json; coupled1000 seeds941/942 then streamed-vs-streamed benchmark,which waits for HNN to exit for timing isolation. Source protocols in results/phase2/PROTOCOL.md and side_project/artifacts/v8/PROTOCOL.md. Verify PIDs and heartbeat; snapshots can stale.26 StreamGLM tests passed and3 new HNN tests passed. New scientific runs are not yet complete.
