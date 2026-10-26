# Completed-grid failure audit

All 256 original candidate outcomes and all selected results are preserved.
Classification is based on per-candidate JSON, traceback logs and convergence gates.

| Backend | Converged | Finished, gate failed | Nonfinite objective/gradient | Timeout |
|---|---:|---:|---:|---:|
| dense | 32 | 0 | 0 | 0 |
| lowrank | 110 | 17 | 0 | 1 |
| nemos | 46 | 29 | 21 | 0 |

## Numerical failures

All 21 exceptions are NeMoS SVRG on the permutation-full-rank synthetic family. Sixteen use stepsize 1.0; five use 0.5. Every traceback ends in the independent full-objective check detecting a nonfinite objective/gradient. This supports instability at these tested settings, not a general failure of NeMoS. Smaller stepsizes were already included in the original grid; all eight NeMoS dataset groups selected a converged model. Do not silently replace divergent candidates with changed configurations.

## Timeout

The single timeout is lowrank_n128_s47/candidate04: rank 16, ridge 0, budget 1500. Supervisor wall duration was 1206.7 s against a 900 s limit, while the last saved progress reported 120.1 s and gradient infinity norm 6.969e-5 (above the 1e-5 gate). A system pause or clock discontinuity is possible; the available logs do not establish the cause. An identical-settings, fresh-initialization diagnostic retry is recorded separately in timeout_retry.json. It does not overwrite the original grid or change its selected models.

## Interpretation

188/256 original candidates passed the convergence gate. Another 46 finished but failed that gate; 22 failed execution. These categories must be reported separately. All 24 backend/dataset selections exist, so no dataset/backend comparison was lost. The comparison is conditional on the tested settings, budgets, implementation and two seeds, not a general solver ranking.

## Per-failure evidence

| Dataset | Candidate | Classification | Ridge | Step |
|---|---|---|---:|---:|
| lowrank_n128_s47 | candidate04 | timeout | 0.0 | None |
| permutation_fullrank_n128_s31 | candidate06 | nonfinite_objective_gradient | 0.0 | 0.5 |
| permutation_fullrank_n128_s31 | candidate07 | nonfinite_objective_gradient | 0.0 | 1.0 |
| permutation_fullrank_n128_s31 | candidate14 | nonfinite_objective_gradient | 0.001 | 0.5 |
| permutation_fullrank_n128_s31 | candidate15 | nonfinite_objective_gradient | 0.001 | 1.0 |
| permutation_fullrank_n128_s31 | candidate23 | nonfinite_objective_gradient | 0.01 | 1.0 |
| permutation_fullrank_n128_s31 | candidate31 | nonfinite_objective_gradient | 0.1 | 1.0 |
| permutation_fullrank_n128_s47 | candidate06 | nonfinite_objective_gradient | 0.0 | 0.5 |
| permutation_fullrank_n128_s47 | candidate07 | nonfinite_objective_gradient | 0.0 | 1.0 |
| permutation_fullrank_n128_s47 | candidate14 | nonfinite_objective_gradient | 0.001 | 0.5 |
| permutation_fullrank_n128_s47 | candidate15 | nonfinite_objective_gradient | 0.001 | 1.0 |
| permutation_fullrank_n128_s47 | candidate22 | nonfinite_objective_gradient | 0.01 | 0.5 |
| permutation_fullrank_n128_s47 | candidate23 | nonfinite_objective_gradient | 0.01 | 1.0 |
| permutation_fullrank_n128_s47 | candidate31 | nonfinite_objective_gradient | 0.1 | 1.0 |
| permutation_fullrank_n64_s31 | candidate07 | nonfinite_objective_gradient | 0.0 | 1.0 |
| permutation_fullrank_n64_s31 | candidate15 | nonfinite_objective_gradient | 0.001 | 1.0 |
| permutation_fullrank_n64_s31 | candidate23 | nonfinite_objective_gradient | 0.01 | 1.0 |
| permutation_fullrank_n64_s31 | candidate31 | nonfinite_objective_gradient | 0.1 | 1.0 |
| permutation_fullrank_n64_s47 | candidate07 | nonfinite_objective_gradient | 0.0 | 1.0 |
| permutation_fullrank_n64_s47 | candidate15 | nonfinite_objective_gradient | 0.001 | 1.0 |
| permutation_fullrank_n64_s47 | candidate23 | nonfinite_objective_gradient | 0.01 | 1.0 |
| permutation_fullrank_n64_s47 | candidate31 | nonfinite_objective_gradient | 0.1 | 1.0 |

## Identical-settings retry outcome

The timeout candidate finished in 126.7 seconds on retry, using 0.311 GiB peak RSS. It reached the 1500-iteration limit, with gradient infinity norm 9.87083e-05, above the 1e-5 gate. It remains ineligible. Validation log likelihood was -51.508607, compared with -50.513115 for the original selected rank-2, ridge-0.01 model (higher is better). No held-out test evaluation was performed for the retry, and no selected model or original candidate record changed.

Accounting with the supplementary retry: 188 converged, 47 completed without convergence, 21 numerical failures. Historical original accounting remains 188/46/21/1. This retry resolves the missing outcome without changing the comparison conclusions.
