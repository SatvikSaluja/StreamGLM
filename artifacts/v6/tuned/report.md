# Tuned StreamGLM / NeMoS comparison

Fixed protocol: 60/20/20 chronological splits, fresh seeds, validation-only selection among gradient-gate-passing candidates. Failed settings remain in candidates.json. Two seeds are exploratory, not a significance certification.

| Dataset | Model | Rank | Ridge | Step | Test bits/spike | Filter correlation | Relative filter error | Selected fit seconds | Tuning seconds | Peak GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lowrank_n64_s31 | dense | None | 0.1 | None | 0.000854251 | 0.2928 | 0.9599 | 2.8 | 48.7 | 0.306 |
| lowrank_n64_s31 | lowrank | 2 | 0.01 | None | 0.00654007 | 0.7587 | 0.7124 | 16.1 | 789.4 | 0.296 |
| lowrank_n64_s31 | nemos | None | 0.1 | 1.0 | 0.000853999 | 0.2928 | 0.9599 | 13.6 | 1696.8 | 0.488 |
| permutation_fullrank_n64_s31 | dense | None | 0.01 | None | 0.0312326 | 0.7866 | 0.6167 | 9.0 | 60.4 | 0.305 |
| permutation_fullrank_n64_s31 | lowrank | 16 | 0.01 | None | 0.0220035 | 0.6426 | 0.7618 | 49.1 | 836.5 | 0.294 |
| permutation_fullrank_n64_s31 | nemos | None | 0.01 | 0.1 | 0.0312312 | 0.7866 | 0.6167 | 215.1 | 1307.8 | 0.494 |
| lowrank_n64_s47 | dense | None | 0.1 | None | 0.000958415 | 0.2950 | 0.9606 | 3.2 | 57.3 | 0.306 |
| lowrank_n64_s47 | lowrank | 2 | 0.01 | None | 0.00669658 | 0.7698 | 0.7128 | 15.7 | 803.6 | 0.296 |
| lowrank_n64_s47 | nemos | None | 0.1 | 1.0 | 0.000958225 | 0.2950 | 0.9606 | 13.7 | 1693.5 | 0.482 |
| permutation_fullrank_n64_s47 | dense | None | 0.01 | None | 0.0313343 | 0.7911 | 0.6121 | 13.0 | 63.4 | 0.304 |
| permutation_fullrank_n64_s47 | lowrank | 16 | 0.01 | None | 0.0222147 | 0.6502 | 0.7568 | 65.7 | 752.1 | 0.296 |
| permutation_fullrank_n64_s47 | nemos | None | 0.01 | 0.5 | 0.0313339 | 0.7911 | 0.6122 | 50.3 | 1307.1 | 0.490 |
| lowrank_n128_s31 | dense | None | 0.1 | None | 2.72649e-06 | 0.1545 | 0.9766 | 6.8 | 129.3 | 0.328 |
| lowrank_n128_s31 | lowrank | 2 | 0.01 | None | 0.00240575 | 0.6027 | 0.7728 | 31.2 | 1141.3 | 0.302 |
| lowrank_n128_s31 | nemos | None | 0.1 | 1.0 | 3.13195e-08 | 0.1538 | 0.9768 | 19.1 | 3761.1 | 0.519 |
| permutation_fullrank_n128_s31 | dense | None | 0.1 | None | 0.0140076 | 0.6445 | 0.8423 | 10.0 | 162.0 | 0.328 |
| permutation_fullrank_n128_s31 | lowrank | 16 | 0.1 | None | 0.00478865 | 0.3736 | 0.9510 | 37.5 | 1216.0 | 0.307 |
| permutation_fullrank_n128_s31 | nemos | None | 0.1 | 0.1 | 0.0140068 | 0.6445 | 0.8423 | 191.8 | 3256.3 | 0.517 |
| lowrank_n128_s47 | dense | None | 0.1 | None | 1.22765e-05 | 0.1495 | 0.9787 | 6.5 | 149.1 | 0.336 |
| lowrank_n128_s47 | lowrank | 2 | 0.01 | None | 0.00253489 | 0.5837 | 0.7937 | 43.4 | 2208.4 | 0.304 |
| lowrank_n128_s47 | nemos | None | 0.1 | 1.0 | 9.81227e-06 | 0.1490 | 0.9789 | 14.3 | 3426.8 | 0.489 |
| permutation_fullrank_n128_s47 | dense | None | 0.1 | None | 0.0142954 | 0.6480 | 0.8421 | 12.5 | 186.0 | 0.328 |
| permutation_fullrank_n128_s47 | lowrank | 16 | 0.1 | None | 0.00503344 | 0.3804 | 0.9502 | 49.2 | 1509.5 | 0.308 |
| permutation_fullrank_n128_s47 | nemos | None | 0.1 | 0.1 | 0.0142946 | 0.6480 | 0.8421 | 168.1 | 2850.2 | 0.512 |

Candidate outcomes: 256; execution failures: 22; gradient passes: 188.

Bits/spike are gains over training constant-rate predictions. Correlations are Pearson correlations of reconstructed filters, including diagonals. Low-rank gradient gates use factor coordinates. Timing includes compilation and differs across solver/search configurations; shared host. Do not interpret failures as generic library inferiority. Rank/penalty winners at search boundaries require new confirmation data before further tuning.

## Failure and convergence accounting

Of 256 candidates, 188 passed the convergence gate, 46 finished without passing it, 21 failed with nonfinite objective/gradient, and one timed out. All 24 model selections were available. See [the detailed audit](../failure_audit/README.md). The identical-settings timeout retry is supplementary; original outcomes remain unchanged.

## Identical-settings retry outcome

The timeout candidate finished in 126.7 seconds on retry, using 0.311 GiB peak RSS. It reached the 1500-iteration limit, with gradient infinity norm 9.87083e-05, above the 1e-5 gate. It remains ineligible. Validation log likelihood was -51.508607, compared with -50.513115 for the original selected rank-2, ridge-0.01 model (higher is better). No held-out test evaluation was performed for the retry, and no selected model or original candidate record changed.

Accounting with the supplementary retry: 188 converged, 47 completed without convergence, 21 numerical failures. Historical original accounting remains 188/46/21/1. This retry resolves the missing outcome without changing the comparison conclusions.
