# Matched dense fits

Fresh processes, float64, unregularized, identical initialization and splits. Each row is a synthetic dataset; seeds are data replicates, not timing repeats. The HNN simulation shared this host. NeMoS native uses a different solver.

| Bins | Seed | Backend | Seconds* | Peak GiB | Gradient norm | Accuracy gate |
|---:|---:|---|---:|---:|---:|:---:|
| 4000 | 7 | streamed | 2.967 | 0.287 | 9.7e-08 | True |
| 4000 | 7 | nemos_common | 3.599 | 0.503 | 7e-08 | True |
| 4000 | 7 | nemos_native | 10.354 | 0.623 | 1.4e-09 | True |

*Includes feature construction, initial objective evaluation, compilation and optimization. Peak RSS covers the whole worker, including held-out scoring. Common-driver fits use SciPy L-BFGS-B (ftol=1e-14, gtol=1e-7, maxcor=10). Native NeMoS uses LBFGS tol=1e-7. All have a 1,000-iteration budget. The separate accuracy gate requires gradient infinity norm <=1e-5. No low-rank, real-data, large-scale speedup or current upstream batching claim.
