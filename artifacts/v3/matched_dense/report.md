# Matched dense fits

Fresh processes, float64, unregularized, identical initialization and splits. Each row is a synthetic dataset; seeds are data replicates, not timing repeats. The HNN simulation shared this host. NeMoS native uses a different solver.

| Bins | Seed | Backend | Seconds* | Peak GiB | Gradient norm | Accuracy gate |
|---:|---:|---|---:|---:|---:|:---:|
| 4000 | 7 | streamed | 2.762 | 0.282 | 9.7e-08 | True |
| 4000 | 7 | nemos_common | 3.339 | 0.502 | 7e-08 | True |
| 4000 | 7 | nemos_native | 8.378 | 0.635 | 1.4e-09 | True |
| 4000 | 11 | streamed | 2.681 | 0.284 | 7e-08 | True |
| 4000 | 11 | nemos_common | 3.395 | 0.516 | 9.3e-08 | True |
| 4000 | 11 | nemos_native | 9.510 | 0.624 | 2.2e-09 | True |
| 4000 | 19 | streamed | 3.473 | 0.289 | 9.7e-08 | True |
| 4000 | 19 | nemos_common | 3.977 | 0.514 | 6.2e-08 | True |
| 4000 | 19 | nemos_native | 11.332 | 0.636 | 1.4e-09 | True |
| 16000 | 7 | streamed | 8.105 | 0.292 | 7.1e-08 | True |
| 16000 | 7 | nemos_common | 6.264 | 0.587 | 8.9e-08 | True |
| 16000 | 7 | nemos_native | 13.213 | 0.695 | 8.1e-10 | True |
| 16000 | 11 | streamed | 7.053 | 0.294 | 9.4e-08 | True |
| 16000 | 11 | nemos_common | 5.962 | 0.586 | 5.8e-08 | True |
| 16000 | 11 | nemos_native | 12.951 | 0.699 | 2.6e-10 | True |
| 16000 | 19 | streamed | 7.742 | 0.286 | 9.4e-08 | True |
| 16000 | 19 | nemos_common | 5.792 | 0.580 | 7.2e-08 | True |
| 16000 | 19 | nemos_native | 13.058 | 0.685 | 9.9e-10 | True |
| 64000 | 7 | streamed | 25.147 | 0.296 | 5e-08 | True |
| 64000 | 7 | nemos_common | 14.695 | 0.866 | 7.1e-08 | True |
| 64000 | 7 | nemos_native | 26.664 | 0.955 | 5.4e-10 | True |
| 64000 | 11 | streamed | 24.294 | 0.299 | 9.8e-08 | True |
| 64000 | 11 | nemos_common | 14.973 | 0.867 | 9e-08 | True |
| 64000 | 11 | nemos_native | 26.098 | 0.955 | 9.4e-10 | True |
| 64000 | 19 | streamed | 28.752 | 0.299 | 8.2e-08 | True |
| 64000 | 19 | nemos_common | 17.402 | 0.879 | 7.7e-08 | True |
| 64000 | 19 | nemos_native | 31.467 | 0.954 | 1.4e-09 | True |

*Includes feature construction, initial objective evaluation, compilation and optimization. Peak RSS covers the whole worker, including held-out scoring. Common-driver fits use SciPy L-BFGS-B (ftol=1e-14, gtol=1e-7, maxcor=10). Native NeMoS uses LBFGS tol=1e-7. All have a 1,000-iteration budget. The separate accuracy gate requires gradient infinity norm <=1e-5. No low-rank, real-data, large-scale speedup or current upstream batching claim.

Native timing also includes the independent objective-audit compilation; use the common-driver arms for the primary implementation comparison.

## Agreement with StreamGLM

| Bins | Seed | Reference | Objective difference | Held-out LL difference | Max prediction difference | All gates pass |
|---:|---:|---|---:|---:|---:|:---:|
| 4000 | 7 | nemos_common | 3.93e-11 | 3.64e-07 | 6.81e-06 | True |
| 4000 | 7 | nemos_native | 5.2e-11 | 3.55e-07 | 9.96e-06 | True |
| 4000 | 11 | nemos_common | 3.29e-11 | 1.16e-06 | 8.01e-06 | False |
| 4000 | 11 | nemos_native | 4.6e-11 | 1.27e-06 | 7.79e-06 | False |
| 4000 | 19 | nemos_common | 1.01e-11 | 2.81e-07 | 7.87e-06 | True |
| 4000 | 19 | nemos_native | 9.54e-12 | 8.67e-08 | 4.23e-06 | True |
| 16000 | 7 | nemos_common | 2.99e-11 | 2.85e-07 | 4.8e-06 | True |
| 16000 | 7 | nemos_native | 2.31e-11 | 5.34e-08 | 8.14e-06 | True |
| 16000 | 11 | nemos_common | 1.58e-11 | 4.62e-07 | 9.01e-06 | True |
| 16000 | 11 | nemos_native | 4.88e-11 | 7.67e-07 | 1.08e-05 | True |
| 16000 | 19 | nemos_common | 6.7e-12 | 1.31e-07 | 6.95e-06 | True |
| 16000 | 19 | nemos_native | 2.56e-11 | 2.64e-07 | 9.72e-06 | True |
| 64000 | 7 | nemos_common | 1.61e-11 | 8.81e-08 | 9.33e-06 | True |
| 64000 | 7 | nemos_native | 1.03e-11 | 1.53e-07 | 4.35e-06 | True |
| 64000 | 11 | nemos_common | 1.33e-11 | 1.44e-07 | 3.05e-06 | True |
| 64000 | 11 | nemos_native | 2.84e-11 | 1e-07 | 5.12e-06 | True |
| 64000 | 19 | nemos_common | 9.31e-13 | 7.65e-09 | 6.19e-06 | True |
| 64000 | 19 | nemos_native | 1.97e-11 | 1.03e-07 | 5.23e-06 | True |

Agreement requires gradient norms <=1e-5, initial objective/gradient differences <=1e-10, final objective and held-out LL differences <=1e-6, and maximum prediction difference <=1e-3. No failed gate is omitted.
