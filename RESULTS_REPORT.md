# StreamGLM results

Snapshot: 2026-10-05. This is a fitting and prediction project; it is distinct from NeuroTwinBench's biological connectivity benchmark.

## Tuned comparison (means across seeds 31 and 47)

| Generating network | Neurons | Low-rank StreamGLM test bits/spike | NeMoS test bits/spike | Low-rank filter correlation | NeMoS correlation |
|---|---:|---:|---:|---:|---:|
| Low rank | 64 | 0.006618 | 0.000906 | 0.764 | 0.294 |
| Low rank | 128 | 0.002470 | 0.000005 | 0.593 | 0.151 |
| Full rank | 64 | 0.022109 | 0.031283 | 0.646 | 0.789 |
| Full rank | 128 | 0.004911 | 0.014151 | 0.377 | 0.646 |

Low rank helps when its structural assumption matches the generator; it hurts on the full-rank fixtures. StreamGLM's dense model closely matches the full NeMoS model's predictive/recovery outcomes. These are different model classes, not proof of implementation superiority. Correlations are Pearson correlations including self-history; bits/spike are test likelihood gains over a training constant-rate baseline. Settings were selected on validation data among gradient-gate-passing candidates. Two seeds remain exploratory.

Of 256 original candidates: 188 passed, 46 finished without passing, 21 had numerical failures and one timed out. The timeout retry completed but did not pass. All 24 selected outcomes were available. See [tuning evidence](results_page/sources/streamglm_tuning_report.md).

## Runtime, memory and real data

64-neuron, 96,000-bin length pilot: low-rank-generated data repeat fit 4.802 s / 0.335 GiB (StreamGLM) versus 5.034 s / 1.018 GiB (NeMoS). Full-rank-generated data: 21.315 s / 0.332 GiB versus 16.682 s / 1.020 GiB. NeMoS is faster in the latter case. These are fixed solver/configuration pilots with different caching/compilation boundaries, not universal speed claims or streaming-versus-streaming results.

Allen VISp cohorts, same session and 300 seconds: 98 neurons gained 0.0456 test bits/spike over self-history; 185 neurons gained 0.0766. This is predictive evidence without a synaptic answer key. The two cohorts are not independent-session replication. Optimizer-success flags in that older report are weaker than the later strict 1e-5 gradient gate.

The independent-Poisson 1,000-neuron workload demonstrates feasibility, not coupled recovery. Coupled seed 941 has six completed candidate records and a candidate-6 checkpoint, but no final report. Seed 942 has generated data, with fits pending. Candidate 2's 0.001304 bits/spike is VALIDATION gain, not final test performance. Final selection, off-diagonal truth recovery and null evaluation remain pending.

## Evidence and reproduction

The [HTML page](results_page/index.html) bundles compact source reports in [results_page/sources](results_page/sources). See [SIMULATIONS.md](SIMULATIONS.md) for commands, resume behavior and new confirmation recipes. Large count arrays, model checkpoints and public-data downloads remain local. No universal NeMoS superiority, arbitrary 1,000-neuron recovery, or real anatomical connectivity claim is supported.
