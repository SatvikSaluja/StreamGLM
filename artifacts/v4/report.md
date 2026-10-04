# StreamGLM 0.3 measured validation

Real Neuropixels predictive tests, an explicitly synthetic scale test, and interoperability with pinned current NeMoS. These establish software capabilities under the recorded conditions; they do not establish anatomical recovery.

## Allen Visual Coding Neuropixels

300 seconds at 5 ms bins; provider quality==good and VISp; ascending unit IDs. Chronological 60/20/20 split, training-only activity filtering, ranks 0/2/5 and two starts for nonzero ranks. Selection uses validation data.

| Retained / requested | Rank | Gain over self-history (nats/bin) | Gain (bits/spike) | Peak GiB | Workflow seconds |
|---:|---:|---:|---:|---:|---:|
| 98/100 | 5 | 0.0879799 | 0.0456145 | 0.433 | 214.5 |
| 185/195 | 5 | 0.196361 | 0.0766088 | 0.492 | 754.0 |

## Synthetic scale test

1000 independent Poisson units, 3600 seconds, 10 ms bins. Peak process RSS 1.241 GiB; workflow 1623.0 seconds. Hypothetical full-recording design: 8.05 GiB. The design size is arithmetic, not an OOM measurement.

Rank-5 test gain over self-history: -0.000381805 nats/bin. No coupling exists in this generator; do not treat this as recovery validation.

## Current NeMoS interoperability

Pinned commit `81c7200a0e66686e98fd1907e7fa11e111a0e66a`. `NeMoSRecordingLoader` retains epoch boundaries and short tails; fixed features are tested against independent whole-epoch NeMoS convolutions. Current upstream already supports this kind of streaming.

SVRG reached gradient norm 9.97e-06; objective difference from unrestricted StreamGLM L-BFGS: 9.25e-07. The initial 300-pass run did not meet the gradient threshold; it is retained. The longer run used the same settings and stopped at the threshold.

## Limitations

- The two real-data subsets overlap within one VISp recording, not independent biological replications.
- Quality labels come from the supplied workshop extract; no waveform-level quality reanalysis.
- Real-data coupling has no stimulus/behavior adjustment and does not establish synaptic or causal connectivity.
- The 1,000-unit hour-long case is independent synthetic Poisson at 10 ms bins, not real Neuropixels at 1 ms.
- Optimizer relative-objective stopping does not certify a global optimum or a small gradient; diagnostics retained.
- The current-NeMoS exercise validates loader interoperability and accuracy, not a performance advantage over upstream streaming.
- CPU timing was measured on a shared host while HNN and other validation jobs ran.
- No remote CI, upstream acceptance, package publication, or 1,000-unit hour-long real-data fit is claimed.
