# NeuroTwinBench (inherited project documentation)

**A ground-truth study of when functional connectivity stops reflecting biological connectivity.**

If you only observe the spikes a circuit produced, how much of the circuit can you recover — and under what conditions does that recovery stop being trustworthy?

Experimental neuroscience cannot answer this, because the true synaptic wiring of a recorded network is unknown. NeuroTwinBench uses biophysically detailed simulation ([HNN-Core](https://github.com/jonescompneurolab/hnn-core)) as a source of exact ground truth, hides the wiring, fits statistical models ([NeMoS](https://github.com/flatironinstitute/nemos) population GLMs on JAX) to the spikes alone, and scores the result against what actually generated them.

This is a **study with reusable code**, not a framework. The deliverable is a falsifiable result with calibration, baselines, a null and error bars. [PROJECT.md](PROJECT.md) is the governing specification.

---

## Install

```bash
conda create -n neurotwin python=3.11 && conda activate neurotwin
pip install -e .
pip install -e '.[hnn]' psutil mpi4py     # hnn-core; needs NEURON
export PATH="$CONDA_PREFIX/bin:$PATH"     # MPIBackend runs a bare `nrniv`
```

## Run

```bash
python tests/test_pipeline.py                     # full check, no NEURON, CI-safe
python scripts/ground_truth_audit.py --mesh 5 5   # identifiability, no simulation
python experiments/calibrate.py --duration 300    # validate the estimator
python experiments/hnn_first.py --n-procs 4       # easiest HNN condition
python experiments/grid.py --seeds 2 3 4          # the two-axis experiment
python figures/make_figures.py                    # rebuild figures from results/
```

---

## The method

```
HNN-Core circuit  ──simulate──>  spikes  ──hide the wiring──>  population GLM
       │                                                            │
       └────────────── exact per-pair ground truth ─────────────────┴──> compare
```

**Built inside-out.** The estimator is validated on data from a known coupled GLM — same basis, same link, so the model is exactly specified — *before* it is pointed at biophysics. Without that control, poor recovery on HNN data is uninterpretable: broken estimator, wrong basis, wrong binning and genuine model mismatch all look identical.

```
population GLM   rho +0.797   sign 100%
pairwise GLM     rho +0.755   sign  96%
CCG baseline     rho +0.519   sign  94%
jitter null      rho +0.116   sign  84%   (chance)
```

---

## What the ground truth says before any simulation

`scripts/ground_truth_audit.py` is pure arithmetic over `net.connectivity` — no NEURON, runs in seconds. It answers *which metrics are identifiable at all* before compute is spent. Five findings shape the whole study.

**1. Weight and delay are perfectly rank-confounded.** From `cell._get_gaussian_connection`, `weight = A_weight·gain·g(d)` and `delay = A_delay/g(d)`, so within any class `weight × delay` is constant and

> ρ(true weight, true delay) = **−1.000000**, in all 16 classes at both 5×5 and 10×10.

Spearman is the primary metric, so the default model cannot separate them. Independent delay recovery needs purpose-built connections.

**2. `lamtha` is not a valid grouping key.** Several classes share a λ with different `A_weight`, and the invariant holds only at fixed `A_weight`:

| grouping | CV | ρ(w, d) |
|---|---|---|
| each λ=50 class individually | 0.0027 | −1.000000 |
| **pooled by λ=50** | **0.6861** | **−0.553918** |

The pooled row reads as *"inhibitory weights have usable range and delay is only partly confounded"* — the inverse of both real findings. Group by `(src_type, target_type, receptor, loc)`.

**3. Dynamic range splits along the E/I axis.** `inplane_distance` cancels out, so `mesh_shape` is the only lever.

| λ | classes | weight CV @5×5 | @10×10 |
|---|---|---|---|
| 3 | all pyramidal-source | 0.55 | 1.36 |
| 20 | basket→basket | 0.017 | 0.069 |
| 50 / 70 | basket→pyramidal | 0.003 / 0.001 | 0.011 / 0.006 |

Ranking inhibitory weights ranks numerical noise — a property of the model, not a failure of the estimator.

**4. E and I weight ranges are disjoint, so pooled rank metrics are invalid.** Excitatory weights run 7.1e-6 to 5.0e-4, inhibitory 9.9e-4 to 5.0e-2. **All 1440 largest-magnitude pairs out of 5840 are inhibitory.** A pooled ρ scores E-vs-I and nothing else, while looking excellent.

**5. The large mesh changes what is observable.** At 5×5, 99% of excitatory connections have effective delay inside a 25 ms window; at 10×10, only ~53%. Nearly half the true edges become invisible regardless of strength — so **5×5 is the development environment**, not merely the small condition.

That last one drives the **evaluation-set rule**: a connection whose delay exceeds the history window cannot be represented by the model, so scoring it as a miss measures the window, not the estimator. Those are reported separately, never silently dropped. The same rule covers neurons that never fired.

---

## What the simulations say

**Drive response is non-monotonic.** Quadrupling the drive *silences* the pyramidal populations — they fall to 0.3 and 0.0 Hz while basket cells fire at 9–10 Hz. The drive recruits interneurons whose weights are ~100× larger, and feedforward inhibition shuts the pyramidal cells down. The static weight disparity of finding 4 predicts a dynamic regime change.

| `rate_constant` | mean Hz | silent | L2bask / L2pyr / L5bask / L5pyr |
|---|---|---|---|
| 10 | 5.7 | 3/70 | 5.0 / 6.3 / 3.7 / 6.1 |
| 40 | 2.9 | 48/70 | 9.0 / **0.3** / 10.3 / **0.0** |
| 120 | 21.2 | 1/70 | 27.3 / 28.8 / 31.0 / 7.2 |

**MPI: 9.5 s per simulated second** at 4 processes, with byte-identical spike counts. 8 procs is *slower* than 4 — with 70 cells there is too little work per rank. And 4.26× on two processes is superlinear, so this is not parallel scaling: `MPIBackend` is simply a faster code path than `JoblibBackend`.

**The grid costs one simulation per seed.** Both experimental axes are post-hoc on the same spikes — drive observability changes which covariates reach the estimator, observed population is a column mask.

---

## Things this project had to retract

Kept visible because the reasoning was sound in each case and the next person will make the same move.

**Location is not identifiable in the default model.** L2pyr→L5pyr AMPA is added twice, once proximal and once distal, with identical weights — an apparently perfect controlled contrast, and originally billed here as the strongest result in the project. But all 625 pairs carry *both* synapses, and a GLM estimates one filter per ordered pair. Nothing recovers what was never separable. A purpose-built network (`network="location"`) now makes it identifiable by splitting the source population, one half per section.

**Solver tolerance is not a confound.** Suspected that early stopping was acting as the real regularizer. Tested it: ρ is invariant to `tol` across two penalties. The spread that prompted the suspicion was sampling noise, which had been flagged as such when measured and then over-corrected on.

**There is no plateau to find on synthetic data.** §5 originally said to pick the duration at which recovery stabilises. On synthetic data the generating model *is* the fitted model, so ρ → 1.0 and no plateau below it can exist. The plateau belongs to the HNN curve, where model mismatch imposes a real ceiling — and the gap between the two curves at matched duration is the actual measurement.

**The pairwise baseline ties the population GLM** (+0.755 vs +0.763). Against the CCG alone, +0.763 vs +0.519 reads as a clear win for joint estimation. Against the pairwise model it is not a win at all yet.

---

## Layout

```
neurotwinbench/
├── ground_truth.py   HNN connectivity -> effective per-pair weight/delay/sign/loc
├── simulate.py       NEURON + drive + cache; default and location networks
├── synthetic.py      known-coupling GLM generator (the calibration fixture)
├── fit.py            population GLM, pairwise baseline, regularizer selection
└── metrics.py        recovery, jitter null, CCG, edge detection, evaluation set
experiments/  calibrate · latency_calibration · hnn_first · grid · location
scripts/      ground_truth_audit.py
figures/      regenerate from results/ alone, no NEURON
upstream/     proposed hnn-core contributions
tests/        runs in CI, no NEURON required
```

Flat until the science requires otherwise. No plugin layer, no adapters, no dashboard.

## License

MIT
