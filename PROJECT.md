# NeuroTwinBench

## A ground-truth study of when functional connectivity stops reflecting biological connectivity

> **Status:** research specification v2.2
> **Primary audience:** computational neuroscientists and research software engineers working on neural data analysis, system identification, and scientific software.

---

## 0. What this is

A controlled computational-neuroscience study comparing the known synaptic organization of a biophysically detailed neural circuit with the functional interactions inferred from its spiking activity alone.

> **Under what observational and biophysical conditions does inferred functional connectivity cease to be a faithful representation of the underlying neural circuit, and what fails first?**

The deliverable is not a framework. It is a falsifiable result supported by calibration, controls, baselines, a meaningful null, repeated simulations, uncertainty, reproducible figures, and tested research software. Reusable infrastructure is extracted only when the experiment requires it.

```text
TRUE BIOPHYSICAL CIRCUIT ──generates──> spike trains ──only these observed──>
    statistical inference ──> FUNCTIONAL CONNECTIVITY ──> compare against truth
```

---

## 1. Why HNN-Core

HNN-Core provides what experimental recordings cannot: exact ground truth — source, target, receptor, nominal weight, effective distance-scaled weight, physical delay, cell types, target location, and (with section targeting) section identity.

The project uses `neymotin_2020_model(...)`, not the deprecated `jones_2009_model(...)` (renamed in hnn-core 0.6.0; old name retained only as a wrapper).

It directly extends previous work on HNN-Core connectivity and synapses rather than treating the simulator as a black box.

---

## 2. Constraints that shape the project

### A — spike information is expensive

Short HNN simulations may produce too few spikes to estimate a high-dimensional coupling model. No universal threshold is assumed; the data budget is measured (§5).

### B — the default network is dense

`neymotin_2020_model` uses all-to-all connectivity within its connection classes. Binary edge/no-edge questions are degenerate. For the default network the primary questions are continuous: anatomical strength ↔ statistical influence, physical delay ↔ effective interaction latency, E/I identity ↔ inferred sign. Binary topology metrics are reserved for sparse custom networks built with `probability < 1`.

### C — common external drive can mimic coupling

Two neurons can covary without influencing one another. A model that cannot see the drive attributes shared input to shared coupling. This is a central scientific variable, not a nuisance (§12).

### D — location ground truth uses HNN's vocabulary

`loc` is an element of `Cell.sect_loc` (`proximal`, `distal`) or a section (`soma`, `apical_tuft`, requires `legacy_mode=False`). The default model uses exactly `proximal`, `distal`, `soma`. Section-level analysis is a separate experiment.

### E — weight and delay are rank-confounded by construction

From `cell.py:_get_gaussian_connection`:

```
scaled_lamtha = lamtha * inplane_distance
g(d)          = exp(-(d_xy / scaled_lamtha)^2)      # xy plane only
weight(d)     = A_weight * gain * g(d)
delay(d)      = A_delay / g(d)
```

Therefore within a connection class `weight * delay = A_weight * gain * A_delay`, a constant, so

> **Spearman ρ(true weight, true delay) = −1 exactly, in every connection class.**

Verified empirically in all 16 classes at both 5×5 and 10×10. Since Spearman is the primary metric, weight and delay cannot be separated in the default model. Independent delay recovery requires custom connections with varied `delay=`.

Note `inplane_distance` cancels: positions scale by `s` and `scaled_lamtha` scales by `s`, so `d/(λ·s)` is invariant. **`mesh_shape` is the only lever on dynamic range.**

### F — dynamic range is partitioned along the E/I axis

`A_delay = 1.0` for every connection in the default model, and λ is class-determined:

| λ | classes | weight CV @5×5 | @10×10 |
|---|---|---|---|
| 3 | **all pyramidal-source** (incl. pyr→basket) | 0.55 | 1.36 |
| 20 | basket→basket | 0.017 | 0.069 |
| 50 | L2bask→pyr | 0.0027 | 0.011 |
| 70 | L5bask→L5pyr | 0.0014 | 0.0057 |

Every excitatory class supports weight-rank analysis. Basket→pyramidal classes do not, at either mesh size — a ρ over them ranks numerical noise. This is a property of the model, not a failure of the estimator, and must be reported as such. Sign recovery remains meaningful for inhibitory classes.

### G — E and I weight ranges are disjoint, so pooled rank metrics are invalid

Nominal `A_weight` spans 200× across classes (2.5e-4 for L2pyr→L5pyr up to 5e-2 for L2bask→L2pyr). After distance scaling at 5×5:

```
excitatory |weight|:  7.141e-06 .. 5.000e-04
inhibitory |weight|:  9.873e-04 .. 5.000e-02      <- no overlap, 1.97x gap
```

**All 1440 largest-magnitude pairs out of 5840 are inhibitory.** A Spearman ρ computed over all connections pooled therefore ranks E against I and nothing else — it would report a near-perfect score while carrying no information about within-class strength recovery.

> Every weight-rank metric is computed **within connection class**. Pooled ρ over mixed E/I connections is not reported, because it is not measuring what its name says.

### H — connection class is the only valid grouping key

`lamtha` looks like a natural grouping and is not one. Several classes share a λ with different `A_weight`, and the weight×delay invariant of §E holds only at fixed `A_weight`:

| grouping | CV | ρ(weight, delay) |
|---|---|---|
| L2bask→L2pyr gabaa (`A_weight` 5e-2) | 0.0027 | −1.000000 |
| L2bask→L2pyr gabab (5e-2) | 0.0027 | −1.000000 |
| L2bask→L5pyr gabaa (1e-3) | 0.0027 | −1.000000 |
| **pooled by λ=50** | **0.6861** | **−0.553918** |

The pooled row is an artifact of mixing two weight scales, not a property of any class. Group by `(src_type, target_type, receptor, loc)` — `GroundTruth.by_class()`. This is the same trap as §G one level down, and it is easy to hit because λ is the variable that *causes* the within-class spread.

`tests/test_pipeline.py::test_lamtha_is_not_a_valid_grouping_key` asserts the artifact is still reproducible, so that if hnn-core ever makes λ a valid key, every per-class claim in §F gets re-checked.

---

## 3. Operating-point pilot

The first artifact is `scripts/ground_truth_audit.py` — pure ground-truth arithmetic, no NEURON, runs in seconds. It reports per connection class: n_pairs, λ, loc, receptor, effective weight min/max/CV, unique values, effective delay min/max/span, **fraction of pairs inside the coupling window**, and ρ(weight, delay).

Run it before committing to any headline metric.

### Fixed starting configuration

```python
net = neymotin_2020_model(mesh_shape=(5, 5), legacy_mode=False)   # 70 cells
```

`mesh_shape=(5,5)` gives 25+25 pyramidal and 10+10 basket = **70 cells**. The 10×10 default gives 270. Drive must sustain spiking; a single evoked transient is not the calibration condition.

**5×5 is the development environment, not just the small condition.** At 5×5, 99% of excitatory connections have effective delay inside a 25 ms window; at 10×10 only ~53% do. The large mesh changes what the estimator can observe at all, so it is a stress condition run later.

### Simulation backend

Reference timings use `MPIBackend`. A benchmark record must include backend, MPI process count, CPU model, hnn-core version, NEURON version, `mesh_shape`, `tstop`, and drive configuration hash. Wall-clock without that context is not a reproducible measurement.

### Pilot outputs

spikes/neuron/second · total spikes/neuron · active-neuron fraction · wall-clock per simulated second · predictor count · samples/predictor · feature-matrix memory · effective-latency resolution · held-out drive contribution.

### Measured operating point (5×5, hnn-core 0.6.1 / NEURON 8.2.7, 8-core WSL2)

**Simulation cost — the binding constraint.** Measured at `tstop=2000 ms`, 5×5, 8-core WSL2:

| backend | wall | s per simulated second | vs serial | spikes |
|---|---|---|---|---|
| JoblibBackend (serial) | 93.8 s | 46.9 | 1.00× | 460 |
| MPIBackend, 2 procs | 22.0 s | 11.0 | 4.26× | 460 |
| **MPIBackend, 4 procs** | **19.0 s** | **9.5** | **4.94×** | 460 |
| MPIBackend, 8 procs | 28.0 s | 14.0 | 3.35× | 460 |

Spike counts are identical across every configuration, so MPI changes throughput and not the result.

Two caveats on reading this table. **4.26× on two processes is superlinear**, so it is not parallel scaling — `MPIBackend` is a faster code path than `JoblibBackend` irrespective of process count, and this comparison conflates backend choice with parallelism. And 8 procs is *slower* than 4: with only 70 cells there is too little work per rank to cover communication. **4 processes is the operating point**; do not assume more is better.

MPI requires `nrniv` on PATH, plus `psutil` and `mpi4py` which are not hnn-core dependencies. Without them it aborts with SIGABRT, and without psutil the real error is masked by a `ModuleNotFoundError` raised inside hnn-core's own cleanup handler. `simulate._require_nrniv_on_path()` preflights all three — a candidate upstream contribution, since the check belongs in `MPIBackend`.

**The grid needs one simulation per seed, not one per cell.** This is the largest single correction to the cost model:

* Axis 1, drive observability (true / proxy / hidden), changes only what covariates are handed to the estimator. Same spikes.
* Axis 2, observed population (full / interneurons hidden / subsample), is a column mask. Same spikes.

Both axes are post-hoc. Only `event_seed` requires a new simulation. So the primary grid of §21 costs **n_seeds simulations**, not n_seeds × 9. At 9.5 s per simulated second and ≥300 s of recording, that is ~50 min per seed, and the full primary grid is hours rather than days.

**Drive response is non-monotonic — more drive gives fewer spikes.**

| `rate_constant` | `w_ampa` | mean Hz | silent | L2bask / L2pyr / L5bask / L5pyr |
|---|---|---|---|---|
| 10 | 2e-4 | 5.7 | 3/70 | 5.0 / 6.3 / 3.7 / 6.1 |
| 40 | 4e-4 | 2.9 | 48/70 | 9.0 / **0.3** / 10.3 / **0.0** |
| 80 | 8e-4 | 10.1 | 23/70 | 20.0 / 12.3 / 19.0 / 0.5 |
| 120 | 1.5e-3 | 21.2 | 1/70 | 27.3 / 28.8 / 31.0 / 7.2 |

Quadrupling the drive *silences the pyramidal population*. The drive recruits
basket cells, and because inhibitory weights are ~100× larger than excitatory
(§G), feedforward inhibition shuts the pyramidal cells down. The static
ground-truth disparity of §G thus predicts a dynamic regime change, and drive
strength cannot be tuned by assuming monotonicity.

`rate_constant=10, w_ampa=2e-4` is the working point: 5.7 Hz mean, 3/70 silent,
all four populations active.

**Silent-neuron counts are duration-dependent.** The identical configuration
reads 28/70 silent at `tstop=200 ms` and 3/70 at 300 ms — most of that "silence"
is a neuron not having fired yet, not a dead cell. Any active-fraction figure
must state its observation window.

---

## 4. Measuring the contribution of external drive

Compare a drive-only model against intercept-only on held-out data; report held-out deviance explained by drive regressors, per neuron and population. The common-drive premise becomes a measured property of the regime, not an assumption.

---

## 5. Data-budget calibration

Increase duration 1 → 2 → 5 → 10 → 20 min, recording spikes/neuron, predictor count, samples/predictor, recovery metric, uncertainty. Choose the shortest duration at which recovery and uncertainty stabilize, subject to simulation cost.

### Measured on synthetic data (20 neurons, 99 connections, 1 seed)

| duration | spikes | GLM ρ | sign | latency err | CCG | jitter null |
|---|---|---|---|---|---|---|
| 60 s | 11,015 | +0.597 | 93% | 2.27 ms | +0.230 | +0.112 |
| 120 s | 21,807 | +0.690 | 99% | 2.05 ms | +0.425 | +0.071 |
| 300 s | 54,087 | +0.823 | 100% | 1.75 ms | +0.519 | +0.125 |

Extended to 600 s the curve keeps climbing: **+0.699 → +0.797 → +0.896**, roughly linear in log duration.

### There is no plateau to find on synthetic data, and looking for one was a mistake

This section originally said: *choose the shortest duration at which recovery stabilises.* That is the wrong criterion for the calibration fixture, and the sweep is what exposed it.

On synthetic data the generating model **is** the fitted model — same basis, same link, same coupling structure (§8). A correctly specified estimator given unlimited data recovers the truth, so ρ → 1.0 and **no plateau below 1.0 exists**. The flat region the original wording was looking for cannot occur by construction. Continuing to extend the sweep would have burned compute chasing an artifact of the specification.

The corrected criterion, in two parts:

1. **Synthetic duration is chosen by cost, not by stabilisation.** Pick a duration, record the ρ it buys, and hold it fixed across everything compared. The number is a reference point, not a discovered optimum.
2. **The plateau belongs to the HNN curve, not this one.** There, model mismatch imposes a genuine ceiling: biophysics is not a GLM, so more data cannot buy recovery past what the model can represent. **Where the HNN curve flattens, and how far below the synthetic curve it does so, is the project's central measurement** (§29/§30) — and it only means something because the synthetic curve above it keeps rising.

So the two curves must be reported together — but **not at matched duration**, which was the wrong pairing.

### Match on spikes per predictor, not duration or bins

Bins per predictor is the wrong power diagnostic. A Poisson GLM's information scales with events, so empty bins contribute almost nothing, and the bins ratio is dangerously reassuring:

| case | predictors | bins/pred | **spikes/pred** | ρ |
|---|---|---|---|---|
| synthetic 20n × 300 s @ 9 Hz | 168 | 1786 | **16.1** | +0.80 |
| synthetic 20n × 600 s @ 9 Hz | 168 | 3571 | **32.1** | +0.89 |
| HNN 70n × 300 s @ 3.8 Hz | 568 | 528 | **2.0** | — |

The same 300 s buys 8× less information on HNN than on synthetic data, because HNN fires at 3.8 Hz against 9 Hz and carries 3.4× more predictors. A 50:1 bins threshold passes that condition comfortably.

Two consequences:

1. **Comparing the curves at matched duration would conflate model mismatch with data budget** — exactly the confound the synthetic fixture exists to eliminate. They must be compared at matched spikes per predictor.
2. To reach the synthetic condition that gave ρ +0.80, HNN needs roughly **2500 simulated seconds at 70 neurons** — about 6.7 h at 9.5 s per simulated second. That is the real cost of one interpretable HNN result, and it is the number the experiment grid must be budgeted against.

`fit.power_check` reports all three ratios so a run cannot be started on the strength of the reassuring one.

Two secondary observations:

* The jitter null sits at ~0.1 at every duration. That is the chance floor for ranking 99 pairs; it does not shrink with more data, so it is a property of the sample size rather than of the estimator.
* The CCG baseline improves with duration too (+0.230 → +0.519). The GLM's margin over it is roughly constant *in ratio* rather than widening, which is worth knowing before claiming the GLM's advantage grows with data.

---

## 6. Temporal-resolution calibration

Generate synthetic interactions at known lags (2, 4, 6, 8, 10 ms …), push them through the identical binning, history window, basis, fit, and latency estimator, and report the smallest resolvable between-condition difference. If a 1 ms shift is not resolvable, later results do not claim 1 ms resolution.

---

## 7. Ground-truth extraction

`ground_truth.py` resolves `net.connectivity` into per-pair effective weight, physical delay, sign, and `loc`, using the formula in §E. Retains `src_gid, target_gid, src_type, target_type, receptor, loc, weight, delay, sign, lamtha`. Validated against a network where values are computed by hand. Small and production-grade — the audit was reconnaissance, not a library.

Candidate upstream HNN-Core contribution if a clean public API seam exists.

---

## 8. Synthetic calibration fixture

Before HNN touches the inference system, the identical pipeline is tested on data from a model whose coupling parameters are known.

```text
KNOWN filters -> synthetic GLM -> spikes -> SAME pipeline -> recovered filters
```

Generating filters live on the *same* nemos basis used for fitting, so the model is exactly well specified and any error is estimation error, not basis mismatch.

This separates estimator failure from biophysical model mismatch. Without it, poor recovery on HNN data is uninterpretable.

**Pre-registered criteria.** Thresholds for weight-rank recovery, sign recovery, effective-latency error, and filter error are established from synthetic calibration and **frozen before the first HNN result is examined** (§27 step 4 → 5 is the deadline).

---

## 9. Graded difficulty ladder

```text
EASIEST  synthetic coupled GLM, known filters
    ↓    sparse HNN | all neurons | true drives | long recording | upper physiological coupling
    ↓    sparse HNN | nominal physiological coupling
    ↓    dense HNN  | all neurons | true drives
    ↓    dense HNN  | drive proxy only
    ↓    dense HNN  | drives hidden
HARDEST  dense HNN  | interneurons hidden | drives hidden
```

### Physiological gate on the easy rung

"Strong coupling" does not mean multiplying weights until recovery appears. Any manipulation must stay inside a defensible physiological range, and the project records baseline parameter, modified parameter, multiplicative change, allowed range, and justification. The condition is valid only if the network still shows plausible firing rates, no runaway excitation or pathological synchrony, and a dipole in the expected range. If recoverability requires nonphysiological weights, that is itself a result and the condition is labelled a stress test.

---

## 10. Fail early

If the easiest valid HNN condition gives ρ ≈ 0.1, sign ≈ null, no stable latency relationship, that triggers a decision immediately: longer recording, lower density, stratify receptor/location, inspect cell-type pairs, quantify mismatch directly. Coupling increases only within the physiological gate.

If recovery stays poor after validated controls, the project pivots to the negative result (§30). It does not depend on obtaining a high correlation.

---

## 11. Inference layer

Pynapple for spike representation, NeMoS `PopulationGLM` on JAX for fitting. Only verified APIs are assumed: `SVRG`, `ProxSVRG`, and `GLM.update(..., n_samples=...)` for manual minibatch loops. No `DataLoader`, `LazyArrayDataLoader`, or `stochastic_fit` — those do not exist. Existing solver/update mechanisms first; only a demonstrated gap becomes new infrastructure or an upstream contribution.

`jax_enable_x64` is required: the Hessian of a coupled Poisson GLM is badly conditioned and float32 LBFGS stalls before converging.

### Solver tolerance: suspected confound, tested, not supported

`tol=1e-6/maxiter=500` is 4× faster than `1e-8/2000` at 14 neurons × 60 s (39.6 s vs 158.2 s) with no detectable change in recovery, and is the default.

That sweep showed ρ varying 0.497–0.575 across tolerances. With `regularizer_strength` at a token `1e-6` the model is near-unregularized, so early stopping plausibly *was* the real regularizer — which would make `tol` a scientific parameter. Tested directly at 20 neurons × 120 s, varying only `tol`:

| `tol` | ρ @ 1e-6 | ρ @ 1e-3 |
|---|---|---|
| 1e-06 | +0.690 | +0.699 |
| 1e-04 | +0.690 | +0.699 |
| 1e-03 | +0.690 | +0.701 |

**Recovery is invariant to `tol` at both penalties.** The suspicion is not supported; the original spread was sampling noise at 14 neurons, as flagged when it was measured. Pin `tol` across compared conditions for reproducibility, but it is a performance knob.

### The regularizer is now selected, not assumed

`select_regularizer` maximises held-out log-likelihood over a **contiguous** time split — never a random one, because spike trains are autocorrelated and the design matrix is built from lagged history, so interleaved test bins share history with training bins and the score is optimistically biased.

On synthetic data (20 neurons × 120 s) the curve is cleanly unimodal, peaking at `1e-3` — a thousand times the token value previously used:

```
1e-06  -0.051950      1e-03  -0.051478  <- best
1e-05  -0.051886      1e-02  -0.051578
1e-04  -0.051605      1e-01  -0.051610
```

The accuracy gain is small (ρ +0.690 → +0.699). The point is that regularization is now explicit and reportable rather than implicit. **Re-select per dataset** — the optimum is a property of the data, and nothing licenses carrying a synthetic-data choice over to HNN.

---

## 12. Drive observability experiment

| Condition | Information given | Interpretation |
|---|---|---|
| A — true drive | actual external-drive events | upper bound |
| B — external proxy | proxy from a signal **external to the modelled population** | realistic |
| C — hidden | no drive covariate | maximum confounding |

> The proxy may represent observable task/input timing or a separately observed input channel, but **may not** be a population PSTH computed from the target neurons themselves.

Otherwise the proxy absorbs genuine recurrent coupling and the middle condition is circular. The gradient R1 → R2 → R3 is interpretable only under that constraint. Proxy generation is fixed before the main comparison.

---

## 13. Weight recovery is not coefficient equality

A biological connection passes through synaptic strength × receptor kinetics × target location × membrane dynamics × network state. The question is whether effective synaptic strength is systematically *related* to inferred statistical influence. Primary measurement: Spearman ρ(|true effective weight|, |inferred filter area|), interpreted as **recoverability / association**, never numerical equality. Sign is scored separately so a sign flip cannot inflate ρ.

---

## 14. Location as a scientific variable

**The "clean contrast" does not exist. Location is not identifiable from spikes in the default model.**

`L2_pyramidal → L5_pyramidal` AMPA is added twice — once `proximal`, once `distal` — with the same weight parameter, which looked like a perfectly controlled location experiment: same source population, target population, receptor, λ and nominal weight, differing only in dendritic location. It was listed here as the primary location result.

Checking the realised pairs kills it:

```
proximal pairs: 625        distal pairs: 625
pairs with BOTH: 625       only proximal: 0       only distal: 0
```

Both `add_connection` calls use the same `src_gids` and `target_gids`, so **every** L2pyr→L5pyr pair carries a proximal *and* a distal synapse. A GLM estimates one coupling filter per ordered pair, so what it recovers is the combined effect of the two. There is no pair anywhere in the default model whose coupling is attributable to a single location.

Location analysis therefore requires a **purpose-built network** — `legacy_mode=False` with explicit section targeting, constructed so that a given pair receives a synapse at one location only.

### The location network (`network="location"`)

Built, and it makes location identifiable by construction. The L2 pyramidal population is split in half by gid; one half projects onto `basal_2`, the other onto `apical_tuft` of the same L5 pyramidal targets, with identical receptor, weight and λ. Feedforward inhibition is retained so the network stays in a firing regime.

```
L2pyr[half A] --AMPA--> L5pyr  basal_2       (proximal group)
L2pyr[half B] --AMPA--> L5pyr  apical_tuft   (distal group)

625 pairs, max locations per pair: 1          <- IDENTIFIABLE
|w| basal_2     1.428e-05 .. 5.000e-04
|w| apical_tuft 1.428e-05 .. 5.000e-04        <- ranges match exactly
```

Each pair has exactly one location and the two groups differ in nothing else, so a difference between them is attributable to dendritic placement rather than connection class.

The sharper question is **latency**, not weight. Distal input traverses more dendrite before reaching the soma, so it should arrive later and more smeared than its physical delay implies. That gap — between physical delay and effective interaction latency, as a function of dendritic distance — is precisely the biophysical transformation §15 names as the object of study, and this network isolates it.

**The confounded split.** Across the whole model, `loc` is heavily confounded with connection class: `proximal` ≈ pyr→pyr recurrent, `soma` ≈ most basket-involving connections, `distal` ≈ L2→L5 cross-layer plus L2bask→L5pyr. A three-way ρ_prox / ρ_dist / ρ_soma comparison is a **connection-class stratification** and must be reported as one — not as a dendritic-location effect.

Section-level analysis requires a separate `legacy_mode=False` network with explicit section targeting, and must not be presented as if those labels existed in the default ground truth.

---

## 15. Delay becomes effective interaction latency

The GLM filter peak is not an estimate of `NetCon.delay`. The observable effect contains transmission delay + receptor kinetics + dendritic filtering + membrane integration. The inferred quantity is named **effective interaction latency**, and the bias between it and physical delay is the object of study, stratified by receptor, location, and cell types.

Per §E, the default model cannot support *independent* delay recovery — delay is a deterministic reciprocal function of the same Gaussian factor as weight. What is well posed there: does the distance-induced latency spread within a connection class correspond to inferred effective latency? Independent delay variation requires custom connections.

---

## 16. Null model

Primary null is **controlled temporal jitter** — each spike displaced uniformly within a fixed window, conserving spike count. Full permutation is not used: it destroys rate structure too and drives the floor to zero, inflating everything above it.

Jitter width must exceed the coupling window (default 2×) or the null retains the coupling it is meant to destroy. The final width is chosen relative to the coupling window and frozen before the final experiment. Trial-wise circular shifts only if they answer a distinct question.

---

## 17. Baselines

Main estimator: NeMoS population GLM. Baseline 1: cross-correlogram. Baseline 2: pairwise GLM. Null: temporally jittered spikes.

**No primary metric is reported without both a baseline and a null.**

### The pairwise baseline is closer than expected

On synthetic data (10 neurons, 60 s, 23 connections):

```
population GLM   rho +0.763   sign 95.7%
pairwise  GLM    rho +0.755   sign 95.7%
```

Essentially no advantage to joint estimation here. That is underpowered (23 connections) and the conditions favour the pairwise model: a sparse synthetic network with weak common input is close to the regime where pairwise estimation is correct. The population GLM should pull ahead where a third neuron actually drives two others — a densely recurrent HNN network with strong shared drive.

Recorded because the opposite temptation is real: had the CCG been the only baseline, the population GLM's +0.763 against CCG's +0.519 would have read as a clear win for joint estimation. Against the pairwise model it is not a win at all yet. **Whether joint estimation earns its cost is an open question this project should answer, not assume.**

---

## 18. Main metrics and the evaluation set

### The evaluation-set rule

A connection whose true delay exceeds the model's history window **cannot be represented by the model at all**. Scoring it as a miss measures the window, not the estimator.

```text
ALL TRUE CONNECTIONS
   ├── representable by current GLM (delay <= history window)  -> evaluate
   └── outside model temporal support                          -> report separately
```

Those connections are never silently dropped. The split is itself a result: *"53% of connections representable, 47% outside temporal support"* at 10×10, versus 99% at 5×5.

### Default dense network

1. Spearman ρ(|effective anatomical weight|, |inferred filter area|)
2. effective interaction-sign agreement
3. relationship between physical delay and effective interaction latency
4. each stratified by `loc` (proximal/distal/soma — as connection class, §14), receptor, source and target cell type

Sign recovery is effective E/I statistical influence, not receptor identification. Metrics are computed **within connection class**; inhibitory classes are degenerate by construction (§F).

### Sparse custom networks

ROC/AUC, precision-recall, edge ranking — well posed only where disconnected pairs genuinely exist.

---

## 19. Observational-bias experiment

Primary condition is **interneuron dropout**: all basket cells removed from the observed set. Interneurons are genuinely undersampled in extracellular recordings and are a small minority of the network.

> What apparent excitatory interactions emerge when inhibitory neurons become latent variables?

Random subsampling is retained as a comparison, not the central story.

---

## 20. Seed semantics and uncertainty

| Seed | Default | Controls | Across trials |
|---|---|---|---|
| `event_seed` | 2 | drive event times | incremented |
| `conn_seed` | 3 | which connections survive `probability < 1.0` | fixed |

These are different questions and must not be pooled into one error bar.

**Dense default network:** `probability = 1.0`, so connectivity is deterministic and `conn_seed` does nothing. Error bars there represent **input/trial variability on fixed anatomy**, not anatomical variability.

**Sparse custom networks:** hierarchical — circuit realizations (`conn_seed`) each containing event realizations (`event_seed`). Report the two sources separately.

Repetition counts come from measured metric stability, not habit.

---

## 21. Core experiment grid

Axis 1 — drive observability: true / proxy / hidden. Axis 2 — observed population: full / interneurons removed / random subsample. Sparsity is a separate knob for topology recovery. Additional dimensions only after the primary experiment gives a stable curve.

---

## 22. Performance engineering

Memory for a dense population design matrix is `samples × source neurons × basis functions × bytes`:

```
N = 70  (5×5):   70 × 8 =   560 features × 1.2e6 samples ->  2.69 GB f32 /  5.38 GB f64
N = 270 (10×10): 270 × 8 = 2160 features × 1.2e6 samples -> 10.37 GB f32 / 20.74 GB f64
```

(20 min at 1 ms bins.) These exclude responses, gradients, optimizer state, convolution temporaries and XLA copies — and float64 is **required** (§11), so the f64 column is the real figure.

Sequence: run real workload → profile → identify bottleneck → use existing NeMoS/JAX mechanism → profile again → only then implement what is missing. Tools: `SVRG`, `ProxSVRG`, minibatch loops via `GLM.update(..., n_samples=...)`, chunked feature construction, JIT/vectorization, GPU, and only if required a mechanism preserving temporal-history context across chunks.

No 10,000-neuron framework is built because it sounds impressive.

---

## 23. Performance result

Every optimization reports before/after peak memory and runtime, and must preserve the relevant metric within a predeclared tolerance. Otherwise it is not a successful optimization.

---

## 24. Software layout

```text
neurotwinbench/
├── ground_truth.py   HNN connectivity -> biological truth
├── simulate.py       HNN simulation + cache
├── synthetic.py      known-coupling calibration generator
├── fit.py            NeMoS population GLM
├── metrics.py        weight/sign/latency metrics, jitter null, CCG baseline
experiments/          one config/script per condition
scripts/              one-shot reconnaissance (ground_truth_audit.py)
figures/              deterministic figure generation
tests/
```

Not `plugins/`, `abstract_backends/`, `future_models/`, `dashboard/`, `generalized_simulator_interface/` — those appear only when two real use cases require the abstraction.

---

## 25. Explicitly not part of v1

Dashboard, web app, simulator plugin system, LFP/dipole connectivity inference, arbitrary future-model interfaces, 10,000-neuron support, elaborate CLI, large provenance framework. Pynapple suffices for the spike path; NWB only if a real interoperability requirement appears.

---

## 26. Reproducibility

Each simulation is identified by network model, `mesh_shape`, `legacy_mode`, connection config, drive config, `event_seed`, `conn_seed`, duration, software versions, MPI process count — cached by content hash. Results store config, circuit realization, drive realization, versions, raw metrics, summaries.

CI runs synthetic end-to-end calibration + a tiny HNN smoke test + ground-truth unit tests + metric tests + figure regeneration from stored results. Full simulations do not run in CI. Benchmark outputs commit as Parquet; every final figure regenerates from stored results.

---

## 27. Build order

```text
1.  OPERATING-POINT PILOT       ground-truth audit, 5×5 config, MPI benchmark,
                                spike stats, drive contribution, data budget,
                                memory estimate, latency resolution
2.  SYNTHETIC CALIBRATION       generator, NeMoS estimator, CCG baseline,
                                pairwise baseline, jitter null
3.  HNN GROUND-TRUTH RESOLVER   effective weight, physical delay, sign, loc
4.  FREEZE TOLERANCES           <-- deadline for pre-registration (§8)
5.  FIRST HNN RESULT
6.  EASIEST VALID HNN RECOVERABILITY TEST
7.  DIFFICULTY LADDER
8.  MAIN TWO-AXIS EXPERIMENT
9.  LOCATION / LATENCY ANALYSIS
10. SPARSE CIRCUIT-REALIZATION ANALYSIS
11. PROFILE REAL BOTTLENECK
12. PERFORMANCE WORK
13. WRITEUP + FIGURES + UPSTREAM PRs
```

Step 2 precedes step 3 deliberately: wiring HNN in first means debugging estimator, binning, basis and biophysics simultaneously.

---

## 28. Upstream contribution strategy

**HNN-Core:** a reusable public method resolving connectivity into per-pair ground truth (`src_gid`, `target_gid`, effective weight, physical delay, receptor, `loc`).

**NeMoS:** only functionality demonstrated missing by the real workload — e.g. if incremental updates suffice but temporal-history feature generation becomes the remaining memory bottleneck.

No upstream feature is invented in advance to decorate the repository.

---

## 29. Successful positive result

> The estimator recovers known coupling on synthetic GLM data within preregistered tolerances. On HNN networks, inferred statistical influence remains associated with effective anatomical weight, but the relationship differs across connection classes and between proximal and distal targets of the same projection. Removing drive information degrades recovery by X; hiding inhibitory neurons changes apparent excitatory coupling by Y. Physical delay and effective interaction latency are systematically related but distinct.

Every quantity comes from the experiment.

---

## 30. Successful negative result

> The estimator accurately recovers coupling when the generating process is itself a GLM, yet fails to recover HNN anatomical coupling even under the easiest physiologically defensible condition. Stratification shows where correspondence breaks across receptor classes, target-location groups, hidden common input, and biophysical spike-generation dynamics.

That is not a failed project. It answers the research question.

---

## 31. The key figure

One primary degradation figure, not twenty unrelated plots:

```text
Recovery
   ▲  ● synthetic GLM
   │      ● easy HNN
   │          ● dense HNN + true drive
   │              ● external proxy
   │                  ● drive hidden
   │                      ● interneurons hidden
   └────────────────────────────────────────►  difficulty
```

Stratified by the taxonomy actually stored. A section-specific experiment only if the first result justifies it.

---

## 32. Governing principles

1. One falsifiable result with controls beats a large framework with no result.
2. Calibrate the estimator before interpreting biophysics.
3. Convert assumptions into measurements — but arithmetic is not an assumption.
4. No primary metric without a baseline and a meaningful null.
5. Discover scientific failure as early as possible.
6. A negative result is valid if the measurement system is validated.
7. Do not equate statistical coupling with anatomical connectivity.
8. Use the ground truth HNN actually stores; do not invent an anatomical taxonomy.
9. Treat event variability and circuit variability as distinct uncertainty sources.
10. Do not derive a drive proxy from the modelled population.
11. Do not claim temporal resolution the pipeline has not demonstrated.
12. Define the evaluation set explicitly; never score a model on what it cannot represent, never silently drop those cases either.
13. Do not choose sample size or repetitions by habit; measure stability.
14. Never push "easy" HNN coupling outside a defensible physiological range without labelling it a stress test.
15. Reuse existing HNN-Core and NeMoS capabilities before implementing replacements.
16. Optimize only bottlenecks observed in a real experiment; every optimization gets a before/after profile.
17. No abstraction until at least two real cases require it.
18. Prefer upstream contributions over permanent local workarounds when the seam is clean.
19. Every final figure must regenerate from a clean checkout plus stored outputs.
20. Honest scope beats impressive-looking scope.

---

## 33. Final project identity

NeuroTwinBench is **not** a framework connecting HNN-Core, Pynapple, NeMoS and JAX. It is:

> **A controlled ground-truth experiment that uses biophysical neural simulation to determine when statistical functional connectivity corresponds to actual circuit structure, when that correspondence breaks, and which biological or observational mechanisms cause the failure.**

The software exists to make that experiment correct, reproducible, testable and efficient. Not the other way around.

---

## API references verified against hnn-core v0.6.1 / nemos 0.2.8

- `hnn_core.Network` — https://jonescompneurolab.github.io/hnn-core/stable/generated/hnn_core.Network.html
- `hnn_core.network_models` — https://jonescompneurolab.github.io/hnn-core/stable/_modules/hnn_core/network_models.html
- `cell._get_gaussian_connection` — xy-plane distance, `scaled_lamtha = lamtha * inplane_distance`
- NeMoS — https://nemos.readthedocs.io/
