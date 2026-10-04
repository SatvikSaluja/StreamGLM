"""Population GLM coupling estimation.

One function: spikes in, coupling filters out. Deliberately thin -- NeMoS does
the work, this only handles basis construction, the NaN padding that
convolution introduces, and unpacking coefficients back into filters.

`exogenous` exists from day one because PROJECT.md section 12 runs the identical
model in three drive conditions (true drive / observable proxy / hidden).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class FitResult:
    filters: np.ndarray  # (n_src, n_tgt, window) coupling filters in time
    baseline: np.ndarray  # (n_tgt,) log-rate intercept
    kernels: np.ndarray  # (window, n_basis) basis on the lag grid
    exog_filters: np.ndarray | None  # (n_exog, n_tgt, window) or None
    bin_s: float
    active: np.ndarray | None = None  # (n_neurons,) False where the neuron never fired

    @property
    def evaluable(self) -> np.ndarray:
        """(n_src, n_tgt) pairs whose coupling could be estimated at all.

        A neuron that never spiked supplies no events as a source and no rate to
        model as a target, so its coupling is not recoverable by any estimator.
        Intersect this into the `connected` mask before scoring: those pairs
        belong outside the evaluation set for the same reason connections beyond
        the history window do (PROJECT.md section 18). Scoring them would
        measure the recording, not the estimator.
        """
        n = self.filters.shape[0]
        if self.active is None:
            return np.ones((n, n), dtype=bool)
        return self.active[:, None] & self.active[None, :]

    @property
    def areas(self) -> np.ndarray:
        """Integrated coupling strength, (n_src, n_tgt). The primary estimate."""
        return self.filters.sum(axis=2)

    @property
    def latency_ms(self) -> np.ndarray:
        """Effective interaction latency from |filter| peak, (n_src, n_tgt).

        Named 'effective' deliberately: this is spike-to-spike influence latency,
        not physical synaptic delay (PROJECT.md section 15).
        """
        return np.abs(self.filters).argmax(axis=2) * self.bin_s * 1000.0


def fit_population_glm(
    spikes: np.ndarray,
    bin_s: float = 0.001,
    history_ms: float = 25.0,
    n_basis: int = 8,
    exogenous: np.ndarray | None = None,
    regularizer_strength: float = 1e-6,
    max_iter: int = 500,
    tol: float = 1e-6,
    solver: str | None = None,
    batch_size: int | None = None,
    float32: bool = False,
) -> FitResult:
    """Fit a coupled population GLM.

    Parameters
    ----------
    spikes : (n_bins, n_neurons) spike counts.
    exogenous : (n_bins, n_exog) observable external covariate, or None.
        Section 12 condition A passes true drive events; condition B passes a
        proxy built from a signal *external* to `spikes`; condition C passes None.
    tol, max_iter : solver stopping criteria. Measured at 14 neurons x 60 s:

        tol    maxiter   time     rho    sign
        1e-08     2000  158.2s  +0.497    96%
        1e-06      500   39.6s  +0.527    96%
        1e-04      300   23.7s  +0.520    94%
        1e-03      150   14.4s  +0.575    92%

        1e-6 is 4x faster than 1e-8 with no detectable loss, so it is the
        default. The spread across that column is sampling noise, not signal:
        one seed, ~45 connections, and the ordering reverses on sign accuracy.

        It was reasonable to suspect that spread meant early stopping was
        acting as the real regularizer, making `tol` a scientific parameter.
        Tested directly at 20 neurons x 120 s, varying only `tol`:

                     rho @1e-6   rho @1e-3
            1e-06       +0.690      +0.699
            1e-04       +0.690      +0.699
            1e-03       +0.690      +0.701

        Recovery is invariant to `tol` at both penalties, so that suspicion is
        not supported. Pin `tol` across compared conditions for reproducibility,
        but it is a performance knob after all.
    regularizer_strength : selected by held-out log-likelihood, not assumed.
        `select_regularizer` gives a clean unimodal curve peaking at 1e-3 on
        synthetic data (20 neurons x 120 s) -- 1000x the token value this
        defaulted to. The accuracy gain is small (rho +0.690 -> +0.699); the
        point is that the penalty is now explicit and reportable rather than
        implicit. Re-select per dataset: the optimum is a property of the data,
        and nothing licenses carrying a synthetic-data choice over to HNN.
    """
    import nemos as nmo

    spikes = np.asarray(spikes, dtype=float)
    n_total = spikes.shape[1]

    # A neuron that never fires has an infinite log-rate intercept, which nemos
    # reports as an opaque initialization failure. Drop such neurons, fit the
    # rest, and expand back to full size so callers keep their column indexing.
    active = spikes.sum(axis=0) > 0
    if not active.any():
        raise ValueError("no neuron fired; nothing to fit")

    design, target, n_coupling_cols, basis, window = _build_design(
        spikes[:, active], bin_s, history_ms, n_basis, exogenous, float32=float32
    )
    n_active = target.shape[1]

    # Default LBFGS is full-batch: every iteration touches all n_bins rows. At
    # 2.56M x 568 that is ~4 h and it OOMs at ~11 GB peak on a fifth of the data.
    # SVRG computes gradients on minibatches instead, which is what section 22
    # says to reach for and what nemos ships for exactly this case.
    solver_kwargs = {"maxiter": max_iter, "tol": tol}
    if batch_size is not None:
        solver_kwargs["batch_size"] = int(batch_size)
    glm = nmo.glm.PopulationGLM(
        regularizer="Ridge",
        regularizer_strength=regularizer_strength,
        solver_name=solver or "LBFGS",
        solver_kwargs=solver_kwargs,
    )
    glm.fit(design, target)

    coef = np.asarray(glm.coef_)  # (n_features, n_active)
    kernels = _basis_kernels(basis, window, n_basis)

    coupling_coef = _unpack(coef[:n_coupling_cols], n_active, n_basis)
    active_filters = np.einsum("ijk,tk->ijt", coupling_coef, kernels)

    filters = np.zeros((n_total, n_total, window))
    filters[np.ix_(active, active)] = active_filters

    exog_filters = None
    if exogenous is not None:
        n_exog = np.atleast_2d(exogenous).shape[1] if np.ndim(exogenous) > 1 else 1
        exog_coef = _unpack(coef[n_coupling_cols:], n_exog, n_basis)
        exog_filters = np.einsum("ijk,tk->ijt", exog_coef, kernels)

    baseline = np.zeros(n_total)
    baseline[active] = np.asarray(glm.intercept_)

    return FitResult(
        filters=filters,
        baseline=baseline,
        kernels=kernels,
        exog_filters=exog_filters,
        bin_s=bin_s,
        active=active,
    )


def fit_pairwise_glm(
    spikes: np.ndarray,
    bin_s: float = 0.001,
    history_ms: float = 25.0,
    n_basis: int = 8,
    regularizer_strength: float = 1e-3,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> FitResult:
    """Baseline 2: each source fitted without conditioning on the others.

    This is the baseline the population GLM has to beat (PROJECT.md section 17).
    The difference between them *is* the value of joint estimation: a pairwise
    model cannot explain away a correlation that actually comes from a third
    neuron, so it should show more spurious coupling in a recurrent network.

    Implemented as `n_neurons` fits rather than `n_neurons**2`. For each source
    a PopulationGLM is fitted with only that source's history predicting every
    target at once, which is the same estimator as a per-pair fit but vectorised
    over targets.
    """
    import nemos as nmo

    spikes = np.asarray(spikes, dtype=float)
    n_neurons = spikes.shape[1]
    kernels = None
    filters = None

    for src in range(n_neurons):
        design, target, _, basis, window = _build_design(
            spikes[:, [src]], bin_s, history_ms, n_basis, None
        )
        # `target` is the source column only; the real targets are every neuron,
        # aligned to the same rows the convolution left valid.
        valid_target = spikes[-len(design):]

        glm = nmo.glm.PopulationGLM(
            regularizer="Ridge",
            regularizer_strength=regularizer_strength,
            solver_kwargs={"maxiter": max_iter, "tol": tol},
        )
        glm.fit(design, valid_target)

        if kernels is None:
            kernels = _basis_kernels(basis, window, n_basis)
            filters = np.zeros((n_neurons, n_neurons, window))

        coef = np.asarray(glm.coef_)  # (n_basis, n_targets)
        filters[src] = np.einsum("kj,tk->jt", coef, kernels)

    return FitResult(
        filters=filters, baseline=np.zeros(n_neurons), kernels=kernels,
        exog_filters=None, bin_s=bin_s,
    )


def drive_proxy(
    drive_spikes: np.ndarray, bin_s: float = 0.001, smooth_ms: float = 25.0
) -> np.ndarray:
    """Section 12 condition B: a coarse, observable stand-in for the true drive.

    Smooths the aggregate external drive with a Gaussian kernel. This is
    deliberately *derived from the drive itself*, not from the modelled
    population: the rule in section 12 is that the proxy must not be a PSTH of
    the neurons whose coupling is being estimated, because such a proxy absorbs
    genuine recurrent coupling and makes the condition circular.

    The realism it encodes is partial knowledge -- you know roughly when input
    arrived, not the exact synaptic event times that condition A supplies.
    """
    from scipy.ndimage import gaussian_filter1d

    aggregate = np.asarray(drive_spikes, dtype=float).sum(axis=1)
    sigma_bins = (smooth_ms / 1000.0) / bin_s
    return gaussian_filter1d(aggregate, sigma=sigma_bins)[:, None]


def power_check(
    spikes: np.ndarray, n_basis: int = 8, n_exog: int = 0, bin_s: float = 0.001
) -> dict[str, float]:
    """Is there enough *information* to fit this model? Count spikes, not bins.

    A Poisson GLM's Fisher information scales with the number of events, so
    empty bins contribute almost nothing. Bins per predictor is therefore the
    wrong diagnostic, and it is dangerously reassuring: a 300 s HNN recording of
    70 neurons at 3.8 Hz gives 528 bins per predictor -- comfortably past any
    bins-based threshold -- while giving only 2.0 **spikes** per predictor.

    For reference, the synthetic calibration that recovered rho +0.80 had 16.1
    spikes per predictor, and rho +0.89 at 32.1. The same 300 s on HNN is 8x
    below the first of those, because HNN fires at 3.8 Hz against synthetic's
    9 Hz and carries 3.4x more predictors.

    This matters beyond efficiency. A weak result at 2 spikes per predictor is
    uninterpretable: model mismatch and insufficient data look identical, which
    is precisely the confound the synthetic fixture exists to rule out.
    """
    spikes = np.asarray(spikes)
    n_bins, n_neurons = spikes.shape
    n_predictors = n_neurons * n_basis + n_exog * n_basis
    spikes_per_neuron = float(spikes.sum()) / max(n_neurons, 1)

    return {
        "n_predictors": n_predictors,
        "bins_per_predictor": n_bins / n_predictors,
        "spikes_per_predictor": spikes_per_neuron / n_predictors,
        "spikes_per_neuron": spikes_per_neuron,
        "mean_rate_hz": spikes_per_neuron / (n_bins * bin_s),
    }


def _build_design(spikes, bin_s, history_ms, n_basis, exogenous, float32=False):
    """(design, target, n_coupling_cols, basis, window), NaN padding removed.

    `float32` halves the design matrix -- 11.6 GB to 5.8 GB at production size.
    x64 was enabled originally because LBFGS stalled before converging, but that
    was at `regularizer_strength=1e-6`, effectively unregularized. With the
    penalty now selected on held-out likelihood the conditioning is far better,
    so float32 is worth testing rather than assumed unusable. Verify recovery
    matches before trusting it on a result.
    """
    import jax

    jax.config.update("jax_enable_x64", not float32)
    import nemos as nmo

    spikes = np.asarray(spikes, dtype=float)
    window = int(round(history_ms / (bin_s * 1000.0)))

    basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs=n_basis, window_size=window)
    features = [basis.compute_features(spikes)]
    n_coupling_cols = features[0].shape[1]

    if exogenous is not None:
        exogenous = np.asarray(exogenous, dtype=float)
        if exogenous.ndim == 1:
            exogenous = exogenous[:, None]
        exog_basis = nmo.basis.RaisedCosineLogConv(
            n_basis_funcs=n_basis, window_size=window, label="exogenous"
        )
        features.append(exog_basis.compute_features(exogenous))

    # Build the final array ONCE at the target dtype and fill it in place.
    # The previous version held three full-size arrays simultaneously --
    # compute_features output, the hstack result, then the boolean-indexed copy
    # -- so peak memory before the optimizer even started was ~3x the design
    # matrix. Convolution pads only the leading `window` rows with NaN, so the
    # valid region is a contiguous slice and needs no fancy indexing.
    dtype = np.float32 if float32 else np.float64
    n_cols = sum(f.shape[1] for f in features)
    first_valid = int(np.argmax(~np.isnan(features[0]).any(axis=1)))
    n_rows = features[0].shape[0] - first_valid

    design = np.empty((n_rows, n_cols), dtype=dtype)
    col = 0
    for f in features:
        design[:, col:col + f.shape[1]] = f[first_valid:]
        col += f.shape[1]
    del features

    if np.isnan(design).any():
        raise ValueError("NaNs outside the leading convolution pad; "
                         "the contiguous-slice assumption does not hold")

    return (design, spikes[first_valid:].astype(dtype, copy=False),
            n_coupling_cols, basis, window)


def select_regularizer(
    spikes: np.ndarray,
    strengths: Sequence[float] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
    bin_s: float = 0.001,
    history_ms: float = 25.0,
    n_basis: int = 8,
    exogenous: np.ndarray | None = None,
    train_fraction: float = 0.8,
    tol: float = 1e-6,
    max_iter: int = 500,
) -> tuple[float, dict[float, float]]:
    """Pick `regularizer_strength` by held-out log-likelihood.

    Returns (best_strength, {strength: held_out_log_likelihood}).

    This exists to remove a confound, not to improve accuracy. With the penalty
    left at a token value the model is effectively unregularized and the
    solver's stopping point regularizes it implicitly via early stopping, which
    makes `tol` silently scientific (see `fit_population_glm`). Selecting the
    penalty on held-out data puts the regularization back where it is explicit
    and reportable.

    The split is **contiguous in time**, not random. Spike trains are
    autocorrelated and the design matrix is built from lagged history, so
    randomly interleaved test bins share history with training bins and the
    held-out score is optimistically biased.
    """
    import nemos as nmo

    design, target, _, _, _ = _build_design(
        spikes, bin_s, history_ms, n_basis, exogenous
    )
    split = int(round(train_fraction * len(design)))
    if split < 1 or split >= len(design):
        raise ValueError(f"train_fraction={train_fraction} leaves an empty split")

    scores: dict[float, float] = {}
    for strength in strengths:
        glm = nmo.glm.PopulationGLM(
            regularizer="Ridge",
            regularizer_strength=strength,
            solver_kwargs={"maxiter": max_iter, "tol": tol},
        )
        glm.fit(design[:split], target[:split])
        scores[strength] = float(
            glm.score(design[split:], target[split:], score_type="log-likelihood")
        )

    best = max(scores, key=scores.get)
    return best, scores


def drive_contribution(
    spikes: np.ndarray,
    drive: np.ndarray,
    bin_s: float = 0.001,
    history_ms: float = 25.0,
    n_basis: int = 8,
    train_fraction: float = 0.8,
    regularizer_strength: float = 1e-3,
    tol: float = 1e-6,
    max_iter: int = 500,
) -> dict[str, float | list[float]]:
    """Section 4: how much spiking does the external drive actually explain?

    Compares a drive-only model against an intercept-only model on held-out
    data, per neuron and for the population.

    This exists so the common-drive premise is a measured property of the
    operating regime rather than an assumption. PROJECT.md originally asserted
    that drive "dominates the variance"; that claim was downgraded to something
    to measure, and this is the measurement. If the drive explains little here,
    the true/proxy/hidden gradient of section 12 has little room to move and
    that is worth knowing before running the grid.

    Returns held-out log-likelihood for both models, their difference, and
    McFadden's pseudo-R2 = 1 - LL_drive/LL_intercept, per neuron.
    """
    import nemos as nmo

    spikes = np.asarray(spikes, dtype=float)
    active = spikes.sum(axis=0) > 0
    design, target, _, _, _ = _build_design(
        spikes[:, active],
        bin_s, history_ms, n_basis, drive,
    )
    # Keep only the drive columns: the coupling block is what we are excluding.
    n_coupling = int(active.sum()) * n_basis
    drive_design = design[:, n_coupling:]
    split = int(round(train_fraction * len(drive_design)))

    def _score(X: np.ndarray | None) -> np.ndarray:
        if X is None:  # intercept-only: a single constant column
            X = np.zeros((len(drive_design), 1))
        glm = nmo.glm.PopulationGLM(
            regularizer="Ridge", regularizer_strength=regularizer_strength,
            solver_kwargs={"maxiter": max_iter, "tol": tol},
        )
        glm.fit(X[:split], target[:split])
        return np.asarray(glm.score(
            X[split:], target[split:], score_type="log-likelihood",
            aggregate_sample_scores=lambda x: np.mean(x, axis=0),
        ))

    ll_drive = np.atleast_1d(_score(drive_design))
    ll_intercept = np.atleast_1d(_score(None))
    # McFadden's pseudo-R2. Log-likelihoods here are negative, so a drive model
    # that fits better gives a ratio below 1 and a positive score.
    with np.errstate(divide="ignore", invalid="ignore"):
        pseudo_r2 = 1.0 - (ll_drive / ll_intercept)

    return {
        "ll_drive": ll_drive.tolist(),
        "ll_intercept": ll_intercept.tolist(),
        "delta_ll": (ll_drive - ll_intercept).tolist(),
        "pseudo_r2": pseudo_r2.tolist(),
        "population_pseudo_r2": float(np.nanmean(pseudo_r2)),
        "n_active": int(active.sum()),
    }


def coupling_stability(
    spikes: np.ndarray,
    n_folds: int = 5,
    bin_s: float = 0.001,
    history_ms: float = 25.0,
    n_basis: int = 8,
    exogenous: np.ndarray | None = None,
    regularizer_strength: float = 1e-3,
    tol: float = 1e-6,
    max_iter: int = 500,
) -> dict[str, np.ndarray]:
    """Per-connection uncertainty, by refitting on contiguous folds.

    Every headline number in this project carries uncertainty across seeds, but
    that says nothing about *which individual connections* were estimated
    reliably. A pair recovered consistently in every fold and a pair whose
    estimate flips sign between folds both contribute equally to a Spearman rho,
    and they should not be read the same way.

    Folds are contiguous blocks, not random subsets: the design matrix is built
    from lagged history, so interleaved folds share history and their estimates
    are not independent.

    Returns per-pair `mean`, `sd`, and `sign_consistency` (the fraction of folds
    agreeing with the majority sign).

    **Use `sign_consistency`, not `sd`.** Measured on synthetic data with known
    connectivity, 8 neurons over 60 s in 4 folds:

        sd                0.6227 connected vs 0.5745 unconnected   not useful
        sign_consistency  0.88   connected vs 0.47   unconnected   useful

    Fold-to-fold magnitude variability barely separates real connections from
    spurious ones -- both wobble by a similar amount. Sign agreement separates
    them cleanly, and 0.47 for unconnected pairs is chance, which is exactly
    what an estimate carrying no information should look like. A pair with a
    large inferred magnitude and `sign_consistency` near 0.5 is noise that
    happened to land somewhere, and reporting its magnitude without this would
    be misleading.
    """
    spikes = np.asarray(spikes, dtype=float)
    n_bins = spikes.shape[0]
    edges = np.linspace(0, n_bins, n_folds + 1).astype(int)

    areas, estimable = [], []
    for lo, hi in zip(edges, edges[1:]):
        exog = None if exogenous is None else np.asarray(exogenous)[lo:hi]
        result = fit_population_glm(
            spikes[lo:hi], bin_s=bin_s, history_ms=history_ms, n_basis=n_basis,
            exogenous=exog, regularizer_strength=regularizer_strength,
            tol=tol, max_iter=max_iter,
        )
        areas.append(result.areas)
        estimable.append(result.evaluable)

    stack = np.stack(areas)  # (n_folds, n_src, n_tgt)
    valid = np.stack(estimable)

    # Count agreement only over folds that actually produced an estimate. A
    # neuron silent in one fold gets a zero there, and counting that zero as
    # disagreement conflates "the estimate flipped sign" with "there was no
    # estimate" -- which are different, and the second is not unreliability. At
    # HNN firing rates intermittent silence is common, so without this a
    # perfectly stable connection reads 0.75 instead of 1.0.
    signs = np.where(valid, np.sign(stack), np.nan)
    with np.errstate(invalid="ignore"):
        majority = np.sign(np.nansum(signs, axis=0))
        # Mask AFTER comparing: `nan == x` is False, not nan, so comparing first
        # and then taking nanmean would skip nothing and silently count every
        # missing fold as a disagreement -- the exact bug this guards.
        agree = np.where(valid, (signs == majority).astype(float), np.nan)
        agreement = np.nanmean(agree, axis=0)

    n_valid = valid.sum(axis=0)
    agreement = np.where(n_valid > 0, agreement, np.nan)

    return {
        "mean": np.where(n_valid > 0, np.nansum(np.where(valid, stack, np.nan), axis=0)
                         / np.maximum(n_valid, 1), np.nan),
        "sd": np.nanstd(np.where(valid, stack, np.nan), axis=0),
        "sign_consistency": agreement,
        "n_folds_estimable": n_valid,
        "folds": stack,
    }


def _unpack(coef: np.ndarray, n_src: int, n_basis: int) -> np.ndarray:
    """(n_src*n_basis, n_tgt) -> (n_src, n_tgt, n_basis).

    NeMoS lays convolved features out source-major: all `n_basis` columns for
    source 0, then source 1, and so on. `test_column_order` pins this.
    """
    n_tgt = coef.shape[1]
    return coef.reshape(n_src, n_basis, n_tgt).transpose(0, 2, 1)


def _basis_kernels(basis, window: int, n_basis: int) -> np.ndarray:
    """Basis evaluated on the lag grid, (window, n_basis)."""
    _, kernels = basis.evaluate_on_grid(window)
    return np.asarray(kernels)
