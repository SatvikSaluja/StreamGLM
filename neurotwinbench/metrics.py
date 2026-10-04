"""Recovery metrics, the jitter null, and the evaluation-set rule.

No primary metric is reported without a baseline and a null (PROJECT.md
principle 4). The evaluation set is always explicit: a connection whose true
delay exceeds the model's history window cannot be represented by the model at
all, so scoring it as a miss measures the window, not the estimator. Those
connections are reported separately, never silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr


@dataclass
class Recovery:
    weight_spearman: float
    sign_accuracy: float
    latency_error_ms: float
    n_evaluated: int
    n_outside_support: int

    @property
    def fraction_representable(self) -> float:
        total = self.n_evaluated + self.n_outside_support
        return self.n_evaluated / total if total else float("nan")

    def __str__(self) -> str:
        return (
            f"rho={self.weight_spearman:+.3f}  sign={self.sign_accuracy:.1%}  "
            f"latency_err={self.latency_error_ms:.2f}ms  "
            f"n={self.n_evaluated} ({self.fraction_representable:.0%} representable)"
        )


def representable(true_delay_ms: np.ndarray, history_ms: float) -> np.ndarray:
    """Mask of connections the model's history window can express at all."""
    return np.asarray(true_delay_ms) <= history_ms


def evaluate(
    true_weight: np.ndarray,
    inferred_area: np.ndarray,
    connected: np.ndarray,
    true_delay_ms: np.ndarray | None = None,
    inferred_latency_ms: np.ndarray | None = None,
    history_ms: float | None = None,
) -> Recovery:
    """Compare inferred coupling against ground truth on the representable set.

    Parameters
    ----------
    true_weight, inferred_area, connected : (n_src, n_tgt) arrays.
        `connected` marks which pairs are genuinely connected; unconnected pairs
        have no true weight to rank against.
    true_delay_ms, history_ms : partition into representable / outside support.
        Omit both to evaluate every connected pair (correct for synthetic data,
        where filters are inside the window by construction).
    """
    connected = np.asarray(connected, dtype=bool)

    if true_delay_ms is not None and history_ms is not None:
        in_support = representable(true_delay_ms, history_ms)
        n_outside = int(np.sum(connected & ~in_support))
        connected = connected & in_support
    else:
        n_outside = 0

    true_w = np.asarray(true_weight)[connected]
    inferred_w = np.asarray(inferred_area)[connected]

    # Magnitude recovery is a rank question; sign is scored separately so that a
    # sign flip does not silently inflate the correlation.
    rho = float(spearmanr(np.abs(true_w), np.abs(inferred_w)).statistic) if true_w.size > 1 else float("nan")
    sign_acc = float(np.mean(np.sign(true_w) == np.sign(inferred_w))) if true_w.size else float("nan")

    latency_err = float("nan")
    if true_delay_ms is not None and inferred_latency_ms is not None:
        err = np.abs(np.asarray(inferred_latency_ms)[connected] - np.asarray(true_delay_ms)[connected])
        latency_err = float(np.mean(err)) if err.size else float("nan")

    return Recovery(
        weight_spearman=rho,
        sign_accuracy=sign_acc,
        latency_error_ms=latency_err,
        n_evaluated=int(connected.sum()),
        n_outside_support=n_outside,
    )


def edge_detection(
    connected: np.ndarray,
    inferred_area: np.ndarray,
    candidate: np.ndarray | None = None,
) -> dict[str, float]:
    """Topology recovery: can real edges be ranked above non-edges?

    Only well posed when designed non-edges exist in meaningful numbers, which
    means a sparse network (`probability < 1`). In the default model nearly
    every eligible pair is connected, so this is degenerate and must not be
    reported -- ranking 3500 edges against ~0 non-edges is not a detection task
    (PROJECT.md section 2B).

    `candidate` restricts scoring to pairs that *could* have been connected --
    the source and target populations of some connection class. Without it the
    non-edge set is dominated by pairs the model never proposed (L5->L2, say),
    which no estimator should be credited for rejecting.

    Returns roc_auc, pr_auc, n_edges, n_non_edges, and prevalence. Prevalence is
    included because PR-AUC has no fixed chance level: its baseline IS the
    prevalence, so a PR-AUC of 0.8 means nothing until you know whether 80% of
    candidate pairs are edges.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    connected = np.asarray(connected, dtype=bool)
    score = np.abs(np.asarray(inferred_area, dtype=float))

    # Copy: np.asarray does not copy an array that is already bool, and the
    # fill_diagonal below would then silently zero the caller's mask.
    mask = np.ones_like(connected) if candidate is None else np.array(candidate, dtype=bool)
    np.fill_diagonal(mask, False)

    truth, values = connected[mask], score[mask]
    n_edges, n_non = int(truth.sum()), int((~truth).sum())
    if n_edges == 0 or n_non == 0:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"),
                "n_edges": n_edges, "n_non_edges": n_non,
                "prevalence": float("nan")}

    return {
        "roc_auc": float(roc_auc_score(truth, values)),
        "pr_auc": float(average_precision_score(truth, values)),
        "n_edges": n_edges,
        "n_non_edges": n_non,
        "prevalence": n_edges / (n_edges + n_non),
    }


def jitter(spikes: np.ndarray, width_bins: int, seed: int = 0) -> np.ndarray:
    """Displace each spike uniformly within +/- `width_bins`.

    Destroys precise short-timescale coupling while approximately preserving
    each neuron's slow rate modulation and the common-drive structure -- a far
    more demanding null than full permutation, which destroys rate structure too
    and drives the floor to zero (PROJECT.md section 16).
    """
    rng = np.random.default_rng(seed)
    spikes = np.asarray(spikes)
    n_bins, n_neurons = spikes.shape
    out = np.zeros_like(spikes)

    for neuron in range(n_neurons):
        times = np.repeat(np.arange(n_bins), spikes[:, neuron])
        if times.size == 0:
            continue
        moved = times + rng.integers(-width_bins, width_bins + 1, size=times.size)
        moved = np.clip(moved, 0, n_bins - 1)
        out[:, neuron] = np.bincount(moved, minlength=n_bins)

    return out


def cross_correlogram(spikes: np.ndarray, max_lag: int) -> np.ndarray:
    """Baseline 1: CCG peak deviation from mean, (n_src, n_tgt).

    Deliberately crude. Its job is to be the number the population GLM must
    beat -- if the GLM does not, that is a result, but only if this was run.
    """
    spikes = np.asarray(spikes, dtype=float)
    n_bins, n_neurons = spikes.shape
    centred = spikes - spikes.mean(axis=0, keepdims=True)
    out = np.zeros((n_neurons, n_neurons))

    for lag in range(1, max_lag + 1):
        # source leads target by `lag` bins
        cov = centred[:-lag].T @ centred[lag:] / (n_bins - lag)
        out = np.where(np.abs(cov) > np.abs(out), cov, out)

    return out


def _self_check() -> None:
    rng = np.random.default_rng(0)
    n = 6
    true_w = rng.normal(size=(n, n))
    connected = ~np.eye(n, dtype=bool)

    # Perfect recovery must score perfectly, or the metric is wrong.
    perfect = evaluate(true_w, true_w, connected)
    assert np.isclose(perfect.weight_spearman, 1.0), perfect
    assert np.isclose(perfect.sign_accuracy, 1.0), perfect

    # Sign-flipped input must keep rho high but score sign at zero: the two are
    # measured independently on purpose.
    flipped = evaluate(true_w, -true_w, connected)
    assert np.isclose(flipped.weight_spearman, 1.0), flipped
    assert np.isclose(flipped.sign_accuracy, 0.0), flipped

    # Evaluation-set rule: connections beyond the window are excluded, counted.
    delays = np.full((n, n), 10.0)
    delays[0, :] = 40.0
    split = evaluate(true_w, true_w, connected, true_delay_ms=delays, history_ms=25.0)
    assert split.n_outside_support == int(connected[0].sum()), split
    assert split.fraction_representable < 1.0

    # Jitter must preserve spike count but destroy fine timing.
    spikes = rng.poisson(0.01, size=(5000, 4))
    jittered = jitter(spikes, width_bins=15, seed=1)
    assert jittered.sum() == spikes.sum(), "jitter must conserve spikes"
    assert not np.array_equal(jittered, spikes)

    print(f"ok: perfect={perfect}\n    split={split}\n    jitter conserved {spikes.sum()} spikes")


if __name__ == "__main__":
    _self_check()
