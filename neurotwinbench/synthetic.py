"""Known-coupling GLM spike generator: the calibration fixture.

Spikes are generated from a coupled Poisson GLM whose coupling filters we write
down ourselves. Fitting these spikes must recover those filters. This separates
estimator error from biophysical model mismatch -- without it, poor recovery on
HNN data is uninterpretable (PROJECT.md section 8).

Filters live on the *same* raised-cosine basis used for fitting, so the model is
exactly well specified and any recovery error is estimation error alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticNetwork:
    """Ground truth plus the spikes it generated."""

    spikes: np.ndarray  # (n_bins, n_neurons) counts
    coupling: np.ndarray  # (n_src, n_tgt, n_basis) basis coefficients
    baseline: np.ndarray  # (n_neurons,) log-rate intercept
    kernels: np.ndarray  # (window, n_basis) basis evaluated on lag grid
    is_inhibitory: np.ndarray  # (n_neurons,) bool
    bin_s: float

    @property
    def filters(self) -> np.ndarray:
        """Coupling filters in time, (n_src, n_tgt, window)."""
        return np.einsum("ijk,tk->ijt", self.coupling, self.kernels)

    @property
    def rates_hz(self) -> np.ndarray:
        return self.spikes.mean(axis=0) / self.bin_s


def raised_cosine_log(n_basis: int, window: int) -> np.ndarray:
    """Log-spaced raised-cosine kernels, (window, n_basis).

    Uses nemos' own basis rather than a reimplementation, so the generating
    filters and the fitted filters live in the *identical* space. Any recovery
    error is then estimation error, not basis mismatch -- which is the entire
    point of the fixture.
    """
    import nemos as nmo

    basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs=n_basis, window_size=window)
    _, kernels = basis.evaluate_on_grid(window)
    return np.asarray(kernels, dtype=float)


def make_coupling(
    n_neurons: int,
    n_basis: int,
    is_inhibitory: np.ndarray,
    connection_prob: float,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Random sparse coupling coefficients, sign set by source cell type."""
    mask = rng.random((n_neurons, n_neurons)) < connection_prob
    np.fill_diagonal(mask, False)

    magnitude = rng.gamma(shape=2.0, scale=strength / 2.0, size=(n_neurons, n_neurons, n_basis))
    sign = np.where(is_inhibitory, -1.0, 1.0)[:, None, None]
    return magnitude * sign * mask[:, :, None]


def simulate(
    n_neurons: int = 20,
    duration_s: float = 600.0,
    bin_s: float = 0.001,
    window: int = 25,
    n_basis: int = 8,
    frac_inhibitory: float = 0.25,
    connection_prob: float = 0.3,
    strength: float = 0.6,
    target_rate_hz: float = 8.0,
    max_rate_hz: float = 200.0,
    seed: int = 0,
) -> SyntheticNetwork:
    """Simulate a coupled Poisson GLM.

    `max_rate_hz` clips the instantaneous rate. Recurrent excitation in a
    log-link GLM is genuinely unstable, so this is a required knob, not a
    convenience -- without it a moderately connected network diverges.
    """
    rng = np.random.default_rng(seed)
    n_bins = int(round(duration_s / bin_s))

    is_inhibitory = np.zeros(n_neurons, dtype=bool)
    is_inhibitory[: int(round(frac_inhibitory * n_neurons))] = True
    rng.shuffle(is_inhibitory)

    kernels = raised_cosine_log(n_basis, window)
    coupling = make_coupling(n_neurons, n_basis, is_inhibitory, connection_prob, strength, rng)
    baseline = np.full(n_neurons, np.log(target_rate_hz * bin_s))

    spikes = np.zeros((n_bins, n_neurons), dtype=np.int16)
    log_ceiling = np.log(max_rate_hz * bin_s)

    for t in range(1, n_bins):
        lo = max(0, t - window)
        history = spikes[lo:t][::-1]  # most recent lag first
        features = kernels[: history.shape[0]].T @ history  # (n_basis, n_src)
        log_rate = baseline + np.einsum("ki,ijk->j", features, coupling)
        rate = np.exp(np.minimum(log_rate, log_ceiling))
        spikes[t] = rng.poisson(rate)

    return SyntheticNetwork(
        spikes=spikes,
        coupling=coupling,
        baseline=baseline,
        kernels=kernels,
        is_inhibitory=is_inhibitory,
        bin_s=bin_s,
    )


def _self_check() -> None:
    """The generator is only useful if its ground truth is self-consistent."""
    kernels = raised_cosine_log(n_basis=8, window=25)
    assert kernels.shape == (25, 8)
    assert np.all(kernels >= 0.0)
    assert np.all(np.isfinite(kernels))
    # Each kernel peaks later than the previous one: the basis spans the window.
    peaks = kernels.argmax(axis=0)
    assert np.all(np.diff(peaks) >= 0), f"kernel peaks not ordered: {peaks}"

    net = simulate(n_neurons=8, duration_s=20.0, connection_prob=0.4, seed=1)
    rates = net.rates_hz
    assert np.all(rates > 0.1), f"silent neurons: {rates}"
    assert np.all(rates < 200.0), f"runaway excitation: {rates}"

    # Sign of every realised filter must match its source cell type.
    filters = net.filters
    area = filters.sum(axis=2)
    connected = area != 0.0
    expected = np.where(net.is_inhibitory, -1.0, 1.0)[:, None]
    assert np.all(np.sign(area[connected]) == np.broadcast_to(expected, area.shape)[connected]), (
        "filter sign does not match source cell type"
    )

    print(f"ok: {net.spikes.shape[1]} neurons, {net.spikes.sum()} spikes, "
          f"rates {rates.min():.1f}-{rates.max():.1f} Hz, "
          f"{int(connected.sum())} connections")


if __name__ == "__main__":
    _self_check()
