"""Small, known low-rank recurrent GLM fixture; not a power calibration."""
import numpy as np
from .reference import materialize_weights


def fixture(n_bins=12000, n_neurons=8, seed=1):
    rng = np.random.default_rng(seed)
    lag = np.arange(1, 9)
    basis = np.column_stack([np.exp(-lag / scale) for scale in (1.5, 4.)])
    basis /= basis.sum(axis=0)
    # Inhibitory weights and nonnegative kernels bound the conditional mean
    # above by exp(b), without clipping or changing the generating link.
    params = {"U": rng.uniform(.15, .55, (2, n_neurons, 2)),
              "V": -rng.uniform(.15, .55, (2, n_neurons, 2)),
              "b": np.full(n_neurons, np.log(.35))}
    w = materialize_weights(params)
    filters = np.einsum("hk,kij->hij", basis, w)
    counts = np.zeros((n_bins, n_neurons), dtype=np.int32)
    for t in range(n_bins):
        h = min(t, len(basis))
        eta = params["b"].copy()
        if h:
            eta += np.einsum("hi,hij->j", counts[t-h:t][::-1], filters[:h])
        counts[t] = rng.poisson(np.exp(eta))
    return counts, basis, params
