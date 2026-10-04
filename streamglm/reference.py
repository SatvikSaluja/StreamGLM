"""Small-data materialized reference. Not for production scaling."""
import numpy as np


def features(counts, basis, epoch_lengths=None):
    """Return features (time, basis, source) and a complete-history mask."""
    counts, basis = np.asarray(counts), np.asarray(basis)
    h, k = basis.shape
    lengths = [len(counts)] if epoch_lengths is None else epoch_lengths
    x = np.zeros((len(counts), k, counts.shape[1]), dtype=float)
    valid = np.zeros(len(counts), dtype=bool)
    start = 0
    for length in lengths:
        for lag in range(1, min(h, length - 1) + 1):
            x[start + lag:start + length] += basis[lag - 1, :, None] * counts[start:start + length - lag, None, :]
        if length > h:
            valid[start + h:start + length] = True
        start += length
    return x, valid


def materialize_weights(params):
    if "W" in params:
        weights = np.array(params["W"])
    else:
        weights = np.einsum("kir,kjr->kij", params["U"], params["V"])
    if "S" in params:
        diagonal = np.arange(weights.shape[1])
        weights[:, diagonal, diagonal] = params["S"]
    return weights
