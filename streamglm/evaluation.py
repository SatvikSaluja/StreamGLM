"""Leakage-aware temporal selection and predictive scoring."""
import numpy as np
from scipy.special import gammaln
from .model import StreamingGLM


def split_recording(recording, fractions=(.6, .2, .2)):
    """Partition EACH epoch chronologically; all split histories reset."""
    if len(fractions) != 3 or min(fractions) <= 0 or not np.isclose(sum(fractions), 1):
        raise ValueError("need three positive fractions summing to one")
    groups = [[], [], []]
    offset = 0
    for length in recording.epoch_lengths:
        edges = np.r_[0, np.floor(length*np.cumsum(fractions)[:2]).astype(int), length]
        if np.min(np.diff(edges)) < 1:
            raise ValueError("epoch too short for three nonempty splits")
        for group, a, b in zip(groups, edges[:-1], edges[1:]):
            group.append((offset+int(a), offset+int(b)))
        offset += length
    return tuple(recording.select(group) for group in groups)


def score(model, params, recording, baseline_mean=None):
    """Unpenalized log likelihood, with factorials, in nats per valid bin."""
    total, spikes, n = 0., 0., 0
    null_ll = 0.
    if baseline_mean is not None:
        baseline_mean = np.maximum(np.asarray(baseline_mean), 1e-12)
    for start, rate, valid in model.predict_chunks(params, recording):
        y = np.asarray(recording.counts[start:start+len(rate)])[valid]
        if len(y) == 0:
            continue
        rate = rate[valid]
        if not np.all(np.isfinite(rate)) or np.any(rate <= 0):
            raise FloatingPointError("invalid predicted rates")
        total += float(np.sum(y*np.log(rate)-rate-gammaln(y+1)))
        if baseline_mean is not None:
            null_ll += float(np.sum(y*np.log(baseline_mean)-baseline_mean-gammaln(y+1)))
        spikes += float(y.sum())
        n += len(y)
    if n == 0:
        raise ValueError("no valid scoring bins")
    result = {"log_likelihood_per_bin": total/n, "valid_bins": n,
              "spikes": spikes, "bin_s": recording.bin_s}
    if baseline_mean is not None:
        result.update({"intercept_log_likelihood_per_bin": null_ll/n,
            "gain_nats_per_bin": (total-null_ll)/n,
            "gain_bits_per_spike": (total-null_ll)/(np.log(2)*spikes) if spikes else None})
    return result


def select_rank(train, validation, basis, ranks=(0, 1, 2), seeds=(0, 1),
                chunk_size=1024, ridge=.01, max_iter=300, tolerance=1e-7, callback=None):
    """Choose a converged candidate by validation likelihood, never test data.

    Rank zero is an independent self-history model. Every candidate has
    separate self-history and an off-diagonal low-rank interaction term.
    """
    candidates, best = [], None
    mean = np.maximum(train.mean_counts(len(basis)), 1e-8)
    for rank in ranks:
        for seed in seeds if rank else seeds[:1]:
            model = StreamingGLM(basis, chunk_size, rank, ridge, self_history=True)
            fitted = model.fit(train, model.initialize(train.n_neurons, seed, mean),
                               max_iter=max_iter, tolerance=tolerance)
            metric = score(model, fitted.params, validation, mean)
            row = {"rank": rank, "seed": seed, "converged": fitted.converged,
                "message": fitted.message, "iterations": fitted.iterations,
                "seconds": fitted.seconds, "gradient_inf_norm": fitted.gradient_inf_norm,
                "validation": metric}
            candidates.append(row)
            if callback:
                callback(candidates)
            if fitted.converged and (best is None or metric["log_likelihood_per_bin"] > best[0]):
                best = (metric["log_likelihood_per_bin"], model, fitted, row)
    if best is None:
        raise RuntimeError("no candidate converged; increase iterations or improve conditioning")
    return best[1], best[2], {"selected": best[3], "candidates": candidates}
