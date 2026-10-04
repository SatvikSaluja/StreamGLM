"""Explicit small-data export to a NeMoS PopulationGLM."""
import inspect
import jax.numpy as jnp
import numpy as np
from .reference import materialize_weights


class NeMoSRecordingLoader:
    """Replayable NeMoS batches from a Recording, with epoch-local history.

    Uses NeMoS's own convolution; retains short tails and counts every valid
    response row exactly once. Memory is proportional to chunk*neurons*basis.
    Compatible with upstream's DataLoader protocol without importing a newer
    NeMoS version until iteration. No full design matrix is constructed.
    """

    def __init__(self, recording, basis, chunk_size=1024):
        self.recording = recording
        self.basis = np.asarray(basis, dtype=float)
        if self.basis.ndim != 2 or min(self.basis.shape) < 1 or not np.isfinite(self.basis).all():
            raise ValueError('basis must be a finite nonempty matrix')
        if chunk_size < 1 or int(chunk_size) != chunk_size:
            raise ValueError('chunk_size must be a positive integer')
        self.chunk_size = int(chunk_size)
        if self.n_samples < 1:
            raise ValueError('no complete-history samples')

    @property
    def n_samples(self):
        return self.recording.n_valid(len(self.basis))

    def __iter__(self):
        import nemos as nmo
        for batch in self.recording.batches(self.chunk_size, len(self.basis)):
            if not batch.valid.any():
                continue
            features = nmo.convolve.create_convolutional_predictor(
                self.basis, batch.history, shift=True)
            design = np.asarray(features)[len(self.basis):][batch.valid]
            yield design.reshape(len(design), -1), batch.counts[batch.valid]

    def sample_batch(self):
        return next(iter(self))


def to_nemos(model, params, max_bytes=512*2**20):
    """Export effective coefficients for NeMoS causal-convolution features.

    NeMoS features must be constructed with the same basis, causal shift=True,
    and flattened from (time, source, basis). This allocates K*N*N coefficients;
    export deliberately has a byte guard and is not part of streaming fitting.
    The returned estimator supports prediction/scoring, not transferred optimizer
    history or a claim that an unrestricted NeMoS fit would have these parameters.
    """
    import nemos as nmo
    n = len(params["b"])
    model._validate_params(params, n)
    if model.n_basis*n*n*np.dtype(model.basis.dtype).itemsize > max_bytes:
        raise MemoryError("dense export exceeds max_bytes; keep factorized parameters")
    if "inverse_link_function" in inspect.signature(nmo.glm.PopulationGLM).parameters:
        glm = nmo.glm.PopulationGLM(observation_model="Poisson", inverse_link_function=jnp.exp)
    else:
        glm = nmo.glm.PopulationGLM(observation_model=
            nmo.observation_models.PoissonObservations(inverse_link_function=jnp.exp))
    weights = materialize_weights(params)
    glm.coef_ = jnp.asarray(weights.transpose(1, 0, 2).reshape(n*model.n_basis, n))
    glm.intercept_ = jnp.asarray(params["b"])
    glm.scale_ = 1.
    return glm
