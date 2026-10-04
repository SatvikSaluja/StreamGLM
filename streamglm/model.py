"""Exact streamed Poisson objectives and low-rank coupling.

W[k, source, target] multiplies filtered source counts. Basis rows are lags
1 through H. Predictions are expected counts per bin, not Hz. No clipping
modifies the exp link. Applications can enable jax_enable_x64 explicitly.
"""
from dataclasses import dataclass
import time
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
from scipy.optimize import minimize


@dataclass
class FitResult:
    params: dict
    converged: bool
    message: str
    iterations: int
    evaluations: int
    objective: float
    gradient_inf_norm: float
    seconds: float
    trace: list


class StreamingGLM:
    """Dense W or W_k=U_k V_k.T, with an unrestricted intercept.

    Ridge penalizes W, not its factors. Factorization is nonconvex and
    changes the model family. Chunks change only objective evaluation.
    """

    def __init__(self, basis, chunk_size=1024, rank=None, ridge=0.0, self_history=False, dtype=None):
        basis = np.asarray(basis)
        if basis.ndim != 2 or min(basis.shape) < 1 or not np.all(np.isfinite(basis)):
            raise ValueError("basis must be a finite (lag, basis) matrix")
        if chunk_size < 1 or int(chunk_size) != chunk_size:
            raise ValueError("chunk_size must be a positive integer")
        if rank is not None and (rank < 0 or int(rank) != rank):
            raise ValueError("rank must be a nonnegative integer")
        if not np.isfinite(ridge) or ridge < 0:
            raise ValueError("ridge must be finite and nonnegative")
        dtype = np.dtype(dtype or ("float64" if jax.config.x64_enabled else "float32"))
        if dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("dtype must be float32 or float64")
        if dtype == np.dtype("float64") and not jax.config.x64_enabled:
            raise ValueError("enable jax_enable_x64 before requesting float64")
        self.basis = jnp.asarray(basis, dtype=dtype)
        self.chunk_size = int(chunk_size)
        self.rank = None if rank is None else int(rank)
        self.ridge = ridge
        self.self_history = bool(self_history)
        self.history, self.n_basis = basis.shape
        self._eta = jax.jit(self._linear_predictor)
        self._value_grad = jax.jit(jax.value_and_grad(self._chunk_loss))
        self._penalty_grad = jax.jit(jax.value_and_grad(self._penalty))

    def initialize(self, n_neurons, seed=0, mean_count=0.05):
        if n_neurons < 1 or (self.rank is not None and self.rank > n_neurons):
            raise ValueError("need positive n_neurons and 0 <= rank <= n_neurons")
        mean = np.broadcast_to(np.asarray(mean_count), (n_neurons,))
        if not np.all(np.isfinite(mean)) or np.any(mean <= 0):
            raise ValueError("mean_count must be finite and positive")
        dtype = self.basis.dtype
        params = {"b": jnp.asarray(np.log(mean), dtype=dtype)}
        if self.self_history:
            params["S"] = jnp.zeros((self.n_basis, n_neurons), dtype=dtype)
        if self.rank is None:
            params["W"] = jnp.zeros((self.n_basis, n_neurons, n_neurons), dtype=dtype)
        else:
            rng = np.random.default_rng(seed)
            for name in ("U", "V"):
                params[name] = jnp.asarray(rng.normal(0, .03, (self.n_basis, n_neurons, self.rank)), dtype=dtype)
        return params

    def _filter(self, x, kernel, count):
        def body(lag, out):
            rows = jax.lax.dynamic_slice_in_dim(x, self.history - lag - 1, count)
            return out + kernel[lag] * rows
        return jax.lax.fori_loop(0, self.history, body, jnp.zeros((count, x.shape[1]), x.dtype))

    def _linear_predictor(self, params, x):
        count = x.shape[0] - self.history
        initial = jnp.broadcast_to(params["b"], (count, params["b"].shape[0]))

        def body(k, eta):
            if self.rank is None:
                filtered = self._filter(x, self.basis[:, k], count)
                w = params["W"][k]
                if self.self_history:
                    w = w - jnp.diag(jnp.diag(w))
                contribution = filtered @ w
            else:
                # (X*b)U = (XU)*b: convolve only rank-r channels.
                projected = x @ params["U"][k]
                filtered = self._filter(projected, self.basis[:, k], count)
                contribution = filtered @ params["V"][k].T
            if self.self_history:
                own = self._filter(x, self.basis[:, k], count)
                if self.rank is not None:
                    diagonal = jnp.sum(params["U"][k] * params["V"][k], axis=1)
                    contribution -= own * diagonal
                contribution += own * params["S"][k]
            return eta + contribution

        # Recompute basis intermediates during reverse-mode differentiation.
        return jax.lax.fori_loop(0, self.n_basis, jax.checkpoint(body), initial)

    def _chunk_loss(self, params, x, y, valid):
        eta = self._linear_predictor(params, x)
        eta = jnp.where(valid[:, None], eta, 0.)
        return jnp.sum(jnp.where(valid[:, None], jnp.exp(eta) - y * eta, 0.))

    def _penalty(self, params):
        if self.rank is None:
            norm = jnp.sum(params["W"] ** 2)
            diagonal = jnp.diagonal(params["W"], axis1=1, axis2=2)
        else:
            # ||UV.T||_F^2 = trace((U.T U)(V.T V)); no N x N allocation.
            uu = jnp.einsum("knr,kns->krs", params["U"], params["U"])
            vv = jnp.einsum("knr,kns->krs", params["V"], params["V"])
            norm = jnp.sum(uu * vv)
            diagonal = jnp.sum(params["U"] * params["V"], axis=2)
        if self.self_history:
            norm = norm - jnp.sum(diagonal ** 2) + jnp.sum(params["S"] ** 2)
        return .5 * self.ridge * norm

    def _validate_params(self, params, n):
        shapes = {"b": (n,)}
        if self.self_history:
            shapes["S"] = (self.n_basis, n)
        if self.rank is None:
            shapes["W"] = (self.n_basis, n, n)
        else:
            shapes.update({key: (self.n_basis, n, self.rank) for key in ("U", "V")})
        if set(params) != set(shapes) or any(np.shape(params[k]) != shape for k, shape in shapes.items()):
            raise ValueError(f"parameter shapes must match {shapes}")

    def value_and_grad(self, params, recording):
        """Mean NLL per valid time bin, summed over neurons, plus ridge.

        Omits log-factorials. Finish each chunk's reverse pass before the
        next chunk. This is a full-data pass, not a minibatch update.
        """
        self._validate_params(params, recording.n_neurons)
        n_valid = recording.n_valid(self.history)
        if n_valid == 0:
            raise ValueError("no bins with a complete within-epoch history")
        total = 0.
        grad = {key: np.zeros_like(np.asarray(value)) for key, value in params.items()}
        for batch in recording.batches(self.chunk_size, self.history, np.dtype(self.basis.dtype)):
            if not batch.valid.any():
                continue
            value, part = self._value_grad(params, batch.history, batch.counts, batch.valid)
            total += float(value)
            for key in grad:
                grad[key] += np.asarray(part[key])
        penalty, penalty_grad = self._penalty_grad(params)
        total = total / n_valid + float(penalty)
        grad = {key: value / n_valid + np.asarray(penalty_grad[key]) for key, value in grad.items()}
        if not np.isfinite(total) or any(not np.all(np.isfinite(g)) for g in grad.values()):
            raise FloatingPointError("nonfinite objective/gradient; inspect scaling or initialization")
        return total, grad

    def predict_chunks(self, params, recording):
        """Yield (start, expected_counts, valid_rows); invalid rows are NaN."""
        self._validate_params(params, recording.n_neurons)
        for batch in recording.batches(self.chunk_size, self.history, np.dtype(self.basis.dtype)):
            eta = np.asarray(self._eta(params, batch.history))[:batch.size]
            valid = batch.valid[:batch.size]
            rates = np.exp(np.where(valid[:, None], eta, 0.))
            if not np.all(np.isfinite(rates[valid])):
                raise FloatingPointError("nonfinite predicted counts")
            rates[~valid] = np.nan
            yield batch.start, rates, valid

    def fit(self, recording, initial=None, max_iter=100, tolerance=1e-7, callback=None,
            checkpoint=None, resume=False):
        """L-BFGS with exact streamed gradients; storage follows parameter count.

        Low-rank success is an optimizer diagnostic, not a global optimum
        or evidence of anatomical recovery. Inspect the gradient and trace.
        """
        from .persistence import save_bundle, resume_bundle
        if max_iter < 1 or tolerance <= 0 or not np.isfinite(tolerance):
            raise ValueError("max_iter and tolerance must be positive")
        if resume and checkpoint is None:
            raise ValueError("resume requires a checkpoint path")
        data_hash = recording.fingerprint() if checkpoint else None
        if resume:
            initial, _, _ = resume_bundle(checkpoint, self, recording, "lbfgs")
        if initial is None:
            initial = self.initialize(recording.n_neurons,
                mean_count=np.maximum(recording.mean_counts(self.history), 1e-8))
        self._validate_params(initial, recording.n_neurons)
        flat, unravel = ravel_pytree(initial)
        trace = []
        started = time.perf_counter()

        def objective(vector):
            value, grad = self.value_and_grad(unravel(jnp.asarray(vector)), recording)
            gradient, _ = ravel_pytree(grad)
            record = {"evaluation": len(trace) + 1, "objective": value,
                      "gradient_inf_norm": float(np.max(np.abs(gradient))),
                      "seconds": time.perf_counter() - started}
            trace.append(record)
            if callback is not None:
                callback(record)
            return value, np.asarray(gradient, dtype=float)

        def accepted(vector):
            if checkpoint:
                save_bundle(checkpoint, self, unravel(jnp.asarray(vector)),
                    {"solver": "lbfgs", "data_hash": data_hash,
                     "resume_semantics": "parameters only; L-BFGS history restarts"})

        result = minimize(objective, np.asarray(flat, dtype=float), method="L-BFGS-B", jac=True, callback=accepted,
                          options={"maxiter": max_iter, "ftol": tolerance, "gtol": tolerance, "maxcor": 10})
        accepted(result.x)
        return FitResult({k: np.asarray(v) for k, v in unravel(jnp.asarray(result.x)).items()},
                         bool(result.success), str(result.message), int(result.nit), int(result.nfev),
                         float(result.fun), float(np.max(np.abs(result.jac))),
                         time.perf_counter() - started, trace)
