"""Replayable minibatch Adam with exact epoch-boundary resume.

    Checkpoints include moments and update count, not only parameters.
    Batches are deterministic and sequential; epochs are complete passes.
"""
import time
import jax
import jax.numpy as jnp
import numpy as np
from .model import FitResult
from .persistence import save_bundle, resume_bundle


def fit_adam(model, recording, initial=None, epochs=10, learning_rate=.01,
             tolerance=1e-5, checkpoint=None, resume=False, callback=None):
    if epochs < 1 or learning_rate <= 0 or not np.isfinite(learning_rate):
        raise ValueError("epochs and learning_rate must be positive")
    if recording.n_valid(model.history) == 0:
        raise ValueError("no complete history")
    if resume and not checkpoint:
        raise ValueError("resume requires a checkpoint")
    started = time.perf_counter()
    trace = []
    updates, completed = 0, 0
    data_hash = recording.fingerprint() if checkpoint else None
    if resume:
        initial, meta, state = resume_bundle(checkpoint, model, recording, "adam")
        if meta["learning_rate"] != learning_rate:
            raise ValueError("learning rate differs from saved optimizer")
        updates, completed = int(meta["updates"]), int(meta["epochs"])
        m = {k: jnp.asarray(state["m_"+k]) for k in initial}
        v = {k: jnp.asarray(state["v_"+k]) for k in initial}
    else:
        if initial is None:
            initial = model.initialize(recording.n_neurons,
                mean_count=np.maximum(recording.mean_counts(model.history), 1e-8))
        m = jax.tree.map(jnp.zeros_like, initial)
        v = jax.tree.map(jnp.zeros_like, initial)
    model._validate_params(initial, recording.n_neurons)
    params = jax.tree.map(jnp.asarray, initial)

    @jax.jit
    def update(params, m, v, t, x, y, valid):
        loss, grad = model._value_grad(params, x, y, valid)
        penalty, penalty_grad = model._penalty_grad(params)
        size = jnp.sum(valid)
        grad = jax.tree.map(lambda g, r: g/size+r, grad, penalty_grad)
        m = jax.tree.map(lambda a, g: .9*a+.1*g, m, grad)
        v = jax.tree.map(lambda a, g: .999*a+.001*g*g, v, grad)
        params = jax.tree.map(lambda p, a, b: p-learning_rate*(a/(1-.9**t))/(jnp.sqrt(b/(1-.999**t))+1e-8), params, m, v)
        return params, m, v, loss/size+penalty

    converged = False
    for epoch in range(completed, completed+epochs):
        for batch in recording.batches(model.chunk_size, model.history, np.dtype(model.basis.dtype)):
            if not batch.valid.any():
                continue
            updates += 1
            params, m, v, loss = update(params, m, v, updates, batch.history, batch.counts, batch.valid)
            if not np.isfinite(float(loss)):
                raise FloatingPointError("nonfinite Adam batch; reduce learning rate or rescale")
        value, gradient = model.value_and_grad(params, recording)
        norm = max(float(np.max(np.abs(g), initial=0)) for g in gradient.values())
        converged = norm <= tolerance
        entry = {"epoch": epoch+1, "updates": updates, "objective": value,
                 "gradient_inf_norm": norm, "seconds": time.perf_counter()-started}
        trace.append(entry)
        if checkpoint:
            state = {**{"m_"+k: v for k, v in m.items()}, **{"v_"+k: a for k, a in v.items()}}
            save_bundle(checkpoint, model, params, {"solver": "adam", "data_hash": data_hash,
                "learning_rate": learning_rate, "updates": updates, "epochs": epoch+1,
                "resume_semantics": "exact continuation at completed epoch boundary"}, state)
        if callback:
            callback(entry)
        if converged:
            break
    return FitResult(jax.tree.map(np.asarray, params), converged,
        "full-gradient tolerance reached" if converged else "epoch limit reached",
        len(trace), updates, value, norm, time.perf_counter()-started, trace)
