"""Small unrestricted fit comparison with the installed NeMoS version."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from .data import Recording
from .model import StreamingGLM
from .persistence import write_json
from .synthetic import fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/v2/upstream_comparison.json"))
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    import nemos as nmo
    counts, basis, _ = fixture(4000, 6, seed=7)
    features = np.asarray(nmo.convolve.create_convolutional_predictor(basis, counts, shift=True))
    valid = np.isfinite(features).all(axis=(1, 2))
    design = features[valid].reshape(valid.sum(), -1)
    kwargs = {"solver_name": "LBFGS", "regularizer": "UnRegularized",
              "solver_kwargs": {"maxiter": 1000, "tol": 1e-10}}
    if "inverse_link_function" in inspect.signature(nmo.glm.PopulationGLM).parameters:
        kwargs.update(observation_model="Poisson", inverse_link_function=jnp.exp)
    else:
        kwargs["observation_model"] = nmo.observation_models.PoissonObservations(inverse_link_function=jnp.exp)
    glm = nmo.glm.PopulationGLM(**kwargs)
    started = time.perf_counter(); glm.fit(design, counts[valid]); nemos_seconds = time.perf_counter()-started
    model = StreamingGLM(basis, chunk_size=256)
    fitted = model.fit(Recording(counts), max_iter=1000, tolerance=1e-12)
    expected = np.asarray(glm.predict(design))
    actual = np.concatenate([p for _, p, _ in model.predict_chunks(fitted.params, Recording(counts))])[valid]
    nemos_params = {"b": np.asarray(glm.intercept_),
                    "W": np.asarray(glm.coef_).reshape(6, 2, 6).transpose(1, 0, 2)}
    nemos_value, grad = model.value_and_grad(nemos_params, Recording(counts))
    info = getattr(glm, "optim_info_", None)
    source = Path(inspect.getfile(nmo.glm.PopulationGLM))
    error = float(np.max(np.abs(actual-expected)))
    result = {"nemos_version": importlib.metadata.version("nemos"),
        "nemos_glm_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "has_stochastic_fit": hasattr(glm, "stochastic_fit"),
        "has_batching_module": hasattr(nmo, "batching"),
        "shape": list(counts.shape), "dtype": "float64", "regularizer": "none",
        "nemos_seconds": nemos_seconds, "streamglm_seconds": fitted.seconds,
        "streamglm_converged": fitted.converged,
        "nemos_converged": bool(info.converged) if info is not None and hasattr(info, "converged") else None,
        "nemos_objective": nemos_value, "streamglm_objective": fitted.objective,
        "objective_absolute_difference": abs(nemos_value-fitted.objective),
        "max_prediction_absolute_difference": error,
        "nemos_gradient_inf_norm": max(float(np.max(np.abs(v))) for v in grad.values()),
        "streamglm_gradient_inf_norm": fitted.gradient_inf_norm,
        "note": "Small dense unregularized fit. Not a comparison against newer upstream streaming support or a large-model speed claim."}
    write_json(args.out, result)
    print(json.dumps(result, indent=2))
    if not fitted.converged or error > 1e-3:
        raise RuntimeError("fit equivalence outside tolerance; inspect saved diagnostics")


if __name__ == "__main__":
    main()
