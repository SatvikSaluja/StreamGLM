"""Atomic, pickle-free model and optimizer files."""
import json
import os
from pathlib import Path
import tempfile
import numpy as np


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        try:
            f.write(text); f.flush(); os.fsync(f.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def save_bundle(path, model, params, metadata=None, state=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {"rank": model.rank, "chunk_size": model.chunk_size, "ridge": model.ridge,
              "self_history": model.self_history, "dtype": str(model.basis.dtype)}
    payload = {"basis": np.asarray(model.basis),
               "metadata": np.array(json.dumps({"format": 1, "config": config, **(metadata or {})}, allow_nan=False))}
    payload.update({"param_"+k: np.asarray(v) for k, v in params.items()})
    payload.update({"state_"+k: np.asarray(v) for k, v in (state or {}).items()})
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        try:
            np.savez(f, **payload); f.flush(); os.fsync(f.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def load_bundle(path):
    from .model import StreamingGLM
    import jax
    with np.load(path, allow_pickle=False) as file:
        meta = json.loads(str(file["metadata"]))
        if meta.get("format") != 1:
            raise ValueError("unsupported model format")
        config = dict(meta["config"])
        dtype = config.pop("dtype")
        if dtype == "float64" and not jax.config.x64_enabled:
            raise ValueError("enable jax_enable_x64 before loading a float64 model")
        model = StreamingGLM(file["basis"], dtype=dtype, **config)
        params = {k[6:]: file[k].copy() for k in file.files if k.startswith("param_")}
        state = {k[6:]: file[k].copy() for k in file.files if k.startswith("state_")}
    model._validate_params(params, len(params["b"]))
    return model, params, meta, state


def resume_bundle(path, model, recording, solver):
    restored, params, meta, state = load_bundle(path)
    if (meta.get("solver") != solver or meta.get("data_hash") != recording.fingerprint()
            or meta["config"] != {"rank": model.rank, "chunk_size": model.chunk_size,
                "ridge": model.ridge, "self_history": model.self_history, "dtype": str(model.basis.dtype)}
            or not np.array_equal(restored.basis, model.basis)):
        raise ValueError("checkpoint data, solver or model configuration does not match")
    return params, meta, state
