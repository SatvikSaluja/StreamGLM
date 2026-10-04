"""Explicit half-open spike binning and portable recording directories."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
from .data import Recording
from .persistence import write_json


def load_recording(folder):
    folder = Path(folder)
    metadata = json.loads((folder/"recording.json").read_text())
    return Recording.from_npy(folder/"counts.npy", metadata["epoch_lengths"],
        bin_s=metadata["bin_s"], unit_ids=metadata["unit_ids"])


def _epochs(epochs, bin_s):
    epochs = np.asarray(epochs, dtype=float)
    if (epochs.ndim != 2 or epochs.shape[1] != 2 or len(epochs) == 0
            or not np.all(np.isfinite(epochs)) or np.any(epochs[:, 1] <= epochs[:, 0])
            or np.any(epochs[1:, 0] < epochs[:-1, 1]) or not np.isfinite(bin_s) or bin_s <= 0):
        raise ValueError("epochs must be sorted disjoint finite intervals; bin_s positive")
    lengths = (epochs[:, 1]-epochs[:, 0])/bin_s
    if np.any(lengths < 1) or not np.allclose(lengths, np.rint(lengths), atol=1e-7, rtol=0):
        raise ValueError("each epoch duration must be an integer number of bins")
    return epochs, np.rint(lengths).astype(int)


def from_spike_times(items, epochs, bin_s, out, unit_ids=None, provenance=None):
    """Write a disk-backed recording from one sorted spike train per unit.

    `items` may be a generator. Memory holds one unit's spike train at a time.
    Epochs are half-open [start, stop); incomplete final bins are rejected.
    Existing output directories are never overwritten.
    """
    epochs, lengths = _epochs(epochs, bin_s)
    if unit_ids is None:
        if not hasattr(items, "__len__"):
            raise ValueError("unit_ids are required for an iterator")
        unit_ids = list(range(len(items)))
    unit_ids = [v.item() if isinstance(v, np.generic) else v for v in unit_ids]
    if not unit_ids or len(set(unit_ids)) != len(unit_ids):
        raise ValueError("unit IDs must be nonempty and unique")
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".streamglm-", dir=out.parent))
    try:
        counts = np.lib.format.open_memmap(temp/"counts.npy", mode="w+", dtype=np.int32,
                                          shape=(int(lengths.sum()), len(unit_ids)))
        for start in range(0, len(counts), 8192):
            counts[start:start+8192] = 0
        seen = 0
        for col, spikes in enumerate(items):
            if col >= len(unit_ids):
                raise ValueError("more spike trains than unit IDs")
            seen += 1
            spikes = np.asarray(spikes, dtype=float)
            if spikes.ndim != 1 or not np.all(np.isfinite(spikes)) or np.any(np.diff(spikes) < 0):
                raise ValueError("spike trains must be sorted and finite")
            offset = 0
            for (a, b), length in zip(epochs, lengths):
                left, right = np.searchsorted(spikes, [a, b], side="left")
                relative = (spikes[left:right]-a)/bin_s
                # Snap only machine-roundoff-scale discrepancies at exact edges.
                rounded = np.rint(relative)
                roundoff = 8*np.finfo(float).eps*np.maximum(1, (np.abs(spikes[left:right])+abs(a))/bin_s)
                relative = np.where(np.abs(relative-rounded) <= roundoff, rounded, relative)
                indices = np.floor(relative).astype(np.int64)
                # A timestamp strictly inside the interval may round to its end.
                indices = np.minimum(indices, length-1)
                hist = np.bincount(indices, minlength=length)
                if hist.max(initial=0) > np.iinfo(np.int32).max:
                    raise OverflowError("spike counts exceed int32")
                counts[offset:offset+length, col] = hist
                offset += length
        if seen != len(unit_ids):
            raise ValueError("fewer spike trains than unit IDs")
        counts.flush(); del counts
        write_json(temp/"recording.json", {"format": 1, "binning_version": 2, "epoch_lengths": lengths.tolist(),
            "epochs_seconds": epochs.tolist(), "bin_s": bin_s, "unit_ids": unit_ids,
            "interval_convention": "half-open [start, stop)", "provenance": provenance or {}})
        os.replace(temp, out)
    except BaseException:
        shutil.rmtree(temp)
        raise
    return load_recording(out)


def from_pynapple(group, epochs, bin_s, out):
    """Convert a TsGroup; intervals and spike timestamps are interpreted in seconds."""
    intervals = epochs.values if hasattr(epochs, "values") else epochs
    keys = list(group.keys())
    return from_spike_times((np.asarray(group[key].index) for key in keys), intervals,
                           bin_s, out, unit_ids=keys, provenance={"adapter": "pynapple.TsGroup"})


def from_nwb(path, epochs, bin_s, out, unit_ids=None):
    """Convert NWB Units lazily, reading a single unit's spikes at a time.

    Unit selection is explicit. Quality thresholds differ by dataset and
    are not silently invented here. Keep provenance of any filtering.
    """
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(str(path), mode="r", load_namespaces=True) as io:
        nwb = io.read()
        if nwb.units is None:
            raise ValueError("NWB file has no Units table")
        ids = list(np.asarray(nwb.units.id[:]))
        wanted = ids if unit_ids is None else list(unit_ids)
        indices = [ids.index(key) for key in wanted]
        trains = (np.asarray(nwb.units["spike_times"][i]) for i in indices)
        return from_spike_times(trains, epochs, bin_s, out, wanted,
            {"adapter": "NWB Units", "source": str(Path(path).resolve()),
             "identifier": nwb.identifier, "unit_selection": "explicit IDs" if unit_ids is not None else "all units"})


def from_hnn_cache(path, out):
    """Extract NPY members of a compressed HNN NPZ without loading all spikes."""
    import zipfile
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".streamglm-hnn-", dir=out.parent))
    try:
        with zipfile.ZipFile(path) as archive:
            with archive.open("spikes.npy") as source, (temp/"counts.npy").open("wb") as target:
                shutil.copyfileobj(source, target, length=1024*1024)
            counts = np.load(temp/"counts.npy", mmap_mode="r")
            def member(name):
                with archive.open(name+".npy") as source:
                    return np.load(source, allow_pickle=False)
            if "segment_lengths.npy" not in archive.namelist():
                raise ValueError("HNN cache needs explicit segment_lengths; do not infer epochs")
            lengths = member("segment_lengths").tolist()
            bin_s = float(member("bin_s"))
            ids = member("gids").tolist()
            Recording(counts, lengths, bin_s, ids)
        write_json(temp/"recording.json", {"format": 1, "epoch_lengths": lengths,
            "bin_s": bin_s, "unit_ids": ids,
            "provenance": {"adapter": "HNN NPZ", "source": str(Path(path).resolve())}})
        os.replace(temp, out)
    except BaseException:
        shutil.rmtree(temp)
        raise
    return load_recording(out)
