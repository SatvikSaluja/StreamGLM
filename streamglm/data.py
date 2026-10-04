"""Replayable batches, including read-only NPY memory maps."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np


@dataclass(frozen=True)
class Batch:
    history: np.ndarray
    counts: np.ndarray
    valid: np.ndarray
    start: int
    size: int


class Recording:
    """Counts (time, neuron), partitioned into independent epochs.

    Exclude the first H rows of each epoch: no invented zero prehistory.
    Validate counts in chunks, without a recording-sized temporary array.
    """

    def __init__(self, counts, epoch_lengths=None, bin_s=1., unit_ids=None):
        if len(counts.shape) != 2 or min(counts.shape) < 1:
            raise ValueError("counts must have nonempty shape (time, neuron)")
        lengths = np.asarray([len(counts)] if epoch_lengths is None else epoch_lengths)
        if (lengths.ndim != 1 or not np.issubdtype(lengths.dtype, np.integer)
                or np.any(lengths <= 0) or int(lengths.sum()) != len(counts)):
            raise ValueError("positive integer epoch lengths must partition counts")
        self.counts = counts
        self.epoch_lengths = tuple(map(int, lengths))
        self.n_neurons = counts.shape[1]
        if not np.isfinite(bin_s) or bin_s <= 0:
            raise ValueError("bin_s must be finite and positive")
        self.bin_s = float(bin_s)
        self.unit_ids = list(range(self.n_neurons)) if unit_ids is None else list(unit_ids)
        if len(self.unit_ids) != self.n_neurons or len(set(self.unit_ids)) != self.n_neurons:
            raise ValueError("unit_ids must uniquely identify every column")

    @classmethod
    def from_npy(cls, path, epoch_lengths=None, **kwargs):
        return cls(np.load(path, mmap_mode="r", allow_pickle=False), epoch_lengths, **kwargs)

    def select(self, ranges, columns=None):
        """Lazy subset of absolute row ranges, preserving every epoch boundary."""
        bounds = np.r_[0, np.cumsum(self.epoch_lengths)]
        selected = []
        previous = 0
        for start, stop in ranges:
            if (not isinstance(start, (int, np.integer)) or not isinstance(stop, (int, np.integer))
                    or not (0 <= start < stop <= len(self.counts)) or start < previous):
                raise ValueError("ranges must be sorted, disjoint, and inside recording")
            previous = stop
            for left, right in zip(bounds[:-1], bounds[1:]):
                a, b = max(start, left), min(stop, right)
                if b > a:
                    selected.append((int(a), int(b)))
        columns = np.arange(self.n_neurons) if columns is None else np.asarray(columns)
        if (columns.ndim != 1 or not np.issubdtype(columns.dtype, np.integer)
                or len(columns) == 0 or len(set(columns)) != len(columns)
                or np.any(columns < 0) or np.any(columns >= self.n_neurons)):
            raise ValueError("columns must be unique valid integer indices")
        view = _SegmentView(self.counts, selected, columns)
        return Recording(view, [b-a for a, b in selected], self.bin_s,
                         [self.unit_ids[c] for c in columns])

    def mean_counts(self, history=0, chunk_size=8192):
        total = np.zeros(self.n_neurons)
        offset = 0
        n = self.n_valid(history)
        if n <= 0:
            raise ValueError("no valid bins")
        for length in self.epoch_lengths:
            for start in range(offset + history, offset + length, chunk_size):
                block = np.asarray(self.counts[start:min(start+chunk_size, offset+length)])
                total += block.sum(axis=0)
            offset += length
        return total / n

    def fingerprint(self, chunk_size=8192):
        """Content hash for resume safety. One read pass, without a large copy."""
        digest = hashlib.sha256(json.dumps({"shape": self.counts.shape,
            "epochs": self.epoch_lengths, "bin_s": self.bin_s,
            "unit_ids": self.unit_ids}, sort_keys=True).encode())
        for start in range(0, len(self.counts), chunk_size):
            block = np.asarray(self.counts[start:start+chunk_size], dtype="<f8")
            digest.update(block.tobytes(order="C"))
        return digest.hexdigest()

    def n_valid(self, history):
        return sum(max(0, n - history) for n in self.epoch_lengths)

    def batches(self, chunk_size, history, dtype=np.float64):
        if chunk_size < 1 or history < 1:
            raise ValueError("chunk_size and history must be positive")
        offset = 0
        for length in self.epoch_lengths:
            for local in range(0, length, chunk_size):
                size = min(chunk_size, length - local)
                begin = max(0, local - history)
                block = np.asarray(self.counts[offset + begin:offset + local + size], dtype=dtype)
                if not np.all(np.isfinite(block)) or np.any(block < 0) or np.any(block != np.floor(block)):
                    raise ValueError("counts must be finite, nonnegative integers")
                x = np.zeros((chunk_size + history, self.n_neurons), dtype=dtype)
                prefix = history - (local - begin)
                x[prefix:history + size] = block
                valid = (np.arange(chunk_size) < size) & (local + np.arange(chunk_size) >= history)
                yield Batch(x, x[history:].copy(), valid, offset + local, size)
            offset += length


class _SegmentView:
    """Slice-only array facade; copies at most the requested chunk."""
    def __init__(self, source, ranges, columns):
        self.source, self.ranges, self.columns = source, ranges, columns
        self.bounds = np.r_[0, np.cumsum([b-a for a, b in ranges])]
        self.shape = (int(self.bounds[-1]), len(columns))

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, item):
        if not isinstance(item, slice):
            raise TypeError("only contiguous row slices are supported")
        start, stop, step = item.indices(len(self))
        if step != 1:
            raise ValueError("slice step must be one")
        pieces = []
        for i, (a, b) in enumerate(self.ranges):
            left, right = max(start, self.bounds[i]), min(stop, self.bounds[i+1])
            if right > left:
                pieces.append(np.asarray(self.source[a+left-self.bounds[i]:a+right-self.bounds[i]])[:, self.columns])
        return np.concatenate(pieces, axis=0) if pieces else np.empty((0, self.shape[1]))
