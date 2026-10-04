"""Run the public Mouse32 NWB example. Downloads only when explicitly run."""
import argparse
import json
from pathlib import Path
import jax
from .adapters import from_nwb, load_recording
from .datasets import mouse32, MOUSE32_URL, MOUSE32_SHA256
from .pipeline import run_pipeline, exponential_basis
from .persistence import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("cache/public"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/v2/real_mouse32"))
    parser.add_argument("--max-iter", type=int, default=800)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    source = mouse32(args.cache)
    directory = args.cache/"mouse32_wake_300s_v2"
    # Fixed 300-second window inside the file's documented wake interval.
    # All units enter import; training-only rate filtering occurs in pipeline.
    if directory.exists():
        metadata = json.loads((directory/"recording.json").read_text())
        if (metadata["epochs_seconds"] != [[8813., 9113.]] or metadata["bin_s"] != .005
                or metadata.get("binning_version") != 2):
            raise ValueError("cached example configuration differs")
        data = load_recording(directory)
    else:
        data = from_nwb(source, [[8813., 9113.]], .005, directory)
    result = run_pipeline(data, args.out, exponential_basis(25, 3), ranks=(0, 1, 2),
                          seeds=(0, 1), max_iter=args.max_iter, chunk_size=1024, ridge=.01)
    write_json(args.out/"dataset.json", {"url": MOUSE32_URL, "sha256": MOUSE32_SHA256,
        "source_tutorial": "https://pynapple.org/examples/tutorial_HD_dataset.html",
        "dataset": "Peyrache-2015 Mouse32-140822", "epochs_seconds": [[8813., 9113.]],
        "bin_s": .005, "recording_type": "head-direction recording; not Neuropixels",
        "quality_filter": "training firing rate >= 0.1 Hz; no additional unit-quality metadata filter"})
    print(json.dumps({"status": result["status"], "neurons": result["retained_neurons"],
        "selected_rank": result["selection"]["selected"]["rank"], "test": result["test"],
        "gain_over_self_history": result["test_gain_over_self_history_nats_per_bin"]}, indent=2))


if __name__ == "__main__":
    main()
