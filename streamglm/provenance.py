"""Record source/artifact hashes and package versions for a local result set."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from pathlib import Path
import platform
from .persistence import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/v2/provenance.json"))
    args = parser.parse_args()
    source = list(Path("streamglm").glob("*.py")) + list(Path("tests_streamglm").glob("*.py"))
    source += [Path("pyproject.toml"), Path("constraints-validated.txt")]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source)}
    artifact_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(args.out.parent.rglob("*")) if p.is_file() and p.resolve() != args.out.resolve()}
    versions = {}
    for name in ("streamglm", "jax", "jaxlib", "numpy", "scipy", "pynapple", "pynwb", "nemos", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    write_json(args.out, {"created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": versions, "source_sha256": hashes, "artifact_sha256": artifact_hashes,
        "note": "Hashes of final source and preserved artifacts. Initial-binning directory is historical, not the current real-data result."})


if __name__ == "__main__":
    main()
