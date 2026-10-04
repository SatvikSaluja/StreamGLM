"""Opt-in, checksum-verified download of the public Pynapple example."""
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request

MOUSE32_URL = "https://osf.io/download/jb2gd/"
MOUSE32_SHA256 = "1a919a033305b8f58b5c3e217577256183b14ed5b436d9c70989dee6dafe0f35"


def mouse32(folder):
    """Fetch the 36.6 MB Peyrache-2015 Mouse32 NWB example, only on request.

    Source and scientific attribution:
    https://pynapple.org/examples/tutorial_HD_dataset.html
    This recording is head-direction data, not Neuropixels or brain-wide data.
    """
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    path = folder/"Mouse32-140822.nwb"
    def digest(path):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024*1024), b""):
                h.update(block)
        return h.hexdigest()
    if path.exists():
        if digest(path) != MOUSE32_SHA256:
            raise ValueError("existing dataset checksum differs; preserve and inspect it")
        return path
    with tempfile.NamedTemporaryFile(dir=folder, delete=False) as f:
        temporary = Path(f.name)
        try:
            with urllib.request.urlopen(MOUSE32_URL, timeout=60) as response:
                size = 0
                while block := response.read(1024*1024):
                    size += len(block)
                    if size > 50_000_000:
                        raise ValueError("download exceeds expected size limit")
                    f.write(block)
            f.flush()
            if digest(temporary) != MOUSE32_SHA256:
                raise ValueError("download checksum differs from pinned dataset")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)
    return path
