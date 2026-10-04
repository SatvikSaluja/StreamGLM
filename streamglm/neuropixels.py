"""Reproduce a predictive Allen Visual Coding Neuropixels demonstration.

Uses the checksum-pinned dataset distributed by the CCN February 2026 workshop.
This is a VISp subset, not brain-wide data or an anatomical recovery experiment.
"""
import argparse
import hashlib
from pathlib import Path
import urllib.request
import zipfile

import numpy as np

from .adapters import from_spike_times
from .persistence import write_json

URL = 'https://osf.io/download/6c7rj'
SHA256 = '6041e46a370e9ad09c13a22fac493f237cd8153c598a052f0552b58c6a42c293'
SOURCE = 'https://github.com/flatironinstitute/ccn-software-feb-2026/blob/main/src/workshop_utils/fetch.py'


def prepare(out, cache, neurons=100, start=1285., duration=300., bin_s=.005):
    import pynapple as nap
    out, cache = Path(out), Path(cache)
    if out.exists():
        raise FileExistsError(out)
    if neurons < 1:
        raise ValueError('neurons must be positive')
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache/'visual_coding_data.zip'
    if not archive.exists():
        temporary = archive.with_suffix('.part')
        with urllib.request.urlopen(URL, timeout=60) as src, temporary.open('wb') as dst:
            while block := src.read(2**20):
                dst.write(block)
        temporary.replace(archive)
    sha = hashlib.sha256()
    with archive.open('rb') as file:
        while block := file.read(2**20):
            sha.update(block)
    digest = sha.hexdigest()
    if digest != SHA256:
        raise ValueError('workshop archive checksum mismatch')
    with zipfile.ZipFile(archive) as zipped:
        for name in ('units.npz', 'flashes.npz'):
            with zipped.open(name) as src, (cache/name).open('wb') as dst:
                while block := src.read(2**20):
                    dst.write(block)
    units = nap.load_file(cache/'units.npz')
    quality = units.metadata
    ids = sorted(quality.index[(quality['quality']=='good') & (quality['brain_area']=='VISp')].tolist())
    if len(ids) < neurons:
        raise ValueError(f'only {len(ids)} good VISp units available; cannot request {neurons}')
    ids = ids[:neurons]
    support = units.time_support.values
    if not any(a <= start and start+duration <= b for a, b in support):
        raise ValueError('requested window outside recording support')
    provenance = {'dataset': 'Allen Visual Coding Neuropixels; CCN workshop VISp subset',
        'url': URL, 'archive_sha256': digest, 'registry_source': SOURCE,
        'selection': 'quality==good and brain_area==VISp; ascending unit ID; first requested N',
        'quality_note': 'Use supplied full-recording quality labels; no additional waveform QC available in this extract.',
        'requested_neurons': neurons, 'available_good_visp': int(sum(quality['quality']=='good')),
        'window': [start, start+duration], 'bin_s': bin_s,
        'interpretation': 'Predictive demonstration during visual stimulation; no stimulus or behavior covariates. Coupling is not causal or anatomical.'}
    recording = from_spike_times((np.asarray(units[key].index) for key in ids),
        [[start, start+duration]], bin_s, out, unit_ids=ids, provenance=provenance)
    write_json(out/'dataset.json', provenance | {'recording_fingerprint': recording.fingerprint()})
    print(f'Prepared {len(recording.counts)} bins x {recording.n_neurons} neurons', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cache', type=Path, default=Path('cache/public'))
    p.add_argument('--neurons', type=int, default=100)
    p.add_argument('--start', type=float, default=1285.)
    p.add_argument('--duration', type=float, default=300.)
    p.add_argument('--bin-s', type=float, default=.005)
    args = p.parse_args()
    prepare(args.out, args.cache, args.neurons, args.start, args.duration, args.bin_s)


if __name__ == '__main__':
    main()
