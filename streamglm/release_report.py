"""Build the v0.3 report from completed, immutable experiment records."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .persistence import write_json


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path('artifacts/v4')
    real = []
    for n in (100, 195):
        directory = root/f'allen_visp_{n}_300s'
        workflow = json.loads((directory/'workflow/report.json').read_text())
        benchmark = json.loads((directory/'benchmark.json').read_text())
        provenance = json.loads((Path('cache')/f'allen_visp_{n}_300s/dataset.json').read_text())
        write_json(directory/'dataset.json', provenance)
        gain = workflow['test_gain_over_self_history_nats_per_bin']
        real.append({'requested_units': n, 'retained_units': workflow['retained_neurons'],
            'selected_rank': workflow['selection']['selected']['rank'],
            'all_optimizer_converged': all(x['converged'] for x in workflow['selection']['candidates']),
            'test_gain_over_self_history_nats_per_bin': gain,
            'test_gain_over_self_history_bits_per_spike': gain*workflow['test']['valid_bins']/(workflow['test']['spikes']*np.log(2)),
            'seconds': benchmark['total_seconds'], 'peak_rss_gib': benchmark['peak_rss_gib'],
            'selected_gradient_inf_norm': workflow['selection']['selected']['gradient_inf_norm']})
    null = json.loads((root/'null_1000_1h/report.json').read_text())
    upstream = json.loads((root/'upstream_streaming_1500/report.json').read_text())
    trace = json.loads((root/'upstream_streaming_1500/progress.json').read_text())
    limitations = [
        'The two real-data subsets overlap within one VISp recording, not independent biological replications.',
        'Quality labels come from the supplied workshop extract; no waveform-level quality reanalysis.',
        'Real-data coupling has no stimulus/behavior adjustment and does not establish synaptic or causal connectivity.',
        'The 1,000-unit hour-long case is independent synthetic Poisson at 10 ms bins, not real Neuropixels at 1 ms.',
        'Optimizer relative-objective stopping does not certify a global optimum or a small gradient; diagnostics retained.',
        'The current-NeMoS exercise validates loader interoperability and accuracy, not a performance advantage over upstream streaming.',
        'CPU timing was measured on a shared host while HNN and other validation jobs ran.',
        'No remote CI, upstream acceptance, package publication, or 1,000-unit hour-long real-data fit is claimed.'
    ]
    record = {'real_data': real, 'synthetic_stress': null,
              'current_upstream': upstream, 'limitations': limitations,
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in Path('streamglm').glob('*.py')}}
    write_json(root/'release_report.json', record)
    lines = ['# StreamGLM 0.3 measured validation', '',
        'Real Neuropixels predictive tests, an explicitly synthetic scale test, '
        'and interoperability with pinned current NeMoS. These establish software '
        'capabilities under the recorded conditions; they do not establish anatomical recovery.', '',
        '## Allen Visual Coding Neuropixels', '',
        '300 seconds at 5 ms bins; provider quality==good and VISp; ascending unit IDs. '
        'Chronological 60/20/20 split, training-only activity filtering, ranks 0/2/5 '
        'and two starts for nonzero ranks. Selection uses validation data.', '',
        '| Retained / requested | Rank | Gain over self-history (nats/bin) | Gain (bits/spike) | Peak GiB | Workflow seconds |',
        '|---:|---:|---:|---:|---:|---:|']
    for r in real:
        lines.append(f"| {r['retained_units']}/{r['requested_units']} | {r['selected_rank']} | "
                     f"{r['test_gain_over_self_history_nats_per_bin']:.6g} | "
                     f"{r['test_gain_over_self_history_bits_per_spike']:.6g} | "
                     f"{r['peak_rss_gib']:.3f} | {r['seconds']:.1f} |")
    lines += ['', '## Synthetic scale test', '',
        f"{null['neurons']} independent Poisson units, {null['recording_seconds']:g} seconds, "
        f"{null['bin_s']*1000:g} ms bins. Peak process RSS {null['peak_rss_gib']:.3f} GiB; "
        f"workflow {null['seconds']:.1f} seconds. Hypothetical full-recording design: "
        f"{null['hypothetical_full_design_gib']:.2f} GiB. The design size is arithmetic, not an OOM measurement.", '',
        f"Rank-5 test gain over self-history: {null['gain_over_self_history_nats_per_bin']:.6g} nats/bin. "
        'No coupling exists in this generator; do not treat this as recovery validation.', '',
        '## Current NeMoS interoperability', '',
        f"Pinned commit `{upstream['upstream_commit']}`. `NeMoSRecordingLoader` retains "
        'epoch boundaries and short tails; fixed features are tested against independent '
        'whole-epoch NeMoS convolutions. Current upstream already supports this kind of streaming.', '',
        f"SVRG reached gradient norm {upstream['stochastic_gradient_inf_norm']:.3g}; "
        f"objective difference from unrestricted StreamGLM L-BFGS: {upstream['objective_difference']:.3g}. "
        'The initial 300-pass run did not meet the gradient threshold; it is retained. '
        'The longer run used the same settings and stopped at the threshold.', '',
        '## Limitations', ''] + ['- '+s for s in limitations] + ['']
    (root/'report.md').write_text('\n'.join(lines))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), constrained_layout=True)
    labels = [f"{r['retained_units']} units" for r in real]
    axes[0].bar(labels, [r['test_gain_over_self_history_bits_per_spike'] for r in real], color='#287b8e')
    axes[0].set(title='Neuropixels predictive gain', ylabel='Bits/spike over self-history')
    axes[0].axhline(0, color='gray', linewidth=.7)
    axes[1].bar(labels+['1,000 synthetic'], [r['peak_rss_gib'] for r in real]+[null['peak_rss_gib']], color=['#287b8e','#287b8e','#b56e20'])
    axes[1].set(title='Whole-workflow peak memory', ylabel='GiB (includes mapped input)')
    axes[2].semilogy([r['pass'] for r in trace], [r['gradient_inf_norm'] for r in trace], color='#287b8e')
    axes[2].axhline(1e-5, color='gray', linestyle='--', label='Accuracy threshold')
    axes[2].set(title='Pinned NeMoS streaming check', xlabel='SVRG passes', ylabel='Full-gradient infinity norm')
    axes[2].legend(fontsize=8)
    fig.suptitle('Real data: 5 min, overlapping subsets. Synthetic: 1 hour at 10 ms. Shared CPU host.', fontsize=10)
    fig.savefig(root/'overview.png', dpi=160)
    fig.savefig(root/'overview.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
