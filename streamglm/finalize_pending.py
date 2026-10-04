"""Wait for tuning completion, then generate a transparent local report."""
import json,os,time
from pathlib import Path
from .persistence import write_json


def report(root):
 rows=json.loads((root/'selected.json').read_text())
 candidates=json.loads((root/'candidates.json').read_text())
 lines=['# Tuned StreamGLM / NeMoS comparison','',
 'Fixed protocol: 60/20/20 chronological splits, fresh seeds, validation-only selection among gradient-gate-passing candidates. Failed settings remain in candidates.json. Two seeds are exploratory, not a significance certification.','',
 '| Dataset | Model | Rank | Ridge | Step | Test bits/spike | Filter correlation | Relative filter error | Selected fit seconds | Tuning seconds | Peak GiB |',
 '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for row in rows:
  if row['status']!='SELECTED' or 'test' not in row.get('evaluation',{}):
   lines.append(f"| {row['dataset']} | {row['backend']} | unavailable: {row['status']} | | | | | | | | |")
   continue
  s=row['selected'];e=row['evaluation']
  lines.append(f"| {row['dataset']} | {row['backend']} | {s['rank']} | {s['ridge']} | {s['stepsize']} | {e['test']['gain_bits_per_spike']:.6g} | {e['filter_correlation']:.4f} | {e['filter_relative_error']:.4f} | {s['seconds']:.1f} | {row['tuning_seconds']:.1f} | {s['peak_rss_gib']:.3f} |")
 lines.extend(['',f"Candidate outcomes: {len(candidates)}; execution failures: {sum(r['status']!='DONE' for r in candidates)}; gradient passes: {sum(r.get('gradient_gate_passed',False) for r in candidates)}.",'',
 'Bits/spike are gains over training constant-rate predictions. Correlations are Pearson correlations of reconstructed filters, including diagonals. Low-rank gradient gates use factor coordinates. Timing includes compilation and differs across solver/search configurations; shared host. Do not interpret failures as generic library inferiority. Rank/penalty winners at search boundaries require new confirmation data before further tuning.'])
 (root/'report.md').write_text('\n'.join(lines)+'\n')


def main():
 root=Path('artifacts/v6/tuned');status=Path('artifacts/v6/finalizer_status.json')
 while True:
  d=json.loads((root/'status.json').read_text())
  if d['status']=='DONE':
   report(root);write_json(status,{'status':'DONE','report':str(root/'report.md'),'updated_unix':time.time()});return
  if d['status']=='FAILED' or not Path(f"/proc/{d.get('pid')}/cmdline").exists():
   write_json(status,{'status':'BLOCKED','reason':'tuning stopped before completion; results retained','updated_unix':time.time()});return
  write_json(status,{'status':'WAITING','pid':os.getpid(),'waiting_for':d.get('pid'),'updated_unix':time.time()})
  time.sleep(30)

if __name__=='__main__':main()
