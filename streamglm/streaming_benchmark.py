"""Fresh-process streamed-vs-streamed fits; all solver settings remain visible."""
import argparse,json,os,subprocess,sys,time,hashlib
from pathlib import Path
from .persistence import write_json

def run(root):
    import numpy as np
    from .three_way import generate
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    protocol={'neurons':64,'seeds':[31,47],'bins':[24000,96000,384000],'families':['lowrank','permutation_fullrank'],
        'backends':['StreamGLM exact chunked L-BFGS','NeMoS SVRG loader step .1','NeMoS SVRG loader step .5'],
        'chunk':512,'ridge':'lowrank .1; fullrank .01','precision':'float64','initialization':'zero coupling, log training mean intercept',
        'splits':'60/20/20 chronological; only training and validation used; no test selection',
        'gradient_gate':1e-5,'iteration_or_pass_budget':1500,'timeout_s':3600,
        'timing':'fresh worker wall time includes imports, preparation, independent checks, fitting and validation; model seconds also retained',
        'memory':'RSS includes resident spike counts; neither backend constructs full design; not an out-of-core input comparison',
        'upstream_commit':'81c7200a0e66686e98fd1907e7fa11e111a0e66a',
        'scope':'all configurations reported, comparisons require both pass independent gradient gate; do not select by held-out test or discard failed settings'}
    cfg=root/'protocol.json'
    if cfg.exists() and json.load(open(cfg))!=protocol:raise ValueError('protocol differs')
    write_json(cfg,protocol);records=[]
    for seed in protocol['seeds']:
      for family in protocol['families']:
        folder=root/f'{family}_s{seed}';folder.mkdir(exist_ok=True);full=folder/'full.npz'
        if not full.exists():
            write_json(root/'status.json',{'status':'GENERATING','seed':seed,'family':family,'pid':os.getpid(),'updated_unix':time.time()})
            counts,basis,_,_=generate(384000,64,seed,family);np.savez_compressed(folder/'full.tmp.npz',counts=counts,basis=basis);(folder/'full.tmp.npz').replace(full)
        with np.load(full) as z:counts,basis=z['counts'],z['basis']
        for bins in protocol['bins']:
          data=folder/f'data{bins}.npz'
          if not data.exists():
            temp=folder/f'data{bins}.tmp.npz';np.savez_compressed(temp,counts=counts[:bins],basis=basis);temp.replace(data)
          for label,backend,step in [('streamglm','dense',None),('nemos01','nemos',.1),('nemos05','nemos',.5)]:
            out=folder/f't{bins}_{label}.json';config=folder/f't{bins}_{label}.config.json'
            if not out.exists():
                write_json(config,{'backend':backend,'rank':None,'ridge':.1 if family=='lowrank' else .01,'stepsize':step,'budget':1500})
                command=[sys.executable,'-u','-m','streamglm.tuned_comparison','--worker','--config',str(config),'--data',str(data),'--out',str(out)]
                started=time.monotonic()
                with out.with_suffix('.log').open('w') as log:
                    p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                    while p.poll() is None:
                        write_json(root/'status.json',{'status':'RUNNING','pid':os.getpid(),'child_pid':p.pid,'bins':bins,'family':family,'seed':seed,'backend':label,'seconds':time.monotonic()-started,'updated_unix':time.time()})
                        if time.monotonic()-started>3600:
                            p.terminate()
                            try:p.wait(timeout=10)
                            except subprocess.TimeoutExpired:p.kill();p.wait()
                            break
                        time.sleep(5)
                if out.exists():a=json.load(open(out))
                else:a={'status':'FAILED','returncode':p.returncode,'reason':'timeout' if time.monotonic()-started>=3600 else 'worker exception'}
                a['worker_wall_seconds']=time.monotonic()-started;write_json(out,a)
            records.append({'bins':bins,'seed':seed,'family':family,'label':label,**json.load(open(out))});write_json(root/'results.json',records)
    lines=['# Streamed-vs-streamed benchmark','','All configurations retained. Interpret timings only where convergence passes. See protocol.json.','','| Bins | Seed | Family | Backend | Gate | Wall s | Peak GiB |','|---|---|---|---|---|---:|---:|']
    for a in records:lines.append(f"| {a['bins']} | {a['seed']} | {a['family']} | {a['label']} | {a.get('gradient_gate_passed',False)} | {a.get('worker_wall_seconds',float('nan')):.2f} | {a.get('peak_rss_gib',float('nan')):.3f} |")
    (root/'report.md').write_text('\n'.join(lines)+'\n');write_json(root/'status.json',{'status':'DONE','completed':len(records),'execution_failures':sum(a['status']!='DONE' for a in records),'gradient_passes':sum(a.get('gradient_gate_passed',False) for a in records),'updated_unix':time.time()})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
