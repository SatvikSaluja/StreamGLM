"""Fresh-process, three-way comparison on two known recurrent GLM families."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
from .persistence import write_json


def generate(bins, neurons, seed, family):
    rng = np.random.default_rng(seed)
    lag = np.arange(1, 9)
    basis = np.exp(-lag[:, None] / np.array([1.5, 4.]))
    basis /= basis.sum(axis=0)
    if family == 'lowrank':
        u = rng.uniform(.15, .55, (2, neurons, 2))
        v = -rng.uniform(.15, .55, (2, neurons, 2))
        weights = np.einsum('kir,kjr->kij', u, v) * (16 / neurons)
    else:
        # Each basis slice is a scaled permutation: exactly full rank.
        weights = np.zeros((2, neurons, neurons))
        for k in range(2):
            weights[k, np.arange(neurons), rng.permutation(neurons)] = -.8
    intercept = np.full(neurons, np.log(.35))
    filters = np.einsum('hk,kij->hij', basis, weights)
    counts = np.zeros((bins, neurons), dtype=np.int32)
    for t in range(bins):
        h = min(t, len(basis))
        eta = intercept.copy()
        if h:
            eta += np.einsum('hi,hij->j', counts[t-h:t][::-1], filters[:h])
        counts[t] = rng.poisson(np.exp(eta))
    return counts, basis, weights, intercept


def worker(args):
    started = time.perf_counter()
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from .data import Recording
    from .model import StreamingGLM
    from .evaluation import score
    from .reference import materialize_weights
    from .persistence import save_bundle
    with np.load(args.data) as f:
        counts, basis, truth, intercept = (f[k] for k in ('counts', 'basis', 'weights', 'intercept'))
    n = counts.shape[1]; k = basis.shape[1]
    cut = int(.8*len(counts))
    train, test = Recording(counts[:cut]), Recording(counts[cut:])
    mean = train.mean_counts(len(basis))
    model = StreamingGLM(basis, chunk_size=512, rank=2 if args.worker == 'lowrank' else None)
    trace = []
    def progress(row):
        trace.append(row)
        write_json(args.out.with_suffix('.progress.json'), {'pid': os.getpid(), 'updated_unix': time.time(), 'trace': trace})
    if args.worker == 'nemos':
        import nemos as nmo
        from .integration import NeMoSRecordingLoader
        glm = nmo.glm.PopulationGLM(observation_model='Poisson', inverse_link_function=jnp.exp,
            regularizer='UnRegularized', solver_name='SVRG', solver_kwargs={'stepsize': .5, 'tol': 1e-7})
        def convert(coef, b):
            return {'W': np.asarray(coef).reshape(n,k,n).transpose(1,0,2), 'b': np.asarray(b)}
        class Gate(nmo.callbacks.Callback):
            def on_pass_end(self, ctx):
                if (ctx.pass_idx+1) % 10: return
                value, grad = model.value_and_grad(convert(ctx.params.coef, ctx.params.intercept), train)
                norm = max(float(np.max(np.abs(g))) for g in grad.values())
                progress({'pass':ctx.pass_idx+1, 'objective':value, 'gradient_inf_norm':norm})
                if norm <= 1e-5: ctx.request_stop('full-gradient gate')
        glm.stochastic_fit(NeMoSRecordingLoader(train,basis,512), n_passes=args.budget,
            init_params=(jnp.zeros((n*k,n)),jnp.log(jnp.asarray(mean))), callbacks=Gate())
        params = convert(glm.coef_,glm.intercept_)
        message = str(glm.stochastic_fit_summary_)
    else:
        fit = model.fit(train, model.initialize(n, seed=0, mean_count=mean), max_iter=args.budget,
                        tolerance=1e-12, callback=progress, checkpoint=args.out.with_suffix('.checkpoint.npz'))
        params = fit.params; message = fit.message
    value, grad = model.value_and_grad(params,train)
    norm = max(float(np.max(np.abs(g))) for g in grad.values())
    test_score = score(model,params,test,mean)
    estimated = materialize_weights(params)
    ef = np.einsum('hk,kij->hij',basis,estimated)
    tf = np.einsum('hk,kij->hij',basis,truth)
    oracle = StreamingGLM(basis,chunk_size=512)
    oracle_score = score(oracle,{'W':truth,'b':intercept},test,mean)
    result = {'status':'DONE','backend':args.worker,'neurons':n,'bins':len(counts),
        'gradient_inf_norm':norm,'gradient_gate_passed':norm<=1e-5,'objective':value,
        'optimizer_message':message,'test':test_score,'oracle_test':oracle_score,
        'filter_relative_error':float(np.linalg.norm(ef-tf)/np.linalg.norm(tf)),
        'filter_correlation':float(np.corrcoef(ef.ravel(),tf.ravel())[0,1]),
        'parameters':sum(v.size for v in params.values()),
        'seconds':time.perf_counter()-started,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        'budget':args.budget,'cpu_affinity':sorted(os.sched_getaffinity(0))}
    save_bundle(args.out.with_suffix('.npz'),model,params)
    write_json(args.out.with_suffix('.json'),result)


def suite(args):
    args.out.mkdir(parents=True,exist_ok=False)
    write_json(args.out/'protocol.json',{'neurons':args.neurons,'seeds':args.seeds,'bins':args.bins,
        'families':['lowrank','permutation_fullrank'],'arms':['dense','lowrank','nemos'],
        'precision':'float64','regularization':'none','split':'80/20 chronological, history reset',
        'gate':1e-5,'budget':args.budget,'worker_timeout_s':args.timeout,
        'lowrank_rank':2,'lowrank_initializations':1,'nemos_stepsize':.5,
        'upstream_commit':'81c7200a0e66686e98fd1907e7fa11e111a0e66a',
        'limitations':['Exploratory fixed settings, not best-tuned competitors.',
          'Gradient norms differ across parameterizations; low-rank stationarity is not a global optimum.',
          'Families have different coupling structure/strength; compare methods within datasets.',
          'Shared CPU host; seeds are data replicates, not timing replicates.',
          'Do not claim speed superiority from unconverged or timed-out fits.'],
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}})
    rows=[]
    for n in args.neurons:
        for seed in args.seeds:
            for family in ('lowrank','permutation_fullrank'):
                stem=f'{family}_n{n}_s{seed}'
                write_json(args.out/'status.json',{'status':'RUNNING','step':'generate','dataset':stem,'pid':os.getpid(),'updated_unix':time.time()})
                counts,basis,weights,intercept=generate(args.bins,n,seed,family)
                data=args.out/f'{stem}_data.npz'
                np.savez_compressed(data,counts=counts,basis=basis,weights=weights,intercept=intercept)
                # Rotate order across seeds to reduce systematic order effects.
                arms=['dense','lowrank','nemos']; offset=seed%3; arms=arms[offset:]+arms[:offset]
                for arm in arms:
                    out=args.out/f'{stem}_{arm}'
                    cmd=[sys.executable,'-u','-m','streamglm.three_way','--worker',arm,'--data',str(data),'--out',str(out),'--budget',str(args.budget)]
                    begin=time.time()
                    with out.with_suffix('.log').open('w') as log:
                        process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                        while process.poll() is None:
                            write_json(args.out/'status.json',{'status':'RUNNING','dataset':stem,'backend':arm,'pid':os.getpid(),'child_pid':process.pid,'updated_unix':time.time(),'worker_elapsed_s':time.time()-begin,'completed':len(rows)})
                            if time.time()-begin>args.timeout:
                                process.terminate()
                                try: process.wait(timeout=10)
                                except subprocess.TimeoutExpired: process.kill(); process.wait()
                                break
                            time.sleep(5)
                    if out.with_suffix('.json').exists(): result=json.loads(out.with_suffix('.json').read_text())
                    else: result={'status':'FAILED','error':'timeout' if time.time()-begin>args.timeout else 'worker failed','returncode':process.returncode}
                    result.update(family=family,seed=seed,neurons=n,backend=arm)
                    write_json(out.with_suffix('.json'),result); rows.append(result)
                    write_json(args.out/'summary.json',rows)
    write_json(args.out/'status.json',{'status':'DONE','completed':len(rows),'failed':sum(r['status']!='DONE' for r in rows),'updated_unix':time.time()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--worker',choices=['dense','lowrank','nemos'])
    p.add_argument('--data',type=Path)
    p.add_argument('--neurons',type=int,nargs='+',default=[16,64,128])
    p.add_argument('--seeds',type=int,nargs='+',default=[7,11,19])
    p.add_argument('--bins',type=int,default=12000)
    p.add_argument('--budget',type=int,default=1500)
    p.add_argument('--timeout',type=int,default=900)
    a=p.parse_args()
    worker(a) if a.worker else suite(a)

if __name__=='__main__': main()
