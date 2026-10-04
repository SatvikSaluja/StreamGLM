"""Validation-selected ridge/rank/stepsize comparison; sealed held-out scoring."""
import argparse
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
from .persistence import write_json


def select_candidate(rows):
    eligible=[r for r in rows if r.get('status')=='DONE' and r.get('gradient_gate_passed')
              and np.isfinite(r.get('validation',{}).get('log_likelihood_per_bin',np.nan))]
    return max(eligible,key=lambda r:r['validation']['log_likelihood_per_bin']) if eligible else None


def worker(a):
    started=time.perf_counter()
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from .data import Recording
    from .evaluation import score
    from .model import StreamingGLM
    from .persistence import save_bundle,load_bundle
    from .integration import NeMoSRecordingLoader
    config=json.loads(a.config.read_text())
    with np.load(a.data) as f:
        counts,basis=f['counts'],f['basis']
    n=counts.shape[1];k=basis.shape[1];h=len(basis)
    train=Recording(counts[:int(.6*len(counts))])
    mean=train.mean_counts(h)
    if a.evaluate:
        from .reference import materialize_weights
        model,params,_,_=load_bundle(Path(config['model']))
        test=Recording(counts[int(.8*len(counts)):])
        with np.load(a.truth) as f: weights,intercept=f['weights'],f['intercept']
        estimated=np.einsum('hk,kij->hij',basis,materialize_weights(params))
        true=np.einsum('hk,kij->hij',basis,weights)
        oracle=StreamingGLM(basis,chunk_size=512)
        write_json(a.out,{'test':score(model,params,test,mean),
            'oracle_test':score(oracle,{'W':weights,'b':intercept},test,mean),
            'filter_relative_error':float(np.linalg.norm(estimated-true)/np.linalg.norm(true)),
            'filter_correlation':float(np.corrcoef(estimated.ravel(),true.ravel())[0,1])})
        return
    # Candidate workers never score test data or read ground truth.
    validation=Recording(counts[int(.6*len(counts)):int(.8*len(counts))])
    model=StreamingGLM(basis,chunk_size=512,rank=config['rank'],ridge=config['ridge'])
    trace=[]
    def progress(row):
        trace.append(row)
        write_json(a.out.with_suffix('.progress.json'),{'pid':os.getpid(),'updated_unix':time.time(),'trace':trace})
    parity=None
    if config['backend']=='nemos':
        import nemos as nmo
        from nemos.glm.params import GLMParams
        glm=nmo.glm.PopulationGLM(observation_model='Poisson',inverse_link_function=jnp.exp,
            regularizer='Ridge',regularizer_strength=config['ridge'],solver_name='SVRG',
            solver_kwargs={'stepsize':config['stepsize'],'tol':1e-7})
        def convert(coef,b):return {'W':np.asarray(coef).reshape(n,k,n).transpose(1,0,2),'b':np.asarray(b)}
        loader=NeMoSRecordingLoader(train,basis,512)
        # Independent nonzero-parameter value/gradient check, including ridge.
        x,y=next(iter(loader));rng=np.random.default_rng(918)
        coef=jnp.asarray(rng.normal(0,.03,(n*k,n)));b=jnp.log(jnp.asarray(mean))
        pp=GLMParams(coef=coef,intercept=b)
        loss=glm.regularizer.penalized_loss(glm._compute_loss,pp,glm.regularizer_strength)
        def explicit(p):
            eta=x@p.coef+p.intercept
            return jnp.sum(jnp.mean(jnp.exp(eta)-y*eta,axis=0))+.5*config['ridge']*jnp.sum(p.coef**2)
        v,g=jax.value_and_grad(loss)(pp,x,y);v2,g2=jax.value_and_grad(explicit)(pp)
        diff=max(float(np.max(np.abs(u-v))) for u,v in zip(jax.tree_util.tree_leaves(g),jax.tree_util.tree_leaves(g2)))
        if not np.isclose(v,v2,rtol=0,atol=1e-10) or diff>1e-10:raise ValueError('NeMoS penalty normalization mismatch')
        parity={'value_difference':float(abs(v-v2)),'gradient_max_difference':diff}
        class Gate(nmo.callbacks.Callback):
            def on_pass_end(self,ctx):
                if (ctx.pass_idx+1)%10:return
                value,grad=model.value_and_grad(convert(ctx.params.coef,ctx.params.intercept),train)
                norm=max(float(np.max(np.abs(g))) for g in grad.values())
                progress({'pass':ctx.pass_idx+1,'objective':value,'gradient_inf_norm':norm})
                if norm<=1e-5:ctx.request_stop('full penalized gradient gate')
        glm.stochastic_fit(loader,n_passes=config['budget'],init_params=(jnp.zeros((n*k,n)),b),callbacks=Gate())
        params=convert(glm.coef_,glm.intercept_);message=str(glm.stochastic_fit_summary_)
    else:
        fit=model.fit(train,model.initialize(n,seed=0,mean_count=mean),max_iter=config['budget'],
            tolerance=1e-12,callback=progress,checkpoint=a.out.with_suffix('.checkpoint.npz'))
        params=fit.params;message=fit.message
    value,grad=model.value_and_grad(params,train)
    norm=max(float(np.max(np.abs(g))) for g in grad.values())
    save_bundle(a.out.with_suffix('.npz'),model,params)
    write_json(a.out,{'status':'DONE',**config,'gradient_inf_norm':norm,'gradient_gate_passed':norm<=1e-5,
        'objective':value,'message':message,'validation':score(model,params,validation,mean),
        'model':str(a.out.with_suffix('.npz')),'penalty_parity':parity,
        'seconds':time.perf_counter()-started,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20})


def run_child(cmd,path,heartbeat,timeout,meta):
    begin=time.time()
    with path.with_suffix('.log').open('w') as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        while proc.poll() is None:
            write_json(heartbeat,{'status':'RUNNING','pid':os.getpid(),'child_pid':proc.pid,
                'updated_unix':time.time(),'worker_elapsed_s':time.time()-begin,**meta})
            if time.time()-begin>timeout:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                break
            time.sleep(3)
    if path.exists():result=json.loads(path.read_text())
    else:result={'status':'FAILED','error':'timeout' if time.time()-begin>timeout else 'worker failed','returncode':proc.returncode}
    result['worker_wall_seconds']=time.time()-begin
    write_json(path,result)
    return result


def suite(a):
    from .three_way import generate
    a.out.mkdir(parents=True,exist_ok=a.resume)
    ridges=[0.,.001,.01,.1] if not a.smoke else [.01]
    ranks=[2,4,8,16] if not a.smoke else [2]
    steps=[.1,.5,1.] if not a.smoke else [.1]
    budget=1500 if not a.smoke else 10
    configs=[]
    for ridge in ridges:
        configs.append(dict(backend='dense',rank=None,ridge=ridge,stepsize=None,budget=budget))
        configs.extend(dict(backend='lowrank',rank=rank,ridge=ridge,stepsize=None,budget=budget) for rank in ranks)
        configs.extend(dict(backend='nemos',rank=None,ridge=ridge,stepsize=step,budget=budget) for step in steps)
    protocol={'neurons':a.neurons,'seeds':a.seeds,'bins':a.bins,'ridges':ridges,'ranks':ranks,'stepsizes':steps,
        'families':['lowrank','permutation_fullrank'],'split':'60/20/20 chronological with history resets',
        'selection':'highest validation likelihood among gradient-gate-passing candidates within each backend',
        'gate':1e-5,'lowrank_initial_seed':0,'refit':False,'candidate_timeout_s':a.timeout,
        'precision':'float64','ridge_definition':'0.5 * ridge * sum(W**2), intercept unpenalized, loss summed over neurons',
        'upstream_commit':'81c7200a0e66686e98fd1907e7fa11e111a0e66a',
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    protocol_path=a.out/'protocol.json'
    if a.resume:
        saved=json.loads(protocol_path.read_text())
        if {k:v for k,v in saved.items() if k!='source_sha256'} != {k:v for k,v in protocol.items() if k!='source_sha256'}:
            raise ValueError('resume scientific protocol mismatch')
        changed=[k for k,v in saved['source_sha256'].items() if protocol['source_sha256'].get(k)!=v]
        if any(k!='tuned_comparison.py' for k in changed):
            raise ValueError(f'resume numerical source changed: {changed}')
        write_json(a.out/f'resume_{time.time_ns()}.json',{'updated_unix':time.time(),'changed_sources':changed,'source_sha256':protocol['source_sha256'],'note':'runner-only resume patch; saved completed workers, including failures, are reused'})
    else:
        write_json(protocol_path,protocol)
    previous_selected={}
    if a.resume and (a.out/'selected.json').exists():
        previous_selected={(r['dataset'],r['backend']):r for r in json.loads((a.out/'selected.json').read_text())}
    records=[];selected=[]
    for n,seed,family in itertools.product(a.neurons,a.seeds,protocol['families']):
        name=f'{family}_n{n}_s{seed}';folder=a.out/name;folder.mkdir(exist_ok=a.resume)
        write_json(a.out/'status.json',{'status':'RUNNING','step':'generate','dataset':name,'pid':os.getpid(),'updated_unix':time.time()})
        data=folder/'data.npz';truth=folder/'truth.npz'
        if not (a.resume and data.exists() and truth.exists()):
            counts,basis,weights,intercept=generate(a.bins,n,seed,family)
            np.savez_compressed(data,counts=counts,basis=basis)
            np.savez_compressed(truth,weights=weights,intercept=intercept)
        candidates=[]
        for i,config in enumerate(configs):
            configpath=folder/f'candidate{i:02d}.config.json';out=folder/f'candidate{i:02d}.json'
            if configpath.exists() and json.loads(configpath.read_text())!=config:
                raise ValueError(f'candidate configuration changed: {configpath}')
            if not configpath.exists(): write_json(configpath,config)
            cmd=[sys.executable,'-u','-m','streamglm.tuned_comparison','--worker','--config',str(configpath),'--data',str(data),'--out',str(out)]
            if a.resume and out.exists():
                row=json.loads(out.read_text())
                if row.get('status') not in ('DONE','FAILED'): raise ValueError(f'invalid checkpoint: {out}')
            else:
                row=run_child(cmd,out,a.out/'status.json',a.timeout,{'dataset':name,'candidate':i,'backend':config['backend'],'completed':len(records)})
            row.update(config);row.update(dataset=name,neurons=n,seed=seed,family=family)
            candidates.append(row);records.append(row);write_json(a.out/'candidates.json',records)
        for backend in ('dense','lowrank','nemos'):
            if (name,backend) in previous_selected:
                selected.append(previous_selected[(name,backend)])
                write_json(a.out/'selected.json',selected)
                continue
            group=[r for r in candidates if r['backend']==backend]
            best=select_candidate(group)
            selection={'dataset':name,'backend':backend,'neurons':n,'seed':seed,'family':family,
                'tuning_seconds':sum(r['worker_wall_seconds'] for r in group),
                'candidates':len(group),'eligible':sum(r.get('gradient_gate_passed',False) for r in group)}
            if best is None:selection.update(status='NO_ELIGIBLE_MODEL')
            else:
                # Freeze selection before starting any held-out evaluation.
                selection.update(status='SELECTED',selected=best)
                locked=folder/f'{backend}_selection.json';write_json(locked,selection)
                configpath=folder/f'{backend}_evaluation.config.json';write_json(configpath,best)
                out=folder/f'{backend}_test.json'
                cmd=[sys.executable,'-u','-m','streamglm.tuned_comparison','--worker','--evaluate','--config',str(configpath),'--data',str(data),'--truth',str(truth),'--out',str(out)]
                evaluation=json.loads(out.read_text()) if a.resume and out.exists() else run_child(cmd,out,a.out/'status.json',300,{'dataset':name,'step':'selected_test','backend':backend})
                selection.update(evaluation=evaluation)
                if 'test' not in evaluation: selection['status']='FAILED_EVALUATION'
            selected.append(selection);write_json(a.out/'selected.json',selected)
    write_json(a.out/'status.json',{'status':'DONE','completed':len(records),'failed':sum(r['status']!='DONE' for r in records),'selected_models':sum(s['status']=='SELECTED' for s in selected),'updated_unix':time.time()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--worker',action='store_true');p.add_argument('--evaluate',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--config',type=Path);p.add_argument('--data',type=Path);p.add_argument('--truth',type=Path)
    p.add_argument('--neurons',nargs='+',type=int,default=[64,128]);p.add_argument('--seeds',nargs='+',type=int,default=[31,47])
    p.add_argument('--bins',type=int,default=12000);p.add_argument('--timeout',type=int,default=900);p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.worker:
        worker(a)
    else:
        existed=a.out.exists()
        a.out.parent.mkdir(parents=True,exist_ok=True)
        lock=a.out.with_name(a.out.name+'.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            suite(a)
        except BaseException as exc:
            if not existed and a.out.exists():
                write_json(a.out/'status.json',{'status':'FAILED','error':repr(exc),'pid':os.getpid(),'updated_unix':time.time()})
            raise

if __name__=='__main__':main()
