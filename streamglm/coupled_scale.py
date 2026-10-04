"""Known coupled 1000-neuron calibration, distinct from the independent null.

Inhibitory block low-rank generator has bounded Poisson means without clipping.
Generation uses exact latent projections, not an N-by-N convolution. Recovery
is assessed on a fixed off-diagonal pair sample, never factor coordinates.
"""
import argparse,json,os,time
from pathlib import Path
import numpy as np
from .persistence import write_json,save_bundle,load_bundle

def generate(root,neurons=1000,bins=360000,seed=941,stop_after=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    spec={'neurons':neurons,'bins':bins,'seed':seed,'bin_s':.01,'rank':4,'basis':2,'history':8,'schema':1}
    cfg=root/'generator.json'
    if cfg.exists() and json.loads(cfg.read_text())!=spec:raise ValueError('generator configuration differs')
    write_json(cfg,spec)
    rng=np.random.default_rng(seed);rank=4;k=2;h=8
    groups=np.arange(neurons)%rank;u=np.zeros((k,neurons,rank));v=np.zeros_like(u)
    sizes=np.bincount(groups,minlength=rank)
    for basis in range(k):
        u[basis,np.arange(neurons),groups]=rng.uniform(.8,1.2,neurons)
        v[basis,np.arange(neurons),groups]=-3./sizes[groups]*rng.uniform(.8,1.2,neurons)
    basis=np.exp(-np.arange(1,h+1)[:,None]/np.array([1.5,4.]));basis/=basis.sum(axis=0)
    intercept=np.full(neurons,np.log(.2))
    truth=root/'truth.npz'
    if not truth.exists():np.savez_compressed(truth,U=u,V=v,b=intercept,basis=basis)
    path=root/'counts.npy';ck=root/'generation_checkpoint.npz';hist=np.zeros((h,k,rank));start=0
    if ck.exists():
        with np.load(ck) as z:start=int(z['next_bin']);hist=z['history'];rng.bit_generator.state=json.loads(str(z['rng']))
    counts=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.int16,shape=(bins,neurons))
    end=bins if stop_after is None else min(bins,start+stop_after)
    for t in range(start,end):
        latent=np.einsum('hk,hkr->kr',basis,hist)
        eta=intercept+np.einsum('kr,kjr->j',latent,v)
        events=rng.poisson(np.exp(eta))
        if events.max()>np.iinfo(np.int16).max:raise OverflowError('spike bin overflow')
        counts[t]=events
        projected=np.einsum('i,kir->kr',events,u)
        hist[1:]=hist[:-1].copy();hist[0]=projected
        if (t+1)%2048==0 or t+1==end:
            counts.flush();tmp=root/'generation_checkpoint.tmp.npz'
            np.savez(tmp,next_bin=t+1,history=hist,rng=json.dumps(rng.bit_generator.state));tmp.replace(ck)
            write_json(root/'generation_status.json',{'status':'DONE' if t+1==bins else 'GENERATING','bins_done':t+1,'bins':bins,'pid':os.getpid(),'updated_unix':time.time()})
    return path

def recovery(model,params,truth,seed=119):
    rng=np.random.default_rng(seed);n=len(truth['b']);size=min(100000,n*(n-1));i=rng.integers(0,n,size*2);j=rng.integers(0,n,size*2);ok=i!=j;i,j=i[ok][:size],j[ok][:size]
    true_coef=np.einsum('kir,kir->ki',truth['U'][:,i],truth['V'][:,j])
    if 'W' in params:estimated=params['W'][:,i,j]
    else:estimated=np.einsum('kir,kir->ki',params['U'][:,i],params['V'][:,j])
    actual=model.basis@true_coef;inferred=model.basis@estimated
    from sklearn.metrics import roc_auc_score,average_precision_score
    labels=np.any(true_coef!=0,axis=0);scores=np.abs(inferred.sum(axis=0))
    return {'sampled_offdiagonal_pairs':len(i),'pair_sampling':'fixed random sample with replacement; descriptive metrics, not independent-replicate uncertainty',
        'filter_correlation':float(np.corrcoef(actual.ravel(),inferred.ravel())[0,1]),
        'filter_relative_error':float(np.linalg.norm(inferred-actual)/np.linalg.norm(actual)),
        'edge_auc':float(roc_auc_score(labels,scores)),'edge_average_precision':float(average_precision_score(labels,scores)),
        'edge_prevalence':float(labels.mean())}

def run(root,seed,neurons,bins):
    import jax
    jax.config.update('jax_enable_x64',True)
    from .data import Recording
    from .model import StreamingGLM
    from .evaluation import split_recording,score
    import resource
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    def status(**kw):write_json(root/'status.json',{'pid':os.getpid(),'updated_unix':time.time(),**kw})
    status(status='RUNNING',step='generation');path=generate(root/'data',neurons,bins,seed)
    with np.load(root/'data/truth.npz') as z:truth={k:z[k] for k in z.files}
    recording=Recording.from_npy(path,bin_s=.01);train,val,test=split_recording(recording);basis=truth['basis'];mean=train.mean_counts(len(basis))
    # Rank4 matches planted rank; rank8 tests a looser constraint. All choices
    # use validation only. Start seeds are independent of generator parameters.
    configs=[(0,.01,0)]+[(rank,ridge,start) for rank in [4,8] for ridge in [.001,.01] for start in [0,1]]
    protocol={'configs':configs,'max_iterations':1500,'gate':1e-5,'seed':seed,'neurons':neurons,'bins':bins,'bin_s':.01,'split':[.6,.2,.2],'self_history':True,'precision':'float64','truth_used_for_initialization':False,'scope':'inhibitory rank4 block synthetic network; not real-data evidence or arbitrary full-rank recovery'}
    write_json(root/'protocol.json',protocol);results=[]
    for idx,(rank,ridge,start) in enumerate(configs):
        target=root/f'candidate{idx}.json';bundle=root/f'candidate{idx}.npz';checkpoint=root/f'checkpoint{idx}.npz'
        if target.exists():results.append(json.load(open(target)));continue
        model=StreamingGLM(basis,chunk_size=2048,rank=rank,ridge=ridge,self_history=True)
        status(status='RUNNING',step='fit',candidate=idx,rank=rank,ridge=ridge,start=start)
        def progress(row):status(status='RUNNING',step='fit',candidate=idx,rank=rank,**row)
        try:
            fitted=model.fit(train,initial=model.initialize(neurons,seed=start,mean_count=mean),max_iter=1500,tolerance=1e-12,
                checkpoint=checkpoint,resume=checkpoint.exists(),callback=progress)
            save_bundle(bundle,model,fitted.params)
            row={'status':'DONE','candidate':idx,'rank':rank,'ridge':ridge,'start':start,'gradient_inf_norm':fitted.gradient_inf_norm,
                'gate_passed':fitted.gradient_inf_norm<=1e-5,'optimizer_message':fitted.message,'iterations':fitted.iterations,'seconds':fitted.seconds,
                'validation':score(model,fitted.params,val,mean),'model':str(bundle)}
        except (FloatingPointError,ValueError) as exc:row={'status':'FAILED','candidate':idx,'rank':rank,'error':repr(exc)}
        write_json(target,row);results.append(row)
    eligible=[a for a in results if a.get('gate_passed') and a['rank']>0]
    report={'status':'DONE','candidates':results,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20}
    if eligible:
        best=max(eligible,key=lambda a:a['validation']['log_likelihood_per_bin']);model,params,_,_=load_bundle(Path(best['model']))
        report.update(selected=best,test=score(model,params,test,mean),recovery=recovery(model,params,truth))
        rng_null=np.random.default_rng(seed+90000);nulls=[]
        for _ in range(20):
            permutation=rng_null.permutation(neurons)
            permuted=dict(params)
            if 'W' in params:permuted['W']=params['W'][:,:,permutation]
            else:permuted['V']=params['V'][:,permutation,:]
            nulls.append(recovery(model,permuted,truth))
        report['target_label_permutation_recovery']=nulls
        report['permutation_scope']='20 target-label permutations of learned off-diagonal filters; descriptive recovery control, not surrogate refits or independent biological replications'
        # A targetwise independently time-shifted held-out source control:
        # preserve marginal spike trains, disrupt cross-neuron timing; score
        # the same learned model on surrogate outcomes and predictors.
        shifted=root/'shifted_test.npy';arr=np.lib.format.open_memmap(shifted,mode='w+',dtype=np.int16,shape=test.counts.shape)
        rng=np.random.default_rng(seed+50000)
        for j in range(neurons):arr[:,j]=np.roll(np.asarray(recording.counts[int(.8*bins):,j]),int(rng.integers(100,len(arr)-100)))
        arr.flush();del arr
        report['shifted_test']=score(model,params,Recording.from_npy(shifted,bin_s=.01),mean)
        if results[0].get('gate_passed'):
            base,bp,_,_=load_bundle(Path(results[0]['model']));report['self_history_test']=score(base,bp,test,mean)
            report['gain_over_self_history_nats_per_bin']=report['test']['log_likelihood_per_bin']-report['self_history_test']['log_likelihood_per_bin']
        oracle=StreamingGLM(basis,chunk_size=2048,rank=4);report['oracle_test']=score(oracle,{k:truth[k] for k in ['U','V','b']},test,mean)
    else:report['interpretation']='No converged coupling candidate; no recovery claim.'
    write_json(root/'report.json',report);status(status='DONE',selected_converged=bool(eligible))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=941);p.add_argument('--neurons',type=int,default=1000);p.add_argument('--bins',type=int,default=360000);a=p.parse_args();run(a.out,a.seed,a.neurons,a.bins)
