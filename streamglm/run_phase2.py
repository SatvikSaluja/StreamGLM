"""Checkpoint-preserving production queue; exclusive benchmark after HNN ends."""
import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
import psutil
from .persistence import write_json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/v8';HNN=Path('/home/satvik/hnn_neuro_bridge/results/phase2/status.json')

def launch(name,cmd):
    start=time.monotonic()
    with (OUT/f'{name}.log').open('a') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while proc.poll() is None:
                free=psutil.virtual_memory().available/2**30
                write_json(OUT/'queue_status.json',{'status':'RUNNING','pid':os.getpid(),'child_pid':proc.pid,'step':name,'elapsed_s':time.monotonic()-start,'available_gib':free,'updated_unix':time.time()})
                if free<3:raise RuntimeError('available memory below 3GiB; checkpoints retained')
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:pass
            if proc.returncode:raise RuntimeError(f'{name} exited {proc.returncode}; see log')
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise

def main():
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def stop(*_):raise RuntimeError('queue interrupted; checkpoints retained')
    signal.signal(signal.SIGTERM,stop)
    try:
        for seed in [941,942]:
            folder=OUT/f'coupled1000_seed{seed}'
            if not (folder/'report.json').exists():launch(f'coupled1000_seed{seed}',[sys.executable,'-u','-m','streamglm.coupled_scale','--out',str(folder),'--seed',str(seed)])
        # Performance measurements must not compete with long HNN simulations.
        while True:
            state=json.loads(HNN.read_text()) if HNN.exists() else {}
            busy=[]
            for p in psutil.process_iter(['pid','cmdline','name']):
                try:
                    cmd=' '.join(p.info['cmdline'] or [])
                    if p.info['name']=='nrniv' or ('python' in (p.info['name'] or '').lower() and ('experiments/run_phase2.py' in cmd or 'experiments/finish_pending.py' in cmd)):busy.append(p.pid)
                except psutil.Error:pass
            if not busy and state.get('status') in ['DONE','FAILED']:break
            write_json(OUT/'queue_status.json',{'status':'WAITING_FOR_EXCLUSIVE_BENCHMARK','pid':os.getpid(),'hnn_status':state.get('status'),'competing_pids':busy,'updated_unix':time.time()});time.sleep(30)
        folder=OUT/'streaming_comparison'
        if not (folder/'report.md').exists():launch('streaming_comparison',[sys.executable,'-u','-m','streamglm.streaming_benchmark','--out',str(folder)])
        write_json(OUT/'queue_status.json',{'status':'DONE','updated_unix':time.time(),'scope':'runs completed; inspect convergence and recovery, no automatic scientific success claim'})
    except BaseException as exc:
        write_json(OUT/'queue_status.json',{'status':'FAILED','error':repr(exc),'updated_unix':time.time()});raise
if __name__=='__main__':main()
