import argparse,concurrent.futures,copy,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from . import ROOT,BASE
from .io import atomic_json,immutable_json,sources,versions,canonical_hash,sha256

def plan(smoke=False):
    protocol=json.loads((ROOT/'protocol.json').read_text());base=json.loads((BASE/'configs/base.json').read_text())
    runs=[]
    for domain,layouts in protocol['domains'].items():
        for layout in (layouts[:1] if smoke else layouts):
            for method in protocol['methods']:
                for seed in ([901] if smoke else protocol['seeds']):
                    c=copy.deepcopy(base);c.update(method=method,domain=domain,seed=seed,run_id=f'{domain}_{layout}_{method}_s{seed}',
                        cost_budget=protocol['cost_budget'],dual_lr=protocol['dual_lr'],dual_max=protocol['dual_max'])
                    c['env']['layout']=layout
                    if smoke:c.update(total_steps=1024,n_envs=2,rollout_steps=32,minibatch_groups=32);c['env']['horizon']=64
                    else:c['total_steps']=protocol['total_steps']
                    runs.append(c)
    return {'study':protocol['study'],'profile':'smoke' if smoke else 'full','protocol':protocol,'runs':runs,
        'eval_episodes':2 if smoke else protocol['evaluation_episodes'],'sources':sources(),'versions':versions()}

def call(cmd,log):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    with log.open('a') as f:
        f.write('\nCOMMAND '+repr(cmd)+'\n');f.flush();p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError(f'Exit {p.returncode}; see {log}')

def one_job(root,c,episodes,small):
    d=root/'runs'/c['run_id'];d.mkdir(parents=True,exist_ok=True);immutable_json(d/'requested_config.json',c)
    try:
        call([sys.executable,'-m','pci_followup.train','--config',str(d/'requested_config.json'),'--out',str(d)],d/'console.log')
        cmd=[sys.executable,'-m','pci_followup.evaluate','--run-dir',str(d),'--episodes',str(episodes)]
        if small:cmd+=['--small']
        call(cmd,d/'console.log')
        if (d/'FAILED.json').exists():(d/'FAILED.json').unlink()
    except Exception as e:atomic_json(d/'FAILED.json',{'error':str(e),'time':time.time()});raise
    return c['run_id']

def run(args):
    root=Path(args.results).resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=plan(args.smoke);immutable_json(root/'plan.json',p)
    print('FOLLOWUP_PLAN',len(p['runs']),'learned runs;',sum(c['total_steps'] for c in p['runs']),'joint steps',flush=True)
    call([sys.executable,'-m','pytest','-q',str(ROOT/'tests')],root/'preflight/python_tests.log')
    if not args.smoke:
        smoke=ROOT/'results/smoke';marker=smoke/'SUITE_COMPLETE.json'
        if not marker.exists():raise RuntimeError('Smoke completion missing')
        m=json.loads(marker.read_text())
        if m['source_digest']!=canonical_hash(sources()):raise RuntimeError('Sources changed after smoke')
        call([sys.executable,'-m','pci_followup.crypto_extended','--preflight','--out',str(root/'preflight/crypto')],root/'preflight/crypto.log')
    completed=0;failed=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures={ex.submit(one_job,root,c,p['eval_episodes'],args.smoke):c['run_id'] for c in p['runs']}
        for f in concurrent.futures.as_completed(futures):
            try:f.result();completed+=1;print(f'LEARNING_COMPLETE {completed}/{len(p["runs"])} {futures[f]}',flush=True)
            except Exception as e:
                failed.append({'run_id':futures[f],'error':str(e)});print('FAILED',futures[f],str(e),flush=True)
                for x in futures:x.cancel()
                break
    if failed:atomic_json(root/'SUITE_FAILED.json',{'failures':failed});raise RuntimeError('Incomplete follow-up grid')
    # Dedicated post-learning timing phase avoids CPU training contention.
    print('LEARNING_FINISHED; CERTIFIED_EPISODES_START',flush=True)
    cmd=[sys.executable,'-m','pci_followup.crypto_extended','--out',str(root/'certified')]
    if args.smoke:cmd+=['--small']
    call(cmd,root/'certified.log')
    call([sys.executable,'-m','pci_followup.analysis','--results',str(root)],root/'analysis.log')
    atomic_json(root/'SUITE_COMPLETE.json',{'status':'COMPLETE','profile':p['profile'],'source_digest':canonical_hash(sources()),
        'learned_runs':len(p['runs']),'versions':versions(),'completed_at_unix':time.time()})
    print('FOLLOWUP_COMPLETE',root,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--smoke',action='store_true');a.add_argument('--jobs',type=int,default=12)
    a.add_argument('--results',default='results/full');run(a.parse_args())
