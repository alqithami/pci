"""Reproducible orchestration. Fresh per-run folders; no legacy Makefile or CSV merge.

The full profiles cannot start training until real crypto preflight passes.
Use --core-only only with --profile smoke for the explicitly limited CPU test.
"""
from __future__ import annotations
import argparse,copy,concurrent.futures,fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
from .io import atomic_json,canonical_hash,source_hashes,system_meta,sha256
from .environment import EnvConfig

ROOT=Path(__file__).resolve().parents[1]

def create_plan(profile):
    if profile not in ('smoke','deadline','extended'):raise ValueError(profile)
    base=json.loads((ROOT/'configs/base.json').read_text())
    if profile=='smoke':
        layouts=['twodoors'];seeds=[0];methods=['ppo_penalty','consensus','pci'];steps=4096;episodes=2
        base.update(n_envs=4,rollout_steps=64,minibatch_groups=64)
    else:
        layouts=['open','twodoors','staggered'];seeds=list(range(5 if profile=='deadline' else 10))
        methods=['ppo_task','ppo_penalty','consensus','pci','pci_shuffled','global_ppo']
        steps=262144 if profile=='deadline' else 524288;episodes=24 if profile=='deadline' else 48
    runs=[]
    def add(method,layout,seed,lam):
        cfg=copy.deepcopy(base);name=f'{method}_{layout}_l{lam:g}_s{seed}'
        cfg.update(method=method,seed=seed,run_id=name,total_steps=steps,consistency_coef=lam)
        cfg['env']['layout']=layout;runs.append(cfg)
    for layout in layouts:
        for method in methods:
            for seed in seeds:add(method,layout,seed,.2)
    if profile!='smoke':
        # A matched regularization-strength sweep, not PCI-only hyperparameter search.
        for method in ('pci','consensus'):
            for lam in (.05,1.0):
                for seed in seeds:add(method,'twodoors',seed,lam)
    controllers=[{'run_id':f'{method}_{layout}','method':method,'layout':layout} for layout in layouts
                 for method in ('rule','market','distributed_consensus','centralized_priority')]
    return {'profile':profile,'version':1,'runs':runs,'controllers':controllers,'eval_episodes':episodes,
            'full_scenarios':profile!='smoke','total_joint_training_steps':sum(x['total_steps'] for x in runs),
            'crypto_anchor':'pci_twodoors_l0.2_s0','crypto_steps':2 if profile=='smoke' else 32,
            'crypto_agents':[2] if profile=='smoke' else [2,4,8],
            'scope':'Custom PCI-Warehouse discrete simulator; fixed, state-instantiated linear overlap maps; no learned institution, cohomology, or real-robot claim'}

def call(cmd,log):
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',encoding='utf-8') as f:
        f.write('\nCOMMAND: '+repr(cmd)+'\n');f.flush()
        p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError(f'Exit {p.returncode}: see {log}')

def run_job(root,job,episodes,full,controller=False):
    folder=root/('controllers' if controller else 'runs')/job['run_id'];folder.mkdir(parents=True,exist_ok=True)
    atomic_json(folder/'job.json',job)
    try:
        if controller:
            cmd=[sys.executable,'-m','pci_bench.job','--controller',job['method'],'--layout',job['layout'],
                 '--out',str(folder),'--episodes',str(episodes)]
            if not full:cmd+=['--small']
            call(cmd,folder/'console.log')
        else:
            config=folder/'requested_config.json';atomic_json(config,job)
            call([sys.executable,'-m','pci_bench.train','--config',str(config),'--out',str(folder)],folder/'console.log')
            cmd=[sys.executable,'-m','pci_bench.evaluate','--run-dir',str(folder),'--episodes',str(episodes)]
            if not full:cmd+=['--small']
            call(cmd,folder/'console.log')
        failure=folder/'FAILED.json'
        if failure.exists():failure.unlink()
        return job['run_id']
    except Exception as e:
        atomic_json(folder/'FAILED.json',{'error':str(e),'time':time.time()});raise

def run(args):
    out=Path(args.results).resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.suite.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError(f'Another runner owns {out}')
    if args.core_only and args.profile!='smoke':raise ValueError('--core-only is allowed only for the limited smoke test')
    plan=create_plan(args.profile);plan['core_only']=bool(args.core_only)
    plan['sources']=source_hashes(ROOT);plan['source_digest']=canonical_hash(plan['sources'])
    path=out/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise RuntimeError('Code/profile changed. Use a NEW results directory; never mix runs.')
    atomic_json(path,plan);atomic_json(out/'machine.json',system_meta())
    if args.plan_only:
        print(json.dumps({k:v for k,v in plan.items() if k not in ('runs','controllers','sources')},indent=2));return
    if args.jobs<1:raise ValueError('jobs must be >=1')
    print(f"PLAN: {len(plan['runs'])} training runs; {len(plan['controllers'])} controllers; {plan['total_joint_training_steps']:,} joint training steps",flush=True)
    started=time.perf_counter()
    print('1. Running Python correctness tests (no skipped crypto substitute).',flush=True)
    call([sys.executable,'-m','pytest','-q',str(ROOT/'tests')],out/'preflight/python_tests.log')
    if not args.core_only:
        print('2. Building every planned population circuit and executing genuine positive/negative proof tests BEFORE training.',flush=True)
        call([sys.executable,'-m','pci_bench.crypto_build','--agents',*map(str,plan['crypto_agents'])],out/'preflight/crypto_build.log')
        call([sys.executable,'-m','pci_bench.crypto_check','--out',str(out/'preflight/crypto')],out/'preflight/crypto_check.log')
    else:
        atomic_json(out/'preflight/CRYPTO_NOT_TESTED.json',{'reason':'--core-only: diagnostic smoke only; no cryptographic or paper-readiness claim'})
    print('3. Preflight passed for the requested scope. Running learning/evaluation jobs.',flush=True)
    tasks=[(j,False) for j in plan['runs']]+[(j,True) for j in plan['controllers']]
    failures=[];finished=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures={pool.submit(run_job,out,j,plan['eval_episodes'],plan['full_scenarios'],c):(j['run_id'],c) for j,c in tasks}
        for future in concurrent.futures.as_completed(futures):
            name,_=futures[future]
            try:future.result();finished+=1;print(f'COMPLETE {finished}/{len(tasks)}: {name}',flush=True)
            except Exception as e:
                failures.append({'run_id':name,'error':str(e)})
                for f in futures:f.cancel()
                print(f'FAILED {name}: {e}. Queued jobs cancelled; active jobs checkpoint/finish.',flush=True)
                break
    if failures:
        atomic_json(out/'SUITE_FAILED.json',{'failures':failures});raise RuntimeError('Suite incomplete; see SUITE_FAILED.json. No automatic partial-result plots.')
    print('4. Independent oracle diagnostics.',flush=True)
    call([sys.executable,'-m','pci_bench.diagnostics','--out',str(out/'diagnostics')],out/'diagnostics.log')
    if not args.core_only:
        print('5. Real paired certificates on prespecified held-out transitions.',flush=True)
        call([sys.executable,'-m','pci_bench.crypto_benchmark','--run-dir',str(out/'runs'/plan['crypto_anchor']),
              '--out',str(out/'crypto_benchmark'),'--steps',str(plan['crypto_steps']),
              '--agents',*map(str,plan['crypto_agents'])],out/'crypto_benchmark.log')
    print('6. Strict analysis and vector figure export.',flush=True)
    call([sys.executable,'-m','pci_bench.analysis','--results',str(out)],out/'analysis.log')
    if not args.core_only:
        report=json.loads((out/'analysis/ANALYSIS_REPORT.json').read_text())
        if report['status']!='COMPLETE':raise RuntimeError('Analysis failed the completeness gate')
    status='CORE_SMOKE_ONLY' if args.core_only else 'COMPLETE'
    atomic_json(out/'SUITE_COMPLETE.json',{'status':status,'profile':args.profile,'jobs':args.jobs,'elapsed_this_invocation_s':time.perf_counter()-started,
        'source_digest':plan['source_digest'],'training_runs':len(plan['runs']),'controller_runs':len(plan['controllers']),
        'full_crypto_measured':not args.core_only,'scientific_effects':'must be interpreted from measurements; no guarantee of favorable effects'})
    print(f'{status}: {out}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--profile',choices=['smoke','deadline','extended'],default='deadline')
    p.add_argument('--jobs',type=int,default=12);p.add_argument('--results',default='results/deadline')
    p.add_argument('--core-only',action='store_true');p.add_argument('--plan-only',action='store_true')
    a=p.parse_args();run(a)
