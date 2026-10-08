"""Held-out evaluation, matched seeds, trace persistence, and explicit controllers."""
from __future__ import annotations
import argparse, copy, gzip, json, time
from dataclasses import replace, asdict
from pathlib import Path
import numpy as np
import torch
from .environment import EnvConfig, Warehouse, MOVES
from .model import ContextPolicy, compatibility_energy
from .io import atomic_json, atomic_csv, sha256, canonical_hash

HEURISTICS=('rule','market','distributed_consensus','centralized_priority')


def heuristic(env,method,rng):
    f=env.features();n=env.n
    # Same known goal distances and disclosed local geometry, not learned rewards.
    score=f[:,:,0]-10*f[:,:,1]
    score[:,0]-=0.01
    cache=getattr(env,'controller_cache',{'pos':env.pos.copy(),'stall':np.zeros(n,dtype=int)})
    stalled=(cache['pos']==env.pos).all(axis=1)
    stall=np.where(stalled,cache['stall']+1,0)
    env.controller_cache={'pos':env.pos.copy(),'stall':stall}
    # Deadlock recovery is explicit: after five stationary decisions, a one-step
    # retreat is preferable to waiting. This is controller logic, not data noise.
    score[:,0]-=1.1*(stall>=5)
    score-=1.25*(f[:,:,7]+f[:,:,8])
    # Fixed, seeded perturbation only breaks tied feasible action preferences.
    score+=rng.uniform(0,1e-5,score.shape)
    if method=='rule':
        return np.argmax(score-2*f[:,:,2],axis=1).astype(np.int64)
    if method=='distributed_consensus':
        # Agents advertise their first choice. Two max-consensus rounds over a
        # complete communication graph elect one bid per proposed destination.
        proposal=np.argmax(score,axis=1); destinations=env.pos+MOVES[proposal]
        bids=[(float(score[i,proposal[i]])+1e-4*env.wait[i],-i) for i in range(n)]
        known=[{i:(tuple(destinations[i]),bids[i])} for i in range(n)]
        for _ in range(2):
            union={k:v for state in known for k,v in state.items()}
            known=[dict(union) for _ in range(n)]
        out=proposal.copy()
        for i in range(n):
            candidates=[j for j,(cell,_) in known[i].items() if cell==tuple(destinations[i])]
            winner=max(candidates,key=lambda j:known[i][j][1])
            if i!=winner:out[i]=0
        return out
    if method=='market':
        # Greedy reservation auction: all feasible agent->cell bids; losers can
        # obtain a second-choice cell, unlike the consensus first-choice baseline.
        items=[]
        for i in range(n):
            for a in range(5):
                target=tuple(env.pos[i]+MOVES[a])
                if f[i,a,1] or (a and any(tuple(env.pos[j])==target for j in range(n) if j!=i)):continue
                items.append((float(score[i,a])+1e-4*env.wait[i],-i,a,target))
        out=np.zeros(n,dtype=np.int64);assigned=set();cells=set()
        for _,neg_i,a,target in sorted(items,reverse=True):
            i=-neg_i
            if i not in assigned and target not in cells:
                out[i]=a;assigned.add(i);cells.add(target)
        return out
    if method=='centralized_priority':
        # Centralized priority inheritance with recursive displacement. Each
        # blocked robot asks the occupant to vacate; two-agent swaps are forbidden.
        # A bounded reactive planner, NOT an optimal/MPC upper bound.
        assigned={}; reserved=set(); occupied={tuple(p):i for i,p in enumerate(env.pos)}
        def plan(i, forbidden, stack):
            if i in assigned:return True
            if i in stack:return False
            stack=stack|{i}
            for a in np.argsort(-score[i],kind='stable'):
                a=int(a);cell=tuple(env.pos[i]+MOVES[a])
                if f[i,a,1] or cell in reserved or cell==forbidden:continue
                occupant=occupied.get(cell)
                snapshot=dict(assigned); rs=set(reserved)
                reserved.add(cell)
                if occupant is not None and occupant!=i:
                    if occupant in assigned:
                        if tuple(env.pos[occupant]+MOVES[assigned[occupant]])==cell:
                            reserved.clear();reserved.update(rs);continue
                    elif not plan(occupant,tuple(env.pos[i]),stack):
                        assigned.clear();assigned.update(snapshot);reserved.clear();reserved.update(rs);continue
                assigned[i]=a
                return True
            return False
        order=sorted(range(n),key=lambda i:(-int(env.wait[i]),(i-env.t)%n))
        for i in order:
            if i not in assigned and not plan(i,None,set()):
                assigned[i]=0;reserved.add(tuple(env.pos[i]))
        return np.array([assigned.get(i,0) for i in range(n)],dtype=np.int64)
    raise ValueError(method)


def jain(values):
    v=np.asarray(values,dtype=float)
    # Empty service is not perfect fairness. It is undefined, reported blank.
    return float(v.sum()**2/(len(v)*np.square(v).sum())) if np.square(v).sum()>0 else None


def one_episode(env,*,model=None,method=None,seed=0,enforce=False,trace_file=None,certifier=None):
    rng=np.random.default_rng(seed+7000000)
    sums={};start=time.perf_counter();energies=[]
    while env.t<env.config.horizon:
        f=env.features()
        if model is not None:
            device=next(model.parameters()).device
            with torch.no_grad():
                dist,_,p=model(torch.as_tensor(f[None],device=device))
                probabilities=dist.probs[0].cpu().numpy().astype(np.float64)
                actions=np.array([rng.choice(5,p=v.astype(float)/v.sum()) for v in probabilities],dtype=np.int64)
                energies.append(float(compatibility_energy(p,torch.as_tensor(f[None],device=device)).mean().cpu()))
        else:actions=heuristic(env,method,rng)
        _,_,_,_,metrics=env.transition(actions,enforce=enforce,certifier=certifier)
        for k,v in metrics.items():sums[k]=sums.get(k,0)+v
        if trace_file is not None:
            trace_file.write(json.dumps({'proposed_actions':actions.tolist(),'record':env.last,'metrics':metrics},separators=(',',':'))+'\n')
    return {'deliveries':sums['delivered'],'throughput_per_1000_steps':1000*sums['delivered']/env.config.horizon,
            'task_return':sums['task_reward'],'proposed_compliance':sums['proposed_compliant']/env.config.horizon,
            'executed_compliance':sums['executed_compliant']/env.config.horizon,
            'collision_attempts':sums['collision_attempts'],'quota_violations':sums['quota_violations'],
            'capacity_excess':sums['capacity_excess'],'interventions':sums['interventions'],
            'actual_vertex_collisions':sums['actual_vertex_collisions'],'move_count':sums['move_count'],
            'max_wait_final':int(env.wait.max()),'jain_delivery':jain(env.delivered),
            'mean_consistency_energy':float(np.mean(energies)) if energies else None,
            'wall_seconds':time.perf_counter()-start,'episode_len':env.config.horizon,
            'per_agent_deliveries':json.dumps(env.delivered.tolist())}


def scenarios(base:EnvConfig,full=True):
    result={'nominal':base,'quota_tight':replace(base,quota=1),'rule_shift':replace(base,rule_change=True)}
    if full:
        result['scale_2']=replace(base,n_agents=2)
        result['scale_8']=replace(base,n_agents=8)
        result['scale_12']=replace(base,n_agents=12)
    return result


def evaluate_run(run_dir,episodes=24,full=True,heuristic_name=None,base_config=None,seed=0):
    run_dir=Path(run_dir);run_dir.mkdir(parents=True,exist_ok=True)
    if heuristic_name is None:
        config=json.loads((run_dir/'config.json').read_text())
        torch.set_num_threads(1)
        model=ContextPolicy(config['method'],config.get('hidden',64))
        ck=torch.load(run_dir/'checkpoint.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model']);model.eval()
        config_hash=sha256(run_dir/'checkpoint.pt')
        base=EnvConfig(**config['env']);seed=config['seed'];method=config['method'];run_id=config['run_id']
        lam=config['consistency_coef']
    else:
        model=None;method=heuristic_name;base=base_config or EnvConfig();lam=0.
        run_id=f'{method}_{base.layout}_s{seed}';config_hash=canonical_hash({'method':method,'env':asdict(base),'seed':seed})
    spec={'model':config_hash,'episodes':episodes,'full':full}
    complete=run_dir/'EVAL_COMPLETE.json'
    if complete.exists():
        old=json.loads(complete.read_text())
        if old['spec']!=spec:raise RuntimeError('Evaluation specification changed; choose a new output directory')
        if sha256(run_dir/'evaluation.csv')!=old['csv_sha256']:raise RuntimeError('Evaluation CSV changed')
        if old.get('learning_sha256') and sha256(run_dir/'learning.csv')!=old['learning_sha256']:raise RuntimeError('Learning evaluation changed')
        return old
    rows=[]
    for scen,ecfg in scenarios(base,full).items():
        for enforce in (False,True):
            for ep in range(episodes):
                eval_seed=100000+ep  # independent of training seed; matched across methods
                env=Warehouse(ecfg,eval_seed)
                trace=None
                if scen=='nominal' and ep<2 and enforce:
                    trace=gzip.open(run_dir/f'trace_{scen}_shield_ep{ep}.jsonl.gz','wt',encoding='utf-8')
                try:result=one_episode(env,model=model,method=method,seed=eval_seed,enforce=enforce,trace_file=trace)
                finally:
                    if trace:trace.close()
                rows.append({'run_id':run_id,'method':method,'train_seed':seed,'layout':base.layout,
                    'consistency_coef':lam,'scenario':scen,'shield':int(enforce),'n_agents':ecfg.n_agents,
                    'eval_episode':ep,'eval_seed':eval_seed,**result})
    atomic_csv(run_dir/'evaluation.csv',rows)
    if model is not None:
        learning=[]
        for checkpoint in sorted(run_dir.glob('policy_*.pt'),key=lambda p:int(p.stem.split('_')[1])):
            train_steps=int(checkpoint.stem.split('_')[1])
            model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
            for ep in range(min(8,episodes)):
                env=Warehouse(base,200000+ep)
                measured=one_episode(env,model=model,seed=200000+ep,enforce=False)
                learning.append({'run_id':run_id,'method':method,'train_seed':seed,'layout':base.layout,
                    'env_steps':train_steps,'eval_episode':ep,'consistency_coef':lam,**measured})
        if learning:atomic_csv(run_dir/'learning.csv',learning)
    report={'spec':spec,'rows':len(rows),'csv_sha256':sha256(run_dir/'evaluation.csv'),
            'learning_sha256':sha256(run_dir/'learning.csv') if (run_dir/'learning.csv').exists() else None}
    atomic_json(complete,report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--episodes',type=int,default=24)
    p.add_argument('--small',action='store_true');a=p.parse_args()
    evaluate_run(a.run_dir,a.episodes,not a.small)
