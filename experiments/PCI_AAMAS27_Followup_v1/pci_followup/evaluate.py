import argparse,copy,gzip,json,time
from pathlib import Path
import numpy as np
import torch
from .model import Policy,energy
from .envs import make_env
from .io import atomic_json,atomic_csv,sha256,canonical_hash
from pci_bench.evaluate import jain

def episode(c,model,seed,enforce,trace=None):
    e=make_env(c,seed);rng=np.random.default_rng(seed+7000000);sums={};costs=[];energies=[];start=time.perf_counter()
    while e.t<e.config.horizon:
        f=torch.from_numpy(e.features()[None])
        with torch.no_grad():
            dist,_,_,p=model(f);v=dist.probs[0].numpy().astype(float)
            actions=np.array([rng.choice(5,p=row/row.sum()) for row in v])
            energies.append(float(energy(p,f,'pci_norm',model.random_action_maps).mean()))
        _,_,cost,_,m=e.transition(actions,enforce=enforce)
        for k,x in m.items():sums[k]=sums.get(k,0.)+x
        costs.append(float(cost.mean()))
        if trace:trace.write(json.dumps({'record':e.last,'metrics':m},separators=(',',':'))+'\n')
    return {'episode_len':e.config.horizon,'deliveries':sums['delivered'],
        'throughput_per_1000_steps':1000*sums['delivered']/e.config.horizon,
        'cost_rate':float(np.mean(costs)),'cost_budget':c['cost_budget'],
        'proposed_compliance':sums['proposed_compliant']/e.config.horizon,
        'executed_compliance':sums['executed_compliant']/e.config.horizon,
        'interventions':sums['interventions'],'collision_attempts':sums['collision_attempts'],
        'wall_attempts':sums['wall_attempts'],'quota_violations':sums['quota_violations'],
        'capacity_excess':sums['capacity_excess'],'actual_vertex_collisions':sums['actual_vertex_collisions'],
        'task_return':sums['task_reward'],'move_count':sums['move_count'],
        'max_wait_final':int(e.wait.max()),'per_agent_deliveries':json.dumps(e.delivered.tolist()),
        'jain_delivery':jain(e.delivered),'mean_semantic_energy':float(np.mean(energies)),
        'wall_seconds':time.perf_counter()-start}

def evaluate(out,episodes=24,small=False):
    out=Path(out);c=json.loads((out/'config.json').read_text());cp=out/'checkpoint.pt'
    spec={'checkpoint':sha256(cp),'episodes':episodes,'small':small,'eval_seed_base':410000}
    done=out/'EVAL_COMPLETE.json'
    if done.exists():
        d=json.loads(done.read_text())
        if d['spec']!=spec or d['csv_sha256']!=sha256(out/'evaluation.csv'):raise RuntimeError('Evaluation changed')
        return d
    torch.set_num_threads(1);model=Policy(c['method'],c['hidden']);ck=torch.load(cp,map_location='cpu',weights_only=False)
    model.load_state_dict(ck['model']);model.eval()
    scenarios={'nominal':{},'quota_tight':{'quota':1},'rule_shift':{'rule_change':True}}
    if not small:scenarios.update({'scale_2':{'n_agents':2},'scale_8':{'n_agents':8},'scale_12':{'n_agents':12}})
    else:scenarios={'nominal':{}}
    rows=[]
    for scen,change in scenarios.items():
        ec=copy.deepcopy(c);ec['env'].update(change)
        for shield in (False,True):
            for ep in range(episodes):
                seed=410000+ep
                path=out/f'trace_{scen}_shield{int(shield)}_ep{ep}.jsonl.gz'
                trace=gzip.open(path,'wt') if scen=='nominal' and ep<2 else None
                try:r=episode(ec,model,seed,shield,trace)
                finally:
                    if trace:trace.close()
                rows.append({'run_id':c['run_id'],'domain':c['domain'],'layout':c['env']['layout'],
                    'method':c['method'],'train_seed':c['seed'],'scenario':scen,'shield':int(shield),
                    'n_agents':ec['env']['n_agents'],'eval_episode':ep,'eval_seed':seed,**r})
    atomic_csv(out/'evaluation.csv',rows)
    learning=[]
    for path in sorted(out.glob('policy_*.pt'),key=lambda p:int(p.stem.split('_')[1])):
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        for ep in range(2 if small else 8):
            r=episode(c,model,510000+ep,False)
            learning.append({'run_id':c['run_id'],'domain':c['domain'],'layout':c['env']['layout'],'method':c['method'],
                'seed':c['seed'],'train_steps':int(path.stem.split('_')[1]),'eval_seed':510000+ep,**r})
    atomic_csv(out/'learning.csv',learning)
    d={'spec':spec,'csv_sha256':sha256(out/'evaluation.csv'),'learning_sha256':sha256(out/'learning.csv'),'episodes':len(rows)}
    atomic_json(done,d);return d

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--episodes',type=int,default=24);p.add_argument('--small',action='store_true')
    a=p.parse_args();evaluate(a.run_dir,a.episodes,a.small)
