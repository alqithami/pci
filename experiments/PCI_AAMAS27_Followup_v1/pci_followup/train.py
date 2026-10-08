"""Matched PPO and empirical-rate primal-dual PPO; no theorem of hard safety.

Separate reward and cost GAEs/critics are fitted for every method. The actor uses
reward pessimistic clipping plus multiplier times pessimistic cost clipping.
Dual ascent targets the specified empirical per-step cost rate. This is an
explicit discrete-action adaptation, NOT a reference MACPO reproduction.
"""
import argparse,json,copy,math,os,random,time
from pathlib import Path
import numpy as np
import torch
from .model import Policy,energy,dual_update
from .envs import make_env
from .io import atomic_csv,atomic_json,atomic_torch,immutable_json,signature,sha256,versions
from pci_bench.model import gae

def vector(model,actor=False):
    pars=(p for h in model.heads for p in h.parameters()) if actor else model.parameters()
    return torch.cat([p.detach().cpu().flatten() for p in pars])

def train(c,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);sig=signature(c)
    immutable_json(out/'config.json',c);done=out/'TRAIN_COMPLETE.json';checkpoint=out/'checkpoint.pt'
    if done.exists():
        d=json.loads(done.read_text())
        if d['signature']!=sig or sha256(checkpoint)!=d['checkpoint_sha256']:raise RuntimeError('Completed training changed')
        return d
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    seed=c['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    E=c['n_envs'];T=c['rollout_steps'];h=c['env']['horizon']
    if c['total_steps']%(E*T):raise ValueError('Unequal rollout budget')
    envs=[make_env(c,seed*10000+i) for i in range(E)]
    model=Policy(c['method'],c['hidden']);opt=torch.optim.Adam(model.parameters(),lr=c['learning_rate'],eps=1e-5)
    obs=np.stack([e.features() for e in envs]);updates=[];episodes=[];sums=[{} for _ in envs]
    step=0;start_update=0;elapsed0=0.;initial=vector(model);dual=float(c['penalty_coef'])
    if checkpoint.exists():
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if ck['signature']!=sig:raise RuntimeError('Checkpoint/source mismatch')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        for e,s in zip(envs,ck['envs']):e.load_state_dict(s)
        obs=ck['obs'];updates=ck['updates'];episodes=ck['episodes'];sums=ck['sums'];step=ck['step'];start_update=ck['update']
        initial=ck['initial'];dual=ck['dual'];elapsed0=ck['elapsed_s']
        torch.set_rng_state(ck['torch_rng']);np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng'])
    if not (out/'policy_0.pt').exists():
        if checkpoint.exists():raise RuntimeError('Initial policy artifact missing from resumed run')
        atomic_torch(out/'policy_0.pt',model.state_dict())
    started=time.perf_counter();nupdates=c['total_steps']//(E*T)
    def save(u):
        atomic_torch(checkpoint,{'signature':sig,'model':model.state_dict(),'optimizer':opt.state_dict(),
            'envs':[e.state_dict() for e in envs],'obs':obs,'updates':updates,'episodes':episodes,'sums':sums,
            'step':step,'update':u,'initial':initial,'dual':dual,'elapsed_s':elapsed0+time.perf_counter()-started,
            'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate()})
        if updates:atomic_csv(out/'updates.csv',updates)
        if episodes:atomic_csv(out/'train_episodes.csv',episodes)
    for u in range(start_update,nupdates):
        before=vector(model,True);buf={k:[] for k in ('obs','actions','logp','values','cvalues','rewards','costs','dones')}
        costs=[];task_rewards=[]
        multiplier=dual if c['method']=='ppo_lagrangian' else c['penalty_coef']
        for t in range(T):
            ob=torch.from_numpy(obs)
            with torch.no_grad():
                dist,v,cv,p=model(ob);a=dist.sample();lp=dist.log_prob(a)
            nextobs=[];rr=[];cc=[];dd=[]
            for i,e in enumerate(envs):
                no,task,cost,terminal,metrics=e.transition(a[i].numpy(),enforce=False)
                reward=float(task.mean());costrate=float(cost.mean());costs.append(costrate);task_rewards.append(reward)
                rr.append(np.full(e.n,reward,np.float32));cc.append(np.full(e.n,costrate,np.float32));dd.append(float(terminal))
                for k,z in metrics.items():sums[i][k]=sums[i].get(k,0.)+z
                sums[i]['cost_sum']=sums[i].get('cost_sum',0.)+costrate
                if terminal:
                    episodes.append({'run_id':c['run_id'],'domain':c['domain'],'layout':c['env']['layout'],'method':c['method'],
                        'train_seed':seed,'env_index':i,'episode':e.episode,'env_steps':step+i+1,
                        'episode_len':h,'deliveries':sums[i]['delivered'],'cost_rate':sums[i]['cost_sum']/h,
                        'proposed_compliance':sums[i]['proposed_compliant']/h,'task_return':sums[i]['task_reward']})
                    sums[i]={};no=e.reset()
                nextobs.append(no)
            for k,z in [('obs',ob),('actions',a),('logp',lp),('values',v),('cvalues',cv)]:buf[k].append(z)
            for k,z in [('rewards',rr),('costs',cc),('dones',dd)]:buf[k].append(torch.tensor(np.array(z),dtype=torch.float32))
            obs=np.stack(nextobs);step+=E
        b={k:torch.stack(z) for k,z in buf.items()}
        with torch.no_grad():_,lv,lcv,_=model(torch.from_numpy(obs))
        adv,ret=gae(b['rewards'],b['values'],b['dones'],lv,c['gamma'],c['gae_lambda'])
        cadv,cret=gae(b['costs'],b['cvalues'],b['dones'],lcv,c['gamma'],c['gae_lambda'])
        # Common reward scale for both advantages. Log all coefficients separately.
        scale=adv.std(unbiased=False)+1e-8
        adv=(adv-adv.mean())/scale;cadv=(cadv-cadv.mean())/scale
        adv=adv.flatten(0,1);cadv=cadv.flatten(0,1);ret=ret.flatten(0,1);cret=cret.flatten(0,1)
        flat={k:v.flatten(0,1) for k,v in b.items()};losses=[];grads=[];kls=[];energies=[];g_ratio=None;opt_steps=0
        for epoch in range(c['update_epochs']):
            perm=torch.randperm(E*T);stop=False
            for lo in range(0,E*T,c['minibatch_groups']):
                idx=perm[lo:lo+c['minibatch_groups']]
                dist,value,cvalue,p=model(flat['obs'][idx]);logratio=dist.log_prob(flat['actions'][idx])-flat['logp'][idx]
                ratio=logratio.exp();clip=ratio.clamp(1-c['clip_coef'],1+c['clip_coef'])
                rloss=torch.maximum(-adv[idx]*ratio,-adv[idx]*clip).mean()
                closs=torch.maximum(cadv[idx]*ratio,cadv[idx]*clip).mean()
                reg=energy(p,flat['obs'][idx],c['method'],model.random_action_maps).mean()
                objective=rloss+multiplier*closs
                if opt_steps==0:
                    ap=list(p for head in model.heads for p in head.parameters())
                    ag=torch.autograd.grad(objective,ap,retain_graph=True,allow_unused=True)
                    rg=torch.autograd.grad(c['consistency_coef']*reg,ap,retain_graph=True,allow_unused=True)
                    norm=lambda seq:math.sqrt(sum(float(x.square().sum()) for x in seq if x is not None))
                    an,rn=norm(ag),norm(rg);g_ratio=(an,rn,rn/max(an,1e-12))
                loss=objective+c['vf_coef']*.5*((value-ret[idx]).square().mean()+(cvalue-cret[idx]).square().mean())-c['entropy_coef']*dist.entropy().mean()+c['consistency_coef']*reg
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),c['max_grad_norm'])
                if not torch.isfinite(gn):raise RuntimeError('Nonfinite gradient')
                opt.step();opt_steps+=1
                kl=float(((ratio-1)-logratio).mean().detach());kls.append(kl);losses.append(float(loss.detach()));grads.append(float(gn));energies.append(float(reg.detach()))
                if kl>c['target_kl']:stop=True;break
            if stop:break
        if c['method']=='ppo_lagrangian':dual=dual_update(dual,float(np.mean(costs)),c['cost_budget'],c['dual_lr'],c['dual_max'])
        delta=float(torch.linalg.vector_norm(vector(model,True)-before))
        if delta<=0 or not np.isfinite(delta):raise AssertionError('Actor did not update')
        elapsed=elapsed0+time.perf_counter()-started
        updates.append({'run_id':c['run_id'],'domain':c['domain'],'layout':c['env']['layout'],'method':c['method'],'seed':seed,
            'update':u+1,'env_steps':step,'optimizer_steps':opt_steps,'actor_change_l2':delta,
            'mean_task_reward':float(np.mean(task_rewards)),'mean_violation_cost':float(np.mean(costs)),
            'cost_budget':c['cost_budget'],'multiplier_used':multiplier,'multiplier_next':dual,
            'regularizer_mean':float(np.mean(energies)),'actor_objective_grad_norm':g_ratio[0],
            'regularizer_grad_norm':g_ratio[1],'regularizer_to_actor_grad_ratio':g_ratio[2],
            'loss':float(np.mean(losses)),'grad_norm':float(np.mean(grads)),'kl':float(np.mean(kls)),
            'elapsed_s':elapsed,'steps_per_s':step/max(elapsed,1e-9)})
        if u+1 in {max(1,nupdates//4),max(1,nupdates//2),max(1,3*nupdates//4),nupdates}:
            atomic_torch(out/f'policy_{step}.pt',model.state_dict())
        if (u+1)%8==0 or u+1==nupdates:
            save(u+1);print(f"{c['run_id']} steps={step}/{c['total_steps']} rate={step/max(elapsed,1e-9):.1f} lambda={dual:.5f}",flush=True)
    d={'signature':sig,'env_steps':step,'updates':nupdates,'checkpoint_sha256':sha256(checkpoint),
        'parameter_change_l2':float(torch.linalg.vector_norm(vector(model)-initial)),
        'elapsed_s':elapsed0+time.perf_counter()-started,'parameters':sum(p.numel() for p in model.parameters())}
    atomic_json(done,d);return d

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();train(json.loads(Path(a.config).read_text()),a.out)
