from __future__ import annotations
import argparse, copy, json, math, os, random, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .environment import EnvConfig, Warehouse
from .model import ContextPolicy, compatibility_energy, gae
from .io import atomic_json, atomic_csv, canonical_hash, sha256, source_hashes, system_meta


def model_vector(model):return torch.cat([p.detach().flatten().cpu() for p in model.parameters()])

def actor_vector(model):return torch.cat([p.detach().flatten().cpu() for h in model.heads for p in h.parameters()])


def train(config:dict,out:Path):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    signature=canonical_hash({'config':config,'sources':source_hashes(root),'versions':system_meta()['versions']})
    done_path=out/'TRAIN_COMPLETE.json'
    if done_path.exists():
        done=json.loads(done_path.read_text())
        if done['signature']!=signature:raise RuntimeError(f'Configuration/source changed for completed run {out}')
        if sha256(out/'checkpoint.pt')!=done['checkpoint_sha256']:raise RuntimeError('Checkpoint changed')
        return done
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=config:
        raise RuntimeError(f'Refusing to mix configs in {out}')
    atomic_json(out/'config.json',config)
    torch.set_num_threads(int(config.get('torch_threads',1)))
    torch.use_deterministic_algorithms(True)
    seed=int(config['seed']);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    device=torch.device(config.get('device','cpu'))
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    env_cfg=EnvConfig(**config['env'])
    nenv=int(config['n_envs']);rollout=int(config['rollout_steps'])
    if config['total_steps']%(nenv*rollout):raise ValueError('total_steps must be divisible by n_envs*rollout_steps')
    envs=[Warehouse(env_cfg,seed*10000+i) for i in range(nenv)]
    model=ContextPolicy(config['method'],config.get('hidden',64)).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],eps=1e-5)
    obs=np.stack([e.features() for e in envs])
    updates=[];episode_rows=[];env_sums=[{} for _ in envs];step=0;elapsed_before=0.;start_update=0
    initial=model_vector(model).clone()
    checkpoint=out/'checkpoint.pt'
    if checkpoint.exists():
        ck=torch.load(checkpoint,map_location=device,weights_only=False)
        if ck['signature']!=signature:raise RuntimeError(f'Cannot resume changed code/config: {out}')
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
        for e,st in zip(envs,ck['envs']):e.load_state_dict(st)
        obs=ck['obs'];step=ck['step'];updates=ck['updates'];episode_rows=ck['episodes'];env_sums=ck['env_sums']
        initial=ck['initial_vector'].cpu();elapsed_before=ck['elapsed'];start_update=ck['update']
        torch.set_rng_state(ck['torch_rng'].cpu());np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng'])
        if device.type=='cuda' and ck.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(ck['cuda_rng'])
    if not (out/'policy_0.pt').exists():torch.save(model.state_dict(),out/'policy_0.pt')
    start=time.perf_counter()
    atomic_json(out/'provenance.json',{'signature':signature,'system':system_meta(),'sources':source_hashes(root),
                                      'semantic_version':'PCI-Warehouse-1.0','crypto_during_training':False})
    def save(update):
        state={'signature':signature,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
               'envs':[e.state_dict() for e in envs],'obs':obs,'step':step,'update':update,
               'updates':updates,'episodes':episode_rows,'env_sums':env_sums,'initial_vector':initial,
               'elapsed':elapsed_before+time.perf_counter()-start,'torch_rng':torch.get_rng_state(),
               'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
               'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None}
        tmp=checkpoint.with_suffix('.tmp');torch.save(state,tmp);os.replace(tmp,checkpoint)
        if updates:atomic_csv(out/'updates.csv',updates)
        if episode_rows:atomic_csv(out/'train_episodes.csv',episode_rows)
    nupdates=config['total_steps']//(nenv*rollout)
    for update in range(start_update,nupdates):
        before=model_vector(model);actor_before=actor_vector(model)
        data={k:[] for k in ('obs','actions','logp','values','rewards','dones')}
        rollout_energy=[];rollout_cost=[];rollout_tasks=[]
        for t in range(rollout):
            ob=torch.as_tensor(obs,device=device)
            with torch.no_grad():
                dist,values,probs=model(ob)
                actions=dist.sample();logp=dist.log_prob(actions)
                energy=compatibility_energy(probs,ob)
            batch_actions=actions.cpu().numpy()
            next_obs=[];rewards=[];dones=[]
            for i,e in enumerate(envs):
                no,task,cost,done,metrics=e.transition(batch_actions[i],enforce=False)
                penalty=0. if config['method']=='ppo_task' else config['penalty_coef']
                # Shared cooperative reward; log sparse deliveries separately from shaping.
                r=float(np.mean(task-penalty*cost))
                rewards.append(np.full(e.n,r,dtype=np.float32));dones.append(float(done))
                sums=env_sums[i]
                for k,v in metrics.items():sums[k]=sums.get(k,0)+v
                sums['energy']=sums.get('energy',0)+float(energy[i].mean().cpu())
                if done:
                    episode_rows.append({'run_id':config['run_id'],'seed':seed,'method':config['method'],
                        'layout':env_cfg.layout,'n_agents':env_cfg.n_agents,'env_id':i,'episode':e.episode,
                        'env_steps_at_end':step+i+1,'episode_len':env_cfg.horizon,
                        'deliveries':sums['delivered'],'pickups':sums['pickups'],
                        'task_return':sums['task_reward'],'proposed_compliance':sums['proposed_compliant']/env_cfg.horizon,
                        'executed_compliance':sums['executed_compliant']/env_cfg.horizon,
                        'collision_attempts':sums['collision_attempts'],'quota_violations':sums['quota_violations'],
                        'capacity_excess':sums['capacity_excess'],'mean_consistency_energy':sums['energy']/env_cfg.horizon})
                    env_sums[i]={};no=e.reset()
                next_obs.append(no)
                rollout_cost.append(float(cost.mean()));rollout_tasks.append(float(task.mean()))
            for key,val in [('obs',ob),('actions',actions),('logp',logp),('values',values)]:data[key].append(val)
            data['rewards'].append(torch.tensor(np.array(rewards),device=device))
            data['dones'].append(torch.tensor(dones,device=device))
            rollout_energy.append(float(energy.mean().cpu()))
            obs=np.stack(next_obs);step+=nenv
        batch={k:torch.stack(v) for k,v in data.items()}
        with torch.no_grad():last=model(torch.as_tensor(obs,device=device))[1]
        adv,returns=gae(batch['rewards'],batch['values'],batch['dones'],last,config['gamma'],config['gae_lambda'])
        flat={k:v.flatten(0,1) for k,v in batch.items()}
        advantage=adv.flatten(0,1);ret=returns.flatten(0,1)
        advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
        n_groups=rollout*nenv
        losses=[];gradients=[];kl_values=[];opt_steps=0
        for epoch in range(config['update_epochs']):
            permutation=torch.randperm(n_groups,device=device)
            stop=False
            for begin in range(0,n_groups,config['minibatch_groups']):
                idx=permutation[begin:begin+config['minibatch_groups']]
                dist,value,probs=model(flat['obs'][idx])
                logratio=dist.log_prob(flat['actions'][idx])-flat['logp'][idx]
                ratio=logratio.exp()
                unclipped=-advantage[idx]*ratio
                clipped=-advantage[idx]*ratio.clamp(1-config['clip_coef'],1+config['clip_coef'])
                actor=torch.maximum(unclipped,clipped).mean()
                critic=0.5*(value-ret[idx]).square().mean()
                entropy=dist.entropy().mean()
                mode=config['method']
                regularizer=compatibility_energy(probs,flat['obs'][idx],mode=mode).mean() if mode in ('pci','pci_shuffled','consensus') else actor.new_tensor(0.)
                loss=actor+config['vf_coef']*critic-config['entropy_coef']*entropy+config['consistency_coef']*regularizer
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),config['max_grad_norm'])
                if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradient')
                optimizer.step();opt_steps+=1
                kl=float(((ratio-1)-logratio).mean().detach().cpu())
                losses.append(float(loss.detach().cpu()));gradients.append(float(grad.detach().cpu()));kl_values.append(kl)
                if kl>config['target_kl']:stop=True;break
            if stop:break
        delta=float(torch.linalg.vector_norm(model_vector(model)-before))
        actor_delta=float(torch.linalg.vector_norm(actor_vector(model)-actor_before))
        if delta==0 or actor_delta==0 or opt_steps==0:raise AssertionError('Policy did not update')
        elapsed=elapsed_before+time.perf_counter()-start
        updates.append({'run_id':config['run_id'],'seed':seed,'method':config['method'],'layout':env_cfg.layout,
            'update':update+1,'env_steps':step,'agent_steps':step*env_cfg.n_agents,
            'mean_task_reward':float(np.mean(rollout_tasks)),'mean_violation_cost':float(np.mean(rollout_cost)),
            'mean_consistency_energy':float(np.mean(rollout_energy)),'optimization_loss':float(np.mean(losses)),
            'mean_grad_norm':float(np.mean(gradients)),'parameter_change_l2':delta,'actor_change_l2':actor_delta,'optimizer_steps':opt_steps,
            'approx_kl':float(np.mean(kl_values)),'elapsed_s':elapsed,'env_steps_per_s':step/max(elapsed,1e-9)})
        if (update+1) in {max(1,nupdates//4),max(1,nupdates//2),max(1,3*nupdates//4),nupdates}:
            temp=out/f'policy_{step}.tmp';torch.save(model.state_dict(),temp);os.replace(temp,out/f'policy_{step}.pt')
        if (update+1)%8==0 or update+1==nupdates:
            save(update+1)
            print(f"{config['run_id']} {step}/{config['total_steps']} env steps; "
                  f"{step/max(elapsed,1e-9):.1f} steps/s; updates={update+1}; delta={delta:.4g}",flush=True)
    total_delta=float(torch.linalg.vector_norm(model_vector(model)-initial))
    if total_delta==0:raise AssertionError('Final model unchanged')
    done={'signature':signature,'env_steps':step,'updates':nupdates,'parameter_change_l2':total_delta,
          'checkpoint_sha256':sha256(checkpoint),'elapsed_s':elapsed_before+time.perf_counter()-start}
    atomic_json(done_path,done);return done

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();train(json.loads(Path(a.config).read_text()),Path(a.out))
