"""Separately versioned matched institutional-risk PPO experiment runtime.

Derived from the retained PCI custom-world train.py (SHA d3f57bab217853fcd9e2e193e8feaf64aa125bb5874e7eb2ca7ec6c83be082f4).
The PPO clipping, critic, entropy, GAE, minibatch grouping, optimizer and KL stop
are retained. Changes are explicit: common admission, balanced fixed profiles,
institutional-only sampled penalties, bounded auxiliary losses, separate sample
and shuffle generators, and closed retention. This is not the old experiment.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, time
from pathlib import Path
import numpy as np
import torch
from .environment import EnvConfig, Warehouse
from .model import ContextPolicy, compatibility_energy, gae
from .objectives import METHODS, MODEL_METHOD, auxiliary, institutional_risk
from .io import (atomic_json, atomic_csv, atomic_npz, atomic_torch, canonical_hash,
                 sha256, source_hashes, system_meta)
from ..vectorized_admission import (institutional_replacement_counts, _validate_state,
                                      _institutional_conflict_possible)

VERSION = 'PCI-RISK-MATCHED-PPO-v1'
ORIGINAL_TRAIN_SHA256 = 'd3f57bab217853fcd9e2e193e8feaf64aa125bb5874e7eb2ca7ec6c83be082f4'
ENV_SHA256 = 'cc2ba5ad437b517f9aa3ab96119ce3fa3dba2c51098a7ff2d77ef6977f3e059c'
MODEL_SHA256 = 'e9a7ab8721d457b60f54fe577ab03841390a57af5bd85f55e83c9f0bbefa2665'
PROFILES = {
 'nominal':dict(layout='twodoors',n_agents=4,horizon=256,quota_window=64,quota=3,capacity=2,rule_change=False),
 'quota_tight':dict(layout='twodoors',n_agents=4,horizon=256,quota_window=64,quota=1,capacity=2,rule_change=False),
 'capacity_tight':dict(layout='twodoors',n_agents=4,horizon=256,quota_window=64,quota=3,capacity=1,rule_change=False),
 'rule_shift':dict(layout='twodoors',n_agents=4,horizon=256,quota_window=64,quota=3,capacity=1,rule_change=True),
}
PROFILE_IDS = tuple(PROFILES)
ENV_PROFILE_IDS = tuple(p for p in PROFILE_IDS for _ in range(2))


def model_vector(model):
    return torch.cat([p.detach().flatten().cpu() for p in model.parameters()])


def actor_vector(model):
    return torch.cat([p.detach().flatten().cpu() for h in model.heads for p in h.parameters()])


def tensor_digest(tensor):
    a=tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


def validate_config(config):
    if config['method'] not in METHODS: raise ValueError('Unknown matched arm')
    if config.get('run_role') not in ('scientific','readiness_fixture'):
        raise ValueError('Explicit scientific or readiness_fixture role required')
    if config['n_envs']!=8 or config['rollout_steps']!=128 or config.get('hidden',64)!=64:
        raise ValueError('Exact original matched architecture/rollout required')
    if tuple(config['env_profile_ids'])!=ENV_PROFILE_IDS:
        raise ValueError('Two fixed slots for every prespecified profile required')
    if 'profiles' in config and config['profiles']!=PROFILES:
        raise ValueError('Profile parameters differ from this runtime')
    if len(config['env_seeds'])!=8 or len(set(config['env_seeds']))!=8:
        raise ValueError('Eight distinct explicit environment streams required')
    for key in ('seed','init_seed','policy_seed','minibatch_seed'):
        if type(config[key]) is not int or config[key]<0: raise ValueError('Invalid explicit seed')
    if config['seed']!=config['init_seed']: raise ValueError('seed is the initialization cluster label')
    if 2201<=config['seed']<=2230 or any(930000<=s<940000 for s in config['env_seeds']):
        raise ValueError('Reserved Stage B namespace must remain untouched')
    if config['total_steps']<=0 or config['total_steps']%(8*128):
        raise ValueError('Complete original PPO rollout blocks required')
    if config['run_role']=='scientific' and config['total_steps']!=262144:
        raise ValueError('Scientific budget must be the declared 262144 joint steps')
    if config.get('torch_threads',1)!=1: raise ValueError('One native math thread required')
    if config.get('checkpoint_every_updates',8)<1: raise ValueError('Invalid checkpoint interval')
    expected={'penalty_coef':0.25,'consistency_coef':0.2,'dual_initial':0.25,
              'dual_lr':0.05,'dual_target':0.01,'dual_max':5.0}
    if any(float(config[k])!=v for k,v in expected.items()):
        raise ValueError('Prespecified objective settings differ')


def _closed_rollout(path, arrays):
    # An orphan from an interrupted update may be reused ONLY when every array
    # is identical after restoring the exact prior checkpoint. It is not dropped
    # or replaced by a different rollout. Corruption or divergent replay fails.
    return atomic_npz(path,arrays,verify_existing=True)


def train(config:dict,out:Path,*,stop_after_updates=None):
    validate_config(config)
    if stop_after_updates is not None and config['run_role']!='readiness_fixture':
        raise ValueError('Early update stop is restricted to implementation fixtures')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    if sha256(Path(__file__).with_name('environment.py'))!=ENV_SHA256 or sha256(Path(__file__).with_name('model.py'))!=MODEL_SHA256:
        raise RuntimeError('Original environment/model source body changed')
    sources=source_hashes(root)
    signature=canonical_hash({'config':config,'sources':sources,'versions':system_meta()['versions']})
    done_path=out/'TRAIN_COMPLETE.json'
    if done_path.exists():
        done=json.loads(done_path.read_text())
        if done['signature']!=signature: raise RuntimeError('Configuration/source changed for completed run')
        for k,n in [('checkpoint_sha256','checkpoint.pt'),('portable_export_sha256','final_export.npz')]:
            if sha256(out/n)!=done[k]: raise RuntimeError('Completed artifact changed: '+n)
        if sha256(out/'CLOSED_FILES.json')!=done['closed_manifest_sha256']:
            raise RuntimeError('Completed closed manifest body changed')
        manifest=json.loads((out/'CLOSED_FILES.json').read_text())
        expected_rollouts={f'rollouts/update_{i:06}.npz' for i in range(1,done['updates']+1)}
        required={'config.json','provenance.json','checkpoint.pt','policy_0.pt',
                  'updates.csv','LAST_COMMIT.json','final_export.npz'}
        if (manifest.get('schema')!='PCI-RISK-TRAIN-CLOSED-FILES-v1' or
            manifest.get('signature')!=signature or not required<=set(manifest['files']) or
            {n for n in manifest['files'] if n.startswith('rollouts/')}!=expected_rollouts):
            raise RuntimeError('Completed manifest closure/schema differs')
        for n,v in manifest['files'].items():
            if Path(n).is_absolute() or '..' in Path(n).parts:
                raise RuntimeError('Nonlocal retained member path')
            if sha256(out/n)!=v['sha256'] or (out/n).stat().st_size!=v['bytes']:
                raise RuntimeError('Completed retained body changed: '+n)
        return done
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=config:
        raise RuntimeError('Refusing to mix configurations')
    atomic_json(out/'config.json',config)
    checkpoint=out/'checkpoint.pt';resume_commit=None
    if checkpoint.exists():
        if not (out/'LAST_COMMIT.json').is_file():
            raise RuntimeError('Checkpoint has no closed commit receipt')
        resume_commit=json.loads((out/'LAST_COMMIT.json').read_text())
        if (resume_commit.get('signature')!=signature or
                sha256(checkpoint)!=resume_commit.get('checkpoint_sha256')):
            raise RuntimeError('Checkpoint/commit body identity differs')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    # Initialize on CPU identically across arms, then move the unchanged network.
    seed=int(config['init_seed']);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    device=torch.device(config.get('device','cpu'))
    if device.type=='cuda':
        if not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8',':16:8'):
            raise RuntimeError('Deterministic CUBLAS workspace must be set before CUDA use')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
    nenv=int(config['n_envs']);rollout=int(config['rollout_steps'])
    envs=[Warehouse(EnvConfig(**PROFILES[p]),int(s))
          for p,s in zip(config['env_profile_ids'],config['env_seeds'])]
    model=ContextPolicy(MODEL_METHOD[config['method']],config.get('hidden',64)).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],eps=1e-5)
    policy_generator=torch.Generator(device=device).manual_seed(config['policy_seed'])
    minibatch_generator=torch.Generator(device=device).manual_seed(config['minibatch_seed'])
    obs=np.stack([e.features() for e in envs])
    updates=[];episode_rows=[];env_sums=[{} for _ in envs]
    step=0;elapsed_before=0.;start_update=0
    initial=model_vector(model).clone();initial_actor=actor_vector(model).clone()
    dual=float(config['dual_initial']);closed_rollouts={}
    counters={'oracle_states':0,'oracle_candidate_rows':0,'zero_cost_tables':0,
              'sampled_native_oracle_equal':0,'proposal_corrected_differences':0,
              'policy_samples':0,'optimizer_steps':0,'zero_shortcut_states':0,
              'certificate_audit_queries':0}
    phase_counts=np.zeros((4,64),dtype=np.int64)
    action_digest='0'*64
    if checkpoint.exists():
        ck=torch.load(checkpoint,map_location=device,weights_only=False)
        if (ck['signature']!=signature or ck['update']!=resume_commit['update'] or
            ck['step']!=resume_commit['env_steps'] or ck['action_digest']!=resume_commit['action_digest']):
            raise RuntimeError('Cannot resume changed code/config or mismatched commit state')
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
        for e,st in zip(envs,ck['envs']): e.load_state_dict(st)
        obs=ck['obs'];step=ck['step'];updates=ck['updates'];episode_rows=ck['episodes'];env_sums=ck['env_sums']
        initial=ck['initial_vector'].cpu();initial_actor=ck['initial_actor_vector'].cpu()
        elapsed_before=ck['elapsed'];start_update=ck['update'];dual=ck['dual'];counters=ck['counters']
        phase_counts=ck['phase_counts'];action_digest=ck['action_digest'];closed_rollouts=ck['closed_rollouts']
        for n,v in closed_rollouts.items():
            if sha256(out/n)!=v: raise RuntimeError('Committed rollout body changed: '+n)
        torch.set_rng_state(ck['torch_rng'].cpu());np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng'])
        policy_generator.set_state(ck['policy_generator_state'].cpu())
        minibatch_generator.set_state(ck['minibatch_generator_state'].cpu())
        if device.type=='cuda' and ck.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in ck['cuda_rng']])
    if not (out/'policy_0.pt').exists(): atomic_torch(out/'policy_0.pt',model.state_dict())
    start=time.perf_counter()
    atomic_json(out/'provenance.json',{'signature':signature,'system':system_meta(),'sources':sources,
        'semantic_version':VERSION,'original_train_sha256':ORIGINAL_TRAIN_SHA256,
        'crypto_during_training':False,'common_admission':True,
        'oracle_table_scope':'625 exact logical cost rows returned at every visited state in every arm; zero-certified rows need no physical enumeration',
        'counter_semantics':{'oracle_candidate_rows':'Logical cost-table rows, not native/scalar calls',
          'zero_cost_tables':'All-zero output tables, possibly beyond shortcut-certified cases',
          'zero_shortcut_states':'Exact conservative zero certificate, independently queried once again for accounting',
          'certificate_audit_queries':'Additional read-only certificate queries for counting; one per state'},
        'seed_roles':'CPU initialization; native environment streams; separate proposal and shuffle Torch generators',
        'new_experiment_not_reclassified_old_training':True})
    def save(update):
        state={'signature':signature,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'envs':[e.state_dict() for e in envs],'obs':obs,'step':step,'update':update,
            'updates':updates,'episodes':episode_rows,'env_sums':env_sums,
            'initial_vector':initial,'initial_actor_vector':initial_actor,
            'elapsed':elapsed_before+time.perf_counter()-start,'torch_rng':torch.get_rng_state(),
            'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
            'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
            'policy_generator_state':policy_generator.get_state(),
            'minibatch_generator_state':minibatch_generator.get_state(),
            'dual':dual,'counters':counters,'phase_counts':phase_counts,
            'action_digest':action_digest,'closed_rollouts':closed_rollouts}
        atomic_torch(checkpoint,state)
        if updates: atomic_csv(out/'updates.csv',updates)
        if episode_rows: atomic_csv(out/'train_episodes.csv',episode_rows)
        atomic_json(out/'LAST_COMMIT.json',{'signature':signature,'update':update,'env_steps':step,
            'checkpoint_sha256':sha256(checkpoint),'closed_rollouts':len(closed_rollouts),
            'action_digest':action_digest})
    nupdates=config['total_steps']//(nenv*rollout)
    for update in range(start_update,nupdates):
        before=model_vector(model);actor_before=actor_vector(model)
        data={k:[] for k in ('obs','actions','logp','values','rewards','dones','raw_costs')}
        records={k:[] for k in ('admitted','physical_replacements','institutional_replacements',
                               'total_replacements','task_rewards','steps')}
        rollout_energy=[];rollout_cost=[];rollout_tasks=[];rollout_risks=[]
        # Dual is frozen throughout this rollout; updated once for the next one.
        penalty=(0. if config['method']=='task' else dual if config['method']=='adaptive' else config['penalty_coef'])
        for t in range(rollout):
            shortcut_flags=[not _institutional_conflict_possible(_validate_state(e)) for e in envs]
            tables=np.stack([institutional_replacement_counts(e) for e in envs])
            if any(flag and row.any() for flag,row in zip(shortcut_flags,tables)):
                raise RuntimeError('Zero-certificate accounting differs from oracle')
            if tables.shape!=(nenv,625) or np.any((tables<0)|(tables>4)):
                raise RuntimeError('Invalid RAW oracle table')
            raw_costs=torch.as_tensor(tables.astype(np.uint8),device=device)
            ob=torch.as_tensor(obs,device=device)
            with torch.no_grad():
                dist,values,probs=model(ob)
                actions=torch.multinomial(dist.probs.reshape(-1,5),1,
                    generator=policy_generator).reshape(nenv,4)
                logp=dist.log_prob(actions)
                energy=compatibility_energy(probs,ob)
                risks=institutional_risk(dist.probs,raw_costs)
            batch_actions=actions.cpu().numpy()
            admitted=[];physical_counts=[];inst_counts=[];total_counts=[];tasks=[];steps=[]
            next_obs=[];rewards=[];dones=[]
            for i,e in enumerate(envs):
                pidx=PROFILE_IDS.index(config['env_profile_ids'][i]);phase_counts[pidx,e.t%64]+=1
                steps.append(e.t)
                candidate,_=e.candidates(batch_actions[i]);physical,_=e.physical_resolve(candidate)
                physical_actions=np.asarray(e.actions_for_positions(physical))
                no,task,_old_composite_cost,done,metrics=e.transition(batch_actions[i],enforce=True)
                applied=np.asarray(e.last['actions'],dtype=np.int64)
                physical_count=int(np.count_nonzero(batch_actions[i]!=physical_actions))
                inst_count=int(np.count_nonzero(physical_actions!=applied))
                total_count=int(np.count_nonzero(batch_actions[i]!=applied))
                code=int(batch_actions[i] @ np.asarray((125,25,5,1)))
                if inst_count!=int(tables[i,code]):
                    raise RuntimeError('Live native sampled cost differs from static oracle')
                admitted.append(applied);physical_counts.append(physical_count)
                inst_counts.append(inst_count);total_counts.append(total_count);tasks.append(task)
                # Same shared cooperative task reward and institution-only stage cost.
                r=float(np.mean(task)-penalty*inst_count/4.)
                rewards.append(np.full(e.n,r,dtype=np.float32));dones.append(float(done))
                sums=env_sums[i]
                for k,v in metrics.items(): sums[k]=sums.get(k,0)+v
                for k,v in [('physical_replacements',physical_count),('institutional_replacements',inst_count),
                            ('total_replacements',total_count),('energy',float(energy[i].mean().cpu())),
                            ('institutional_risk',float(risks[i].cpu()))]: sums[k]=sums.get(k,0)+v
                if done:
                    episode_rows.append({'run_id':config['run_id'],'seed':config['seed'],'method':config['method'],
                        'profile':config['env_profile_ids'][i],'env_id':i,'episode':e.episode,
                        'env_steps_at_end':step+i+1,'episode_len':e.config.horizon,
                        'deliveries':sums['delivered'],'pickups':sums['pickups'],'task_return':sums['task_reward'],
                        'proposed_compliance':sums['proposed_compliant']/e.config.horizon,
                        'executed_compliance':sums['executed_compliant']/e.config.horizon,
                        'physical_replacements':sums['physical_replacements'],
                        'institutional_replacements':sums['institutional_replacements'],
                        'total_replacements':sums['total_replacements'],
                        'mean_raw_semantic_energy':sums['energy']/e.config.horizon,
                        'mean_institutional_risk':sums['institutional_risk']/e.config.horizon})
                    env_sums[i]={};no=e.reset()
                next_obs.append(no);rollout_cost.append(inst_count/4.);rollout_tasks.append(float(task.mean()))
            for key,val in [('obs',ob),('actions',actions),('logp',logp),('values',values),('raw_costs',raw_costs)]:
                data[key].append(val)
            data['rewards'].append(torch.tensor(np.array(rewards),device=device))
            data['dones'].append(torch.tensor(dones,device=device))
            for key,val in [('admitted',admitted),('physical_replacements',physical_counts),
                            ('institutional_replacements',inst_counts),('total_replacements',total_counts),
                            ('task_rewards',tasks),('steps',steps)]: records[key].append(np.asarray(val))
            action_digest=hashlib.sha256(bytes.fromhex(action_digest)+batch_actions.astype('<i8').tobytes()+
                logp.cpu().numpy().astype('<f4').tobytes()+np.asarray(inst_counts,dtype='<i8').tobytes()).hexdigest()
            counters['oracle_states']+=nenv;counters['oracle_candidate_rows']+=nenv*625
            counters['zero_cost_tables']+=int(np.count_nonzero(~tables.any(axis=1)))
            counters['zero_shortcut_states']+=sum(shortcut_flags)
            counters['certificate_audit_queries']+=nenv
            counters['sampled_native_oracle_equal']+=nenv;counters['policy_samples']+=nenv*4
            counters['proposal_corrected_differences']+=sum(total_counts)
            rollout_energy.append(float(energy.mean().cpu()));rollout_risks.append(float(risks.mean().cpu()))
            obs=np.stack(next_obs);step+=nenv
        batch={k:torch.stack(v) for k,v in data.items()}
        with torch.no_grad(): last=model(torch.as_tensor(obs,device=device))[1]
        adv,returns=gae(batch['rewards'],batch['values'],batch['dones'],last,config['gamma'],config['gae_lambda'])
        # Flatten TIME and ENVIRONMENT only. The four-agent joint-state group is intact.
        flat={k:v.flatten(0,1) for k,v in batch.items()}
        advantage=adv.flatten(0,1);ret=returns.flatten(0,1)
        advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
        n_groups=rollout*nenv
        losses=[];gradients=[];kl_values=[];aux_values=[];opt_steps=0
        for epoch in range(config['update_epochs']):
            permutation=torch.randperm(n_groups,device=device,generator=minibatch_generator)
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
                regularizer=auxiliary(config['method'],dist,probs,flat['obs'][idx],flat['raw_costs'][idx])
                loss=actor+config['vf_coef']*critic-config['entropy_coef']*entropy+config['consistency_coef']*regularizer
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),config['max_grad_norm'])
                if not torch.isfinite(grad): raise RuntimeError('Nonfinite gradient')
                optimizer.step();opt_steps+=1
                kl=float(((ratio-1)-logratio).mean().detach().cpu())
                losses.append(float(loss.detach().cpu()));gradients.append(float(grad.detach().cpu()))
                kl_values.append(kl);aux_values.append(float(regularizer.detach().cpu()))
                if kl>config['target_kl']: stop=True;break
            if stop: break
        counters['optimizer_steps']+=opt_steps
        delta=float(torch.linalg.vector_norm(model_vector(model)-before))
        actor_delta=float(torch.linalg.vector_norm(actor_vector(model)-actor_before))
        if delta==0 or actor_delta==0 or opt_steps==0: raise AssertionError('Policy did not update')
        # All arm objectives use the same sampled institutional stage cost.
        if config['method']=='adaptive':
            dual=float(np.clip(dual+config['dual_lr']*(float(np.mean(rollout_cost))-config['dual_target']),0.,config['dual_max']))
        elapsed=elapsed_before+time.perf_counter()-start
        updates.append({'run_id':config['run_id'],'seed':config['seed'],'method':config['method'],
            'update':update+1,'env_steps':step,'agent_steps':step*4,
            'mean_task_reward':float(np.mean(rollout_tasks)),
            'mean_institutional_cost':float(np.mean(rollout_cost)),
            'mean_predicted_institutional_risk':float(np.mean(rollout_risks)),
            'mean_raw_semantic_energy':float(np.mean(rollout_energy)),
            'mean_auxiliary_loss':float(np.mean(aux_values)),'optimization_loss':float(np.mean(losses)),
            'penalty_this_rollout':float(penalty),'dual_next_rollout':dual,
            'mean_grad_norm':float(np.mean(gradients)),'parameter_change_l2':delta,
            'actor_change_l2':actor_delta,'optimizer_steps':opt_steps,'approx_kl':float(np.mean(kl_values)),
            'elapsed_s':elapsed,'env_steps_per_s':step/max(elapsed,1e-9)})
        arrays={k:v.detach().cpu().numpy() for k,v in batch.items()}
        arrays.update({k:np.stack(v) for k,v in records.items()})
        arrays.update(advantages=adv.detach().cpu().numpy(),returns=returns.detach().cpu().numpy(),
                      bootstrap_values=last.detach().cpu().numpy(),penalty=np.asarray(penalty))
        name=f'rollouts/update_{update+1:06}.npz'
        closed_rollouts[name]=_closed_rollout(out/name,arrays)
        if (update+1) in {max(1,nupdates//4),max(1,nupdates//2),max(1,3*nupdates//4),nupdates}:
            atomic_torch(out/f'policy_{step}.pt',model.state_dict())
        stopping=stop_after_updates is not None and update+1>=stop_after_updates
        if (update+1)%config.get('checkpoint_every_updates',8)==0 or update+1==nupdates or stopping:
            save(update+1)
            print(f"{config['run_id']} {step}/{config['total_steps']} joint steps; {step/max(elapsed,1e-9):.1f}/s; updates={update+1}",flush=True)
        if stopping and update+1<nupdates:
            paused={'status':'PAUSED_READINESS_FIXTURE','signature':signature,'update':update+1,
                    'env_steps':step,'checkpoint_sha256':sha256(checkpoint)}
            atomic_json(out/'PAUSED.json',paused);return paused
    if step!=config['total_steps'] or counters['oracle_states']!=step or counters['sampled_native_oracle_equal']!=step:
        raise RuntimeError('Incomplete training or oracle accounting')
    if (phase_counts.shape!=(4,64) or int(phase_counts.sum())!=step or
            not np.all(phase_counts==step//(4*64))):
        raise RuntimeError('Complete equal-profile/window-phase accounting differs')
    if len(episode_rows)!=8*(step//(8*256)):
        raise RuntimeError('Complete episode accounting differs')
    total_delta=float(torch.linalg.vector_norm(model_vector(model)-initial))
    if total_delta==0: raise AssertionError('Final model unchanged')
    portable={k:v.detach().cpu().numpy() for k,v in model.state_dict().items()}
    export_hash=atomic_npz(out/'final_export.npz',portable,verify_existing=True)
    files={}
    for f in sorted(out.rglob('*')):
        if f.is_file() and f.name not in ('TRAIN_COMPLETE.json','CLOSED_FILES.json','PAUSED.json') and not f.name.endswith('.tmp'):
            files[f.relative_to(out).as_posix()]={'bytes':f.stat().st_size,'sha256':sha256(f)}
    manifest={'schema':'PCI-RISK-TRAIN-CLOSED-FILES-v1','signature':signature,'files':files}
    atomic_json(out/'CLOSED_FILES.json',manifest)
    done={'schema':'PCI-RISK-TRAIN-COMPLETE-v1','status':'COMPLETE','run_role':config['run_role'],
        'method':config['method'],'seed':config['seed'],'run_id':config['run_id'],
        'signature':signature,'config_sha256':canonical_hash(config),'sources':sources,
        'protocol_sha256':config.get('protocol_sha256'),'env_steps':step,'updates':nupdates,
        'episodes':len(episode_rows),'parameter_change_l2':total_delta,
        'initial_vector_sha256':tensor_digest(initial),'initial_actor_vector_sha256':tensor_digest(initial_actor),
        'checkpoint_sha256':sha256(checkpoint),'portable_export_sha256':export_hash,
        'closed_manifest_sha256':sha256(out/'CLOSED_FILES.json'),
        'elapsed_s':elapsed_before+time.perf_counter()-start,'counters':counters,
        'phase_counts':phase_counts.tolist(),'action_digest':action_digest,'final_dual':dual,
        'new_proof_or_cloud_calls_inside_runtime':0}
    atomic_json(done_path,done)
    (out/'PAUSED.json').unlink(missing_ok=True)
    return done


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--stop-after-updates',type=int)
    a=p.parse_args();train(json.loads(a.config.read_text()),a.out,stop_after_updates=a.stop_after_updates)
