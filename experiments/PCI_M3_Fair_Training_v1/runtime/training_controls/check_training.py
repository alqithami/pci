#!/usr/bin/env python3
"""Bounded CPU implementation fixtures, never a scientific learning cohort.

Run from the project root. --output is a NEW JSON receipt filename; the sibling
<receipt-stem>_files directory retains the small closed fixture artifacts.
No package installation, GPU, cloud, protected seed or heldout call occurs.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, copy, hashlib, importlib.util, json, shutil, signal, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from training.pci_risk_train import trainer
from training.pci_risk_train.model import ContextPolicy, gae
from training.pci_risk_train.objectives import (METHODS,MODEL_METHOD,auxiliary,
    institutional_risk,SEMANTIC_BOUND_SCALE,CONSENSUS_BOUND_SCALE)
from training.pci_risk_train.io import atomic_json, source_hashes, sha256, system_meta


def config(method,steps=1024,run_id=None):
    return dict(run_role='readiness_fixture',method=method,seed=811001,init_seed=811001,
        policy_seed=842100001,minibatch_seed=842200001,
        env_seeds=[842000000+i for i in range(8)],env_profile_ids=list(trainer.ENV_PROFILE_IDS),
        profiles=copy.deepcopy(trainer.PROFILES),n_envs=8,rollout_steps=128,total_steps=steps,
        hidden=64,learning_rate=0.0003,clip_coef=0.2,entropy_coef=0.02,gamma=0.99,
        gae_lambda=0.95,max_grad_norm=0.5,target_kl=0.03,vf_coef=0.5,
        update_epochs=4,minibatch_groups=128,torch_threads=1,device='cpu',
        penalty_coef=0.25,consistency_coef=0.2,dual_initial=0.25,dual_lr=0.05,
        dual_target=0.01,dual_max=5.0,checkpoint_every_updates=8,
        run_id=run_id or 'fixture_'+method)


def recursive_equal(a,b):
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray): return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict): return isinstance(b,dict) and set(a)==set(b) and all(recursive_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)): return type(a)==type(b) and len(a)==len(b) and all(recursive_equal(x,y) for x,y in zip(a,b))
    return a==b


def expect_failure(call,label):
    try: call()
    except (ValueError,RuntimeError) as exc: return {'status':'PASS','rejection':str(exc),'label':label}
    raise AssertionError('Expected fail-closed rejection: '+label)


def run(receipt,deadline):
    start=time.perf_counter();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    receipt=receipt.resolve()
    if receipt.exists(): raise RuntimeError('Refusing to replace an existing fixture receipt')
    work=receipt.with_name(receipt.stem+'_files')
    if work.exists(): raise RuntimeError('Fresh fixture work directory required')
    work.mkdir(parents=True)
    checks={};source_before=source_hashes(ROOT/'training')
    anchors=json.loads(Path(__file__).with_name('FIXTURE_INPUT_ANCHORS.json').read_text())
    assert sha256(Path(__file__).with_name('fixture_inputs.npz'))==anchors['fixture_input_sha256']
    assert sha256(Path(__file__).with_name('fixture_actor.npz'))==anchors['source_members']['policy_exports_v1/exports/pci_s0/policy.npz']
    assert sha256(Path(__file__).with_name('reference_network_gradients.py'))==anchors['reference_gradient_module_sha256']
    report={'schema':'PCI-M2-CPU-TRAINING-READINESS-FIXTURES-v1','status':'RUNNING',
        'role':'Bounded implementation validation only, not scientific training/evaluation',
        'sources':source_before,'fixture_source_sha256':sha256(Path(__file__)),
        'system':system_meta(),'work_directory':str(work),'checks':checks,
        'fixture_input_anchor_sha256':sha256(Path(__file__).with_name('FIXTURE_INPUT_ANCHORS.json'))}
    def timeout(signum,frame): raise TimeoutError('Bounded CPU fixture deadline exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,deadline)
    counts={'constructors':0,'resets':0,'transitions':0}
    originals={n:getattr(trainer.Warehouse,n) for n in ('__init__','reset','transition')}
    for name,key in [('__init__','constructors'),('reset','resets'),('transition','transitions')]:
        def wrapped(self,*args,_name=name,_key=key,**kwargs):
            counts[_key]+=1;return originals[_name](self,*args,**kwargs)
        setattr(trainer.Warehouse,name,wrapped)
    try:
        # Independent retained M1 analytic gradients versus actual Torch autograd.
        spec=importlib.util.spec_from_file_location('m1_reference_gradients',Path(__file__).with_name('reference_network_gradients.py'))
        ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
        with np.load(Path(__file__).with_name('fixture_actor.npz'),allow_pickle=False) as a:
            weights={k:a[k].copy() for k in a.files}
        with np.load(Path(__file__).with_name('fixture_inputs.npz'),allow_pickle=False) as a:
            features=a['features'].copy();raw=a['institutional_replacements'].copy()
        torch.manual_seed(811000);model=ContextPolicy('pci',64).double()
        model.load_state_dict({k:torch.from_numpy(v) for k,v in weights.items()})
        x=torch.tensor(features,dtype=torch.float64);cost=torch.tensor(raw,dtype=torch.int64)
        gradient_rows=[]
        for method,mode,scale in [('directrisk',None,1.),('semantic','pci',SEMANTIC_BOUND_SCALE),('consensus','consensus',CONSENSUS_BOUND_SCALE)]:
            dist,_,heads=model(x);model.zero_grad(set_to_none=True)
            value=auxiliary(method,dist,heads,x,cost);value.backward()
            expected=(ref.risk_grad(weights,features,raw) if mode is None else ref.energy_grad(weights,features,mode))
            errors=[]
            for n,p in model.named_parameters():
                if n.startswith('heads.'):
                    actual=p.grad.detach().numpy();wanted=scale*expected['gradient'][n]
                    assert np.allclose(actual,wanted,rtol=1e-8,atol=1e-10),n
                    errors.append(float(np.max(np.abs(actual-wanted))))
                else: assert p.grad is None
            assert abs(float(value.detach())-scale*expected['value'])<=1e-10
            gradient_rows.append({'method':method,'value_error':abs(float(value.detach())-scale*expected['value']),
                                  'max_actor_gradient_abs_error':max(errors)})
        checks['analytic_vs_torch_actual_actor_gradients']={'status':'PASS','literal_states':4,'comparisons':gradient_rows}
        dist,_,heads=model(x);zero=torch.zeros_like(cost)
        model.zero_grad(set_to_none=True);z=institutional_risk(dist.probs,zero).mean();z.backward()
        assert float(z.detach())==0. and all(p.grad is not None and not torch.count_nonzero(p.grad) for h in model.heads for p in h.parameters())
        ones=torch.ones((1,625),dtype=torch.uint8)
        assert abs(float(institutional_risk(torch.full((1,4,5),0.2,dtype=torch.float64),ones)[0])-0.25)<1e-12
        checks['raw_divided_by_four_once_and_zero_rows']={'status':'PASS','constant_raw_one_value':0.25,
            'literal_zero_cost_rows':int(np.count_nonzero(~raw.any(axis=1))),'all_zero_gradient_exact':True}
        # The mean includes inactive states; no conditioning on active support.
        dist,_,_=model(x)
        full=institutional_risk(dist.probs,cost)
        assert len(full)==4 and float(full[0].detach())==0.
        assert abs(float(full.mean().detach())-ref.risk_value(weights,features,raw))<1e-10
        checks['complete_state_mean_includes_inactive']={'status':'PASS','states':4,'inactive_rows_retained':1}
        # Genuine terminal cut prevents value/advantage leakage across episodes.
        rewards=torch.tensor([[[1.]*4],[[2.]*4]])
        values=torch.tensor([[[3.]*4],[[4.]*4]])
        dones=torch.tensor([[1.],[0.]])
        adv,ret=gae(rewards,values,dones,torch.full((1,4),100.),0.9,0.8)
        assert torch.equal(adv[0],torch.full((1,4),-2.)) and torch.equal(ret[0],torch.ones((1,4)))
        other=gae(rewards,values,dones,torch.full((1,4),999.),0.9,0.8)[0]
        assert torch.equal(adv[0],other[0])
        checks['gae_terminal_boundary']={'status':'PASS'}
        # Same weights/masks/critic before method-specific objectives take effect.
        init=[]
        for m in METHODS:
            torch.manual_seed(811001);m0=ContextPolicy(MODEL_METHOD[m],64)
            init.append(trainer.tensor_digest(trainer.model_vector(m0)))
        assert len(set(init))==1
        checks['all_arm_identical_cpu_initialization']={'status':'PASS','methods':list(METHODS),'model_vector_sha256':init[0]}
        arm_runs={};configs={};native_checks=0;corrected=0
        for method in METHODS:
            cfg=config(method);configs[method]=cfg;dest=work/method
            result=trainer.train(cfg,dest);assert result['status']=='COMPLETE' and result['env_steps']==1024
            assert result['initial_vector_sha256']==init[0]
            assert result['counters']['sampled_native_oracle_equal']==1024
            assert np.array_equal(np.asarray(result['phase_counts']),np.full((4,64),4))
            ck=torch.load(dest/'checkpoint.pt',map_location='cpu',weights_only=False)
            assert len(ck['updates'])==1 and ck['updates'][0]['optimizer_steps']>0
            assert np.isfinite(ck['updates'][0]['mean_grad_norm']) and ck['updates'][0]['actor_change_l2']>0
            assert result['counters']['oracle_candidate_rows']==640000
            # Reconstruct old-policy likelihoods from the INITIAL policy at every
            # retained proposal, never from the admitted substitute.
            policy=ContextPolicy(MODEL_METHOD[method],64)
            policy.load_state_dict(torch.load(dest/'policy_0.pt',map_location='cpu',weights_only=True))
            with np.load(dest/'rollouts/update_000001.npz',allow_pickle=False) as a:
                obs=torch.tensor(a['obs'].reshape(1024,4,5,23))
                proposal=torch.tensor(a['actions'].reshape(1024,4))
                with torch.no_grad(): d,_,_=policy(obs); lp=d.log_prob(proposal)
                # Different GEMM group sizes may change float32 rounding slightly.
                assert np.allclose(lp.numpy(),a['logp'].reshape(1024,4),atol=2e-6,rtol=0.)
                assert a['raw_costs'].shape==(128,8,625)
                differences=int(np.count_nonzero(a['actions']!=a['admitted']))
                corrected+=differences;native_checks+=1024
                assert a['actions'].shape==a['logp'].shape==(128,8,4)
            arm_runs[method]={'status':'PASS','optimizer_steps':ck['updates'][0]['optimizer_steps'],
                'actor_change_l2':ck['updates'][0]['actor_change_l2'],
                'sampled_native_oracle_equal':1024,'corrected_proposal_components':differences,
                'complete_receipt_sha256':sha256(dest/'TRAIN_COMPLETE.json')}
        assert corrected>0
        checks['six_arm_one_update_and_proposal_likelihood']={'status':'PASS','arms':arm_runs,
            'checked_proposal_decisions':6*1024*4,'native_sampled_oracle_checks':native_checks,
            'corrected_proposal_components':corrected,'statistics_are_implementation_fixtures_only':True}
        # Exact adaptive resume exercises optimizer/RNG/world/dual restoration.
        cfg=config('adaptive',2048,'fixture_resume_common')
        continuous=work/'resume_continuous';resumed=work/'resume_split'
        one=trainer.train(cfg,continuous)
        paused=trainer.train(cfg,resumed,stop_after_updates=1)
        assert paused['status']=='PAUSED_READINESS_FIXTURE'
        corrupt_resume=work/'corrupt_checkpoint';shutil.copytree(resumed,corrupt_resume)
        with (corrupt_resume/'checkpoint.pt').open('ab') as f: f.write(b'CORRUPT_CLOSED_CHECKPOINT')
        before=counts.copy()
        checks['corrupt_checkpoint_commit_guard']=expect_failure(lambda:trainer.train(cfg,corrupt_resume),'checkpoint body mismatch')
        assert counts==before
        two=trainer.train(cfg,resumed)
        a=torch.load(continuous/'checkpoint.pt',map_location='cpu',weights_only=False)
        b=torch.load(resumed/'checkpoint.pt',map_location='cpu',weights_only=False)
        fields=['model','optimizer','envs','obs','step','update','episodes','env_sums',
            'initial_vector','initial_actor_vector','torch_rng','numpy_rng','python_rng','cuda_rng',
            'policy_generator_state','minibatch_generator_state','dual','counters','phase_counts','action_digest']
        for field in fields: assert recursive_equal(a[field],b[field]),field
        clean=lambda rows:[{k:v for k,v in row.items() if k not in ('elapsed_s','env_steps_per_s')} for row in rows]
        assert recursive_equal(clean(a['updates']),clean(b['updates']))
        for n in a['closed_rollouts']:
            with np.load(continuous/n,allow_pickle=False) as x,np.load(resumed/n,allow_pickle=False) as y:
                assert set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files)
        assert one['action_digest']==two['action_digest'] and one['final_dual']==two['final_dual']
        checks['exact_adaptive_checkpoint_resume']={'status':'PASS','uninterrupted_updates':2,
            'split_updates':[1,1],'fields_compared':fields,'all_rollout_arrays_identical':True,
            'all_update_statistics_except_time_identical':True,'action_digest':one['action_digest']}
        # Reducing the manifest cannot hide missing retained rollout bodies.
        tampered=work/'tampered_manifest';shutil.copytree(work/'task',tampered)
        f=tampered/'CLOSED_FILES.json';obj=json.loads(f.read_text());obj['files']={};f.write_text(json.dumps(obj))
        before=counts.copy()
        checks['tampered_closed_manifest_guard']=expect_failure(lambda:trainer.train(configs['task'],tampered),'manifest body mismatch')
        assert counts==before
        assert counts['transitions']==10240
        assert source_hashes(ROOT/'training')==source_before
        report.update(status='PASS',elapsed_seconds=time.perf_counter()-start,
            fixture_world_calls=counts,scientific_training_or_evaluation_episodes=0,
            GPU_or_cloud_or_proof_calls=0,sources_unchanged=True,
            retained_fixture_bytes=sum(p.stat().st_size for p in work.rglob('*') if p.is_file()))
    except BaseException as exc:
        report.update(status='FAIL',error=repr(exc),elapsed_seconds=time.perf_counter()-start,
                      fixture_world_calls=counts,scientific_training_or_evaluation_episodes=0)
        atomic_json(receipt,report);raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0.)
        for name,value in originals.items(): setattr(trainer.Warehouse,name,value)
    atomic_json(receipt,report)
    print(json.dumps({'status':report['status'],'receipt':str(receipt),'elapsed_seconds':report['elapsed_seconds'],
        'fixture_world_calls':counts,'retained_fixture_bytes':report['retained_fixture_bytes']},sort_keys=True))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    p.add_argument('--deadline-seconds',type=float,default=85.)
    a=p.parse_args();run(a.output,a.deadline_seconds)
