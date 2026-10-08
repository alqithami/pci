from __future__ import annotations
import argparse,copy,json,os,time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.exceptions import InvalidSignature
from .environment import EnvConfig,Warehouse,monitor_record
from .model import ContextPolicy
from .crypto import NodeWorker,artifact_paths,prove_record,AdmissionVerifier
from .crypto_build import build
from .evaluate import one_episode
from .io import atomic_json,atomic_csv,sha256


def signed_disclosure(record,private):
    payload=json.dumps(record,sort_keys=True,separators=(',',':')).encode()
    t=time.perf_counter_ns();sig=private.sign(payload);gen=(time.perf_counter_ns()-t)/1e6
    t=time.perf_counter_ns();private.public_key().verify(sig,payload);valid=monitor_record(record)['compliant'];ver=(time.perf_counter_ns()-t)/1e6
    if not valid:raise AssertionError('Disclosure predicate failed')
    return {'protocol':'ed25519_disclosure','proof_encoding':'raw_signature','valid':True,'witness_and_prove_ms':gen,'verify_ms':ver,
            'proof_json_bytes':len(sig),'public_json_bytes':len(payload),'host_binding_ms':0.,
            'wall_total_ms':gen+ver,'n_agents':len(record['actions']),'layout':record['config']['layout']}


def benchmark(run_dir,out,steps=32,agent_counts=(2,4,8)):
    out=Path(out)
    if (out/'CRYPTO_BENCH_COMPLETE.json').exists():
        saved=json.loads((out/'CRYPTO_BENCH_COMPLETE.json').read_text())
        if saved['model_sha256']!=sha256(Path(run_dir)/'checkpoint.pt') or saved['measurement_csv_sha256']!=sha256(out/'proof_measurements.csv'):
            raise RuntimeError('Completed proof benchmark changed')
        if saved['certified_transitions']!=steps*len(agent_counts):raise RuntimeError('Proof benchmark configuration changed')
        return
    if out.exists() and any(out.iterdir()):
        moved=out.with_name(out.name+'_interrupted_'+str(time.time_ns()));os.replace(out,moved)
    out.mkdir(parents=True,exist_ok=True)
    config=json.loads((Path(run_dir)/'config.json').read_text())
    if config['env']['layout']!='twodoors':raise ValueError('The preregistered crypto anchor is twodoors')
    torch.set_num_threads(1);model=ContextPolicy(config['method'],config['hidden'])
    ck=torch.load(Path(run_dir)/'checkpoint.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(ck['model']);model.eval()
    private=Ed25519PrivateKey.generate();measurements=[];episodes=[]
    # Test actual signature rejection too.
    sig=private.sign(b'original')
    try:private.public_key().verify(sig,b'changed')
    except InvalidSignature:pass
    else:raise AssertionError('Ed25519 tamper accepted')
    for n in agent_counts:build('twodoors',n)
    with NodeWorker() as worker:
        for n in agent_counts:
            pg=artifact_paths('twodoors',n,'groth16');pp=artifact_paths('twodoors',n,'plonk')
            admission=AdmissionVerifier(worker,pg,'groth16')
            ecfg=replace(EnvConfig(**config['env']),n_agents=n,horizon=steps)
            env=Warehouse(ecfg,300000+n)
            run_tag=90000+n
            def certify(record):
                index=record['step']
                # This call occurs BEFORE the environment mutates its positions.
                row,built,result=prove_record(record,worker,pg,'groth16',out/f'n{n}'/f't{index:04d}'/'groth16',run_tag,admission)
                row.update({'case_id':f'n{n}_t{index}','admission_mode':'synchronous_before_execution'});measurements.append(row)
                # Same recorded transition, paired protocol comparison, not a second action.
                row,_,_=prove_record(record,worker,pp,'plonk',out/f'n{n}'/f't{index:04d}'/'plonk',run_tag)
                row.update({'case_id':f'n{n}_t{index}','admission_mode':'paired_same_transition'});measurements.append(row)
                row=signed_disclosure(record,private)
                row.update({'case_id':f'n{n}_t{index}','admission_mode':'paired_disclosed_record'});measurements.append(row)
                atomic_csv(out/'proof_measurements.csv',measurements)
                return True
            result=one_episode(env,model=model,seed=300000+n,enforce=True,certifier=certify)
            episodes.append({'n_agents':n,'certified_steps':steps,**result})
    atomic_csv(out/'certified_episodes.csv',episodes)
    atomic_json(out/'CRYPTO_BENCH_COMPLETE.json',{'model_sha256':sha256(Path(run_dir)/'checkpoint.pt'),
        'measurements':len(measurements),'certified_transitions':steps*len(agent_counts),
        'scoped_claim':'joint all-agent operational transition proofs; trusted simulator state commitments; no inference proof',
        'measurement_csv_sha256':sha256(out/'proof_measurements.csv')})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=32)
    p.add_argument('--agents',nargs='+',type=int,default=[2,4,8]);a=p.parse_args();benchmark(a.run_dir,a.out,a.steps,tuple(a.agents))
