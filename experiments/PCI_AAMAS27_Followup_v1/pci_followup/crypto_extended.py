"""Full frozen-policy episodes; no learning, no modified circuit, no outcome selection.

Every executed transition has a fresh Groth16 certificate. At fixed within-episode
indices, both SNARKs and disclosed-record signatures precede execution, in balanced
six-way order. Upstream circuit and checkpoint hashes must match the frozen run.
"""
import argparse,copy,gzip,itertools,json,re,time,os
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from . import BASE
from .io import atomic_json,atomic_csv,sha256,immutable_json
from pci_bench.environment import EnvConfig,Warehouse,monitor_record
from pci_bench.model import ContextPolicy
from pci_bench.evaluate import one_episode
from pci_bench.crypto import NodeWorker,build_witness,artifact_paths,public_vector,AdmissionVerifier,CryptoError,FIELD
from pci_bench.crypto_check import samples
from pci_bench.crypto_benchmark import signed_disclosure
SAMPLES=(0,32,63,64,96,127,128,160,191,192,224,255)
ORDERS=list(itertools.permutations(('groth16','plonk','ed25519_disclosure')))

def negative_checks(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);checks=[];good,neg=samples()
    pads=[[1,11],[3,11],[5,11],[7,11],[9,11],[11,11]]
    with NodeWorker() as worker:
        for n in (2,4,8):
            def padded(record):
                r=copy.deepcopy(record);r['config']['n_agents']=n
                r['before']['positions']+=pads[:n-2];r['before']['counts']+=[[0,0] for _ in range(n-2)]
                r['actions']+=[0]*(n-2);r['quotas']+=[[3,3] for _ in range(n-2)]
                return r
            built=build_witness(padded(good),worker,run_tag=801+n,nonce=201,salt=301)
            for protocol in ('groth16','plonk'):
                paths=artifact_paths('twodoors',n,protocol)
                def prove(inp):return worker.request('prove',protocol=protocol,input=inp,wasm=paths['wasm'],zkey=paths['zkey'],vkey=paths['vkey'])
                result=prove(built['input'])
                if result['valid'] is not True or result['publicSignals']!=public_vector(built['expected'],paths):raise AssertionError('Honest proof rejected')
                verifier=AdmissionVerifier(worker,paths,protocol)
                assert verifier.accept(result['proof'],result['publicSignals'],built['expected'])
                assert not verifier.accept(result['proof'],result['publicSignals'],built['expected'])
                checks.extend([f'n{n}_{protocol}_honest',f'n{n}_{protocol}_replay'])
                tampered=result['publicSignals'].copy();tampered[-1]=str((int(tampered[-1])+1)%FIELD)
                tested=worker.request('verify',protocol=protocol,vkey=paths['vkey'],proof=result['proof'],publicSignals=tampered)
                if tested['valid']:raise AssertionError('Tampered public signals accepted by cryptographic verifier')
                checks.append(f'n{n}_{protocol}_public_tamper')
                for field in ('step','episode','policy_version','rules_root','action_root','state_root'):
                    expected=copy.deepcopy(built['expected']);expected[field]=str(int(expected[field])+1)
                    assert not AdmissionVerifier(worker,paths,protocol).accept(result['proof'],result['publicSignals'],expected)
                    checks.append(f'n{n}_{protocol}_wrong_{field}')
                for name,record in neg.items():
                    record=padded(record)
                    if monitor_record(record)['compliant']:raise AssertionError('Invalid adversarial fixture')
                    bad=build_witness(record,worker,run_tag=801+n,nonce=202,salt=301,strict=False)
                    try:prove(bad['input'])
                    except CryptoError as ex:
                        if not re.search(r'assert|constraint|template',str(ex),re.I):raise
                    else:raise AssertionError(f'Unsafe witness accepted: {name}')
                    checks.append(f'n{n}_{protocol}_{name}_circuit_rejected')
                bad=copy.deepcopy(built['input']);bad['used'][0][0]='1'
                try:prove(bad)
                except CryptoError as ex:
                    if not re.search(r'assert|constraint|template',str(ex),re.I):raise
                else:raise AssertionError('Substituted state accepted')
                checks.append(f'n{n}_{protocol}_state_substitution')
                bad=copy.deepcopy(built['input']);bad['used'][0][0]=str(FIELD-1)
                vals=[801+n,0,0]
                for i in range(n):vals += [int(bad['x'][i]),int(bad['y'][i]),int(bad['used'][i][0]),int(bad['used'][i][1])]
                bad['state_root']=str(worker.commit(1001,vals,301))
                try:prove(bad)
                except CryptoError as ex:
                    if not re.search(r'assert|constraint|template',str(ex),re.I):raise
                else:raise AssertionError('Unsigned field alias accepted')
                checks.append(f'n{n}_{protocol}_unsigned_range')
                atomic_json(out/f'n{n}_{protocol}_honest.json',result)
                atomic_json(out/'progress.json',{'checks':checks,'count':len(checks)})
    atomic_json(out/'CRYPTO_PREFLIGHT_COMPLETE.json',{'status':'COMPLETE','checks':checks,'count':len(checks)})

def run_case(out,n,seed,regime,small=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    run=BASE/'results/deadline/runs'/f'pci_twodoors_l0.2_s{seed}'
    cfg=json.loads((run/'config.json').read_text());cp=run/'checkpoint.pt'
    expected=json.loads((run/'TRAIN_COMPLETE.json').read_text())['checkpoint_sha256']
    if sha256(cp)!=expected:raise RuntimeError('Frozen checkpoint changed')
    horizon=4 if small else 256
    spec={'checkpoint_sha256':expected,'n_agents':n,'train_seed':seed,'regime':regime,'horizon':horizon,'samples':list(SAMPLES),'version':1}
    immutable_json(out/'spec.json',spec)
    if (out/'COMPLETE.json').exists():
        d=json.loads((out/'COMPLETE.json').read_text())
        if sha256(out/'measurements.csv')!=d['csv_sha256']:raise RuntimeError('Completed measurements changed')
        return d
    # Interrupted cases are preserved; a fresh attempt is labelled, never mixed.
    attempt=out/'attempt_0';idx=0
    while attempt.exists():idx+=1;attempt=out/f'attempt_{idx}'
    attempt.mkdir();torch.set_num_threads(1)
    model=ContextPolicy(cfg['method'],cfg['hidden']);model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model']);model.eval()
    ecfg=replace(EnvConfig(**cfg['env']),n_agents=n,horizon=horizon,rule_change=regime=='rule_shift')
    env_seed=610000+n*100+seed
    env=Warehouse(ecfg,env_seed);rows=[];reset_rows=[];previous=None
    private=Ed25519PrivateKey.generate();case_id=f'n{n}_s{seed}_{regime}'
    with NodeWorker(timeout=600) as worker,gzip.open(attempt/'trace.jsonl.gz','wt') as trace:
        paths={p:artifact_paths('twodoors',n,p) for p in ('groth16','plonk')}
        verifier=AdmissionVerifier(worker,paths['groth16'],'groth16')
        def certify(record):
            nonlocal previous
            t=record['step'];resets=t>0 and t%64==0
            if previous is not None:
                if record['before']['positions']!=previous['positions']:raise AssertionError('State continuity broken')
                expected_counts=np.zeros((n,2),dtype=int) if resets else np.array(previous['counts'])
                if record['before']['counts']!=expected_counts.tolist():raise AssertionError('Counter reset/history mismatch')
            binding_start=time.perf_counter_ns()
            built=build_witness(record,worker,run_tag=700000+n*100+seed*10+int(regime=='rule_shift'))
            binding_ms=(time.perf_counter_ns()-binding_start)/1e6
            if resets:reset_rows.append({'step':t,'policy_version':built['expected']['policy_version'],'counts_zero':not np.any(record['before']['counts'])})
            paired=t in SAMPLES
            order=ORDERS[(SAMPLES.index(t)+seed)%6] if paired else ('groth16',)
            for order_index,protocol in enumerate(order):
                started=time.perf_counter()
                if protocol=='ed25519_disclosure':row=signed_disclosure(record,private)
                else:
                    p=paths[protocol];result=worker.request('prove',protocol=protocol,input=built['input'],wasm=p['wasm'],zkey=p['zkey'],vkey=p['vkey'])
                    if result['valid'] is not True or result['publicSignals']!=public_vector(built['expected'],p):raise AssertionError('Proof or binding mismatch')
                    if protocol=='groth16':
                        if not verifier.accept(result['proof'],result['publicSignals'],built['expected']):raise AssertionError('Admission rejected')
                        if paired and verifier.accept(result['proof'],result['publicSignals'],built['expected']):raise AssertionError('Replay accepted')
                    folder=attempt/'proofs'/f't{t:04d}'/protocol;folder.mkdir(parents=True)
                    for name,obj in [('input',built['input']),('expected',built['expected']),('proof',result['proof']),('public',result['publicSignals']),('record',record)]:atomic_json(folder/f'{name}.json',obj)
                    row={k:v for k,v in result.items() if k not in ('proof','publicSignals')}
                    row.update(protocol=protocol,proof_encoding='snarkjs_json',host_binding_ms=None,wall_total_ms=(time.perf_counter()-started)*1000,n_agents=n,layout='twodoors')
                row.update(case_id=case_id,step=t,paired=int(paired),order_index=order_index,shared_record_binding_ms=binding_ms,
                    protocol_order='>'.join(order),train_seed=seed,regime=regime,policy_version=int(built['expected']['policy_version']))
                rows.append(row)
            checked=monitor_record(record)
            if not checked['compliant']:raise AssertionError('Noncompliant executed proof')
            previous={'positions':checked['next_positions'],'counts':(np.array(record['before']['counts'])+np.array(checked['entries'])).tolist()}
            if t%16==0 or t+1==horizon:
                atomic_csv(attempt/'measurements.csv',rows);print(case_id,'certified',t+1,'/',horizon,flush=True)
            return True
        result=one_episode(env,model=model,seed=env_seed,enforce=True,trace_file=trace,certifier=certify)
    if horizon==256 and [r['step'] for r in reset_rows]!=[64,128,192]:raise AssertionError('Missing reset coverage')
    atomic_csv(out/'measurements.csv',rows);atomic_json(out/'episode.json',result);atomic_json(out/'resets.json',reset_rows)
    d={'status':'COMPLETE','spec':spec,'certified_transitions':horizon,'measurements':len(rows),'reset_count':len(reset_rows),
        'deliveries':result['deliveries'],'csv_sha256':sha256(out/'measurements.csv'),'attempt':str(attempt.name)}
    atomic_json(out/'COMPLETE.json',d);return d

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--preflight',action='store_true');p.add_argument('--small',action='store_true')
    a=p.parse_args()
    if a.preflight:negative_checks(a.out)
    else:
        cases=[]
        for n in ((2,) if a.small else (2,4,8)):
            for seed in ((0,) if a.small else (0,1)):
                for regime in (('nominal',) if a.small else ('nominal','rule_shift')):
                    cases.append(run_case(Path(a.out)/f'n{n}_s{seed}_{regime}',n,seed,regime,a.small))
        atomic_json(Path(a.out)/'CERTIFIED_EPISODES_COMPLETE.json',{'status':'COMPLETE','cases':cases,
            'certified_transitions':sum(c['certified_transitions'] for c in cases),'measurements':sum(c['measurements'] for c in cases)})
