"""Mandatory genuine proof self-test. Failure never becomes a skipped/passed test."""
from __future__ import annotations
import argparse,copy,json,re
from dataclasses import asdict
from pathlib import Path
from .environment import EnvConfig,Warehouse,monitor_record
from .crypto import NodeWorker,build_witness,artifact_paths,public_vector,CryptoError,AdmissionVerifier,FIELD
from .io import atomic_json


def samples():
    e=Warehouse(EnvConfig(n_agents=2),19)
    e.pos[:]=[[2,2],[10,3]]
    good=e.record_for_actions([1,4])
    cases={}
    for name,pos,actions in [('collision',[[2,2],[4,2]],[1,2]),('swap',[[2,2],[3,2]],[1,2]),
                             ('wall',[[5,2],[10,3]],[1,0]),('capacity',[[4,3],[8,3]],[1,2]),
                             ('quota',[[4,3],[10,3]],[1,0])]:
        x=copy.deepcopy(good);x['before']['positions']=pos;x['actions']=actions
        if name=='capacity':x['capacities']=[1,2]
        if name=='quota':x['before']['counts'][0][0]=3
        if monitor_record(x)['compliant']:raise AssertionError(f'Invalid negative fixture: {name}')
        cases[name]=x
    return good,cases


def check(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);passed=[]
    good,cases=samples()
    with NodeWorker() as worker:
        golden=worker.hash([1,2])
        if golden!=7853200120776062878684798364095072458815029376092732009249414926327459813530:
            raise AssertionError(f'Poseidon golden vector mismatch: {golden}')
        passed.append('circomlibjs_poseidon_golden_1_2')
        built=build_witness(good,worker,run_tag=101,nonce=201,salt=301)
        for protocol in ('groth16','plonk'):
            paths=artifact_paths('twodoors',2,protocol)
            def prove(inp):return worker.request('prove',protocol=protocol,input=inp,wasm=paths['wasm'],zkey=paths['zkey'],vkey=paths['vkey'])
            result=prove(built['input'])
            if not result['valid'] or result['publicSignals']!=public_vector(built['expected'],paths):
                raise AssertionError('Honest proof / public ABI / host-Circom commitment mismatch')
            passed.append(protocol+'_honest_proof_and_poseidon_parity')
            admission=AdmissionVerifier(worker,paths,protocol)
            if admission.accept(result['proof'],result['publicSignals'],built['expected']) is not True:raise AssertionError('Honest admission failed')
            if admission.accept(result['proof'],result['publicSignals'],built['expected']) is not False:raise AssertionError('Replay accepted')
            passed.append(protocol+'_replay_rejected')
            tamper=result['publicSignals'].copy();tamper[-1]=str((int(tamper[-1])+1)%FIELD)
            tested=worker.request('verify',protocol=protocol,vkey=paths['vkey'],proof=result['proof'],publicSignals=tamper)
            if tested['valid']:raise AssertionError('Tampered public statement verified')
            passed.append(protocol+'_public_tamper_rejected')
            for name,record in cases.items():
                negative=build_witness(record,worker,run_tag=101,nonce=202,salt=301,strict=False)
                try:prove(negative['input'])
                except CryptoError as ex:
                    if not re.search(r'assert|constraint|template',str(ex),re.I):raise
                else:raise AssertionError(f'{protocol}: unsafe {name} witness accepted')
                passed.append(protocol+'_'+name+'_circuit_rejected')
            # Malicious witness modification cannot retain the trusted state root.
            bad=copy.deepcopy(built['input']);bad['used'][0][0]='1'
            try:prove(bad)
            except CryptoError as ex:
                if not re.search(r'assert|constraint|template',str(ex),re.I):raise
            else:raise AssertionError('Unbound private state accepted')
            passed.append(protocol+'_private_state_substitution_rejected')
            # Explicit in-circuit range rejection, not merely a Python precheck.
            bad=copy.deepcopy(built['input']);bad['used'][0][0]=str(FIELD-1)
            vals=[101,0,0]
            for i in range(2):vals += [int(bad['x'][i]),int(bad['y'][i]),int(bad['used'][i][0]),int(bad['used'][i][1])]
            bad['state_root']=str(worker.commit(1001,vals,301))
            try:prove(bad)
            except CryptoError as ex:
                if not re.search(r'assert|constraint|template',str(ex),re.I):raise
            else:raise AssertionError('Unsigned field alias accepted')
            passed.append(protocol+'_unsigned_range_rejected')
            atomic_json(out/f'{protocol}_honest.json',result)
    report={'status':'PASS','checks':passed,'count':len(passed),'scope':'actual Node Poseidon + Circom/snarkjs positive and negative tests'}
    atomic_json(out/'CRYPTO_TESTS_PASSED.json',report)
    print(json.dumps(report,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/preflight/crypto');a=p.parse_args();check(a.out)
