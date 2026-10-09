#!/usr/bin/env python3
"""Real new-circuit binding tests; no fallback to an encoding-only hasher."""
import argparse,copy,json,re,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--variant',type=Path,required=True);a=p.parse_args();root=a.variant.resolve();sys.path.insert(0,str(root))
from pci_bench.crypto import NodeWorker,build_witness,artifact_paths,public_vector,AdmissionVerifier,CryptoError
from pci_bench.crypto_check import samples
from pci_bench.io import atomic_json
record,_=samples();checks=[];out=root/'validation/blinded_binding';out.mkdir(parents=True,exist_ok=True)
with NodeWorker() as worker:
    for protocol in ('groth16','plonk'):
        paths=artifact_paths('twodoors',2,protocol)
        inputs=[]
        for opening in (4001,4002):
            b=build_witness(record,worker,run_tag=8001,nonce=9001,salt=3001,action_salt=opening)
            result=worker.request('prove',protocol=protocol,input=b['input'],wasm=paths['wasm'],zkey=paths['zkey'],vkey=paths['vkey'])
            if result['valid'] is not True or result['publicSignals']!=public_vector(b['expected'],paths):raise AssertionError('Binding or verification failed')
            inputs.append(b)
            atomic_json(out/f'{protocol}_{opening}.json',{'built':b,'proof_result':result})
            checks.append(f'{protocol}_honest_opening_{opening}')
        if inputs[0]['expected']['action_root']==inputs[1]['expected']['action_root']:raise AssertionError('Opening not bound')
        checks.append(f'{protocol}_different_action_roots')
        bad=copy.deepcopy(inputs[0]['input']);bad['action_salt']='4002'
        try:worker.request('prove',protocol=protocol,input=bad,wasm=paths['wasm'],zkey=paths['zkey'],vkey=paths['vkey'])
        except CryptoError as exc:
            if not re.search(r'assert|constraint|template',str(exc),re.I):raise
        else:raise AssertionError('Changed action opening accepted for unchanged root')
        checks.append(f'{protocol}_salt_substitution_rejected')
        verifier=AdmissionVerifier(worker,paths,protocol)
        if not verifier.accept(result['proof'],result['publicSignals'],inputs[1]['expected']):raise AssertionError('Fresh admission failed')
        if verifier.accept(result['proof'],result['publicSignals'],inputs[1]['expected']):raise AssertionError('Replay accepted')
        checks.append(f'{protocol}_replay_rejected')
atomic_json(out/'REAL_BINDING_CHECKS.json',{'status':'PASS','checks':checks,'count':len(checks),'scope':'Two-agent genuine circuit smoke only; no privacy proof, end-to-end benchmark, or historical-results replacement.'})
print('REAL_BLINDED_BINDING_TESTS_PASS',len(checks))
