#!/usr/bin/env python3
"""Host encoding/generator tests only. The hash double is NOT Poseidon."""
import argparse,hashlib,json,sys,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--variant',type=Path,required=True);a=p.parse_args();root=a.variant.resolve();sys.path.insert(0,str(root))
from pci_bench.crypto import build_witness,FIELD
from pci_bench.crypto_check import samples
from pci_bench.crypto_generate import generate
class EncodingTestHash:
    """Deterministic serialization test double; no cryptographic parity claim."""
    def hash(self,v):return int(hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest(),16)%FIELD
    def commit(self,domain,v,salt=0):return self.hash([domain,v,salt])
w=EncodingTestHash();record,_=samples();checks=[]
def ok(name,b):
    if not b:raise AssertionError(name)
    checks.append(name)
a1=build_witness(record,w,run_tag=1,nonce=2,salt=3,action_salt=4)
a2=build_witness(record,w,run_tag=1,nonce=2,salt=3,action_salt=5)
ok('action_salt_is_private_input',a1['input']['action_salt']=='4' and 'action_salt' not in a1['expected'])
ok('distinct_openings_change_action_root',a1['expected']['action_root']!=a2['expected']['action_root'])
ok('state_and_public_rules_unmodified',a1['expected']['state_root']==a2['expected']['state_root'] and a1['expected']['rules_root']==a2['expected']['rules_root'])
ok('fixed_opening_reproduces_encoding',a1==build_witness(record,w,run_tag=1,nonce=2,salt=3,action_salt=4))
for name,bad in [('negative',-1),('field_limit',FIELD),('fractional',1.5),('bool',True)]:
    try:build_witness(record,w,run_tag=1,action_salt=bad)
    except ValueError:checks.append('reject_'+name+'_action_salt')
    else:raise AssertionError(name)
with tempfile.TemporaryDirectory() as td:
    f=generate('twodoors',2,td);text=f.read_text()
    ok('generator_private_input_and_domain', 'signal input action_salt;' in text and 'VectorCommit(4+N,1004);actions.salt <== action_salt;' in text)
    public=text.split('component main {public [')[-1].split(']')[0]
    ok('generator_public_abi_excludes_salt','action_salt' not in public)
report={'status':'HOST_CHECKS_PASS_CRYPTO_NOT_RUN','checks':checks,'count':len(checks),'hash':'serialization-only SHA-256 test double, not circuit Poseidon','limitations':'No Circom compilation, witness generation, proof generation, verification, security audit, or new timing measurements.'}
(root/'validation/HOST_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
