#!/usr/bin/env python3
"""Host/protocol tests. Never substitute these for genuine prover validation."""
import copy, itertools, json, tempfile
from pathlib import Path
import numpy as np
import benchmark as b
from pci_bench.environment import Warehouse, EnvConfig, monitor_record
from pci_bench.crypto_check import samples
from cryptography.exceptions import InvalidSignature

def main():
    checks=[];rng=np.random.default_rng(119304)
    good,bad=samples()
    for name,rec in [('valid',good),*bad.items()]:
        result=b.oracle(rec);ref=monitor_record(rec)
        assert result['compliant']==ref['compliant']
        assert result['next_positions']==ref['next_positions'] and result['entries']==ref['entries']
        checks.append('oracle_fixture_'+name)
    transitions=0
    for n,regime in itertools.product((2,4,8),('nominal','quota_tight','rule_shift')):
        env=Warehouse(EnvConfig(n_agents=n,horizon=256,quota=1 if regime=='quota_tight' else 3,rule_change=regime=='rule_shift'),1729)
        for _ in range(256):
            a=rng.integers(0,5,n)
            c,_=env.candidates(a);p,_=env.physical_resolve(c);star=env.shield(c)
            predicted=env.actions_for_positions(star)
            env.transition(a,enforce=True)
            rec=env.last;out=b.oracle(rec)
            assert out['compliant'] and out['next_positions']==env.pos.tolist() and rec['actions']==predicted
            transitions+=1
        checks.append('oracle_256_transitions_n'+str(n)+'_'+regime)
    key=b.signing_key();payload={'version':1,'compliant':True}
    env,elapsed,size,public=b.sign_envelope(payload,key);b.verify_signature(env)
    assert elapsed>=0 and size==64 and public>0
    changed=copy.deepcopy(env);changed['payload']['compliant']=False
    try:b.verify_signature(changed)
    except InvalidSignature:pass
    else:raise AssertionError('Modified signed payload accepted')
    checks.append('real_ed25519_tamper')
    e,_,_,_=b.sign_envelope({'compliant':True,'record':bad['capacity']},key);b.verify_signature(e)
    assert not b.oracle(e['payload']['record'])['compliant']
    checks.append('authentic_false_assertion_distinguished')
    p=b.PROTOCOL;assert p['full_episodes']==len(p['methods'])*len(p['training_seeds'])*len(p['populations'])*len(p['regimes'])*p['episodes_per_cell']
    assert p['certified_transitions']==p['full_episodes']*p['horizon']
    assert p['paired_records']==p['full_episodes']*len(p['paired_steps'])
    assert p['backend_measurements']==p['certified_transitions']+3*p['paired_records']
    assert p['saved_snark_proofs']==p['certified_transitions']+p['paired_records']
    orders=list(itertools.permutations(p['backends']));assert len(orders)==len(p['paired_steps'])==24
    for offset in range(24):assert len({orders[(k+offset)%24] for k in range(24)})==24
    for boundary in (64,128,192):assert boundary-1 in p['paired_steps'] and boundary in p['paired_steps'] and boundary+1 in p['paired_steps']
    checks.append('protocol_counts_order_and_reset_coverage')
    with tempfile.TemporaryDirectory() as t:
        path=Path(t)/'table.csv';b.atomic_csv(path,[{'a':1,'b':2}]);assert b.read_rows(path)==[{'a':'1','b':'2'}]
        try:b.atomic_csv(path,[{'a':1},{'b':2}])
        except ValueError:pass
        else:raise AssertionError('Schema mismatch ignored')
        path.write_text('a,b\n1,2,3\n')
        try:b.read_rows(path)
        except ValueError:pass
        else:raise AssertionError('Malformed CSV silently accepted')
    checks.append('strict_csv_schema')
    report={'status':'PASS','checks':checks,'count':len(checks),'oracle_random_transitions':transitions,
            'scope':'Host integration, native physical transition oracle, real signatures; no SNARK proof test or model learning in these tests.'}
    b.atomic_json(b.ROOT/'validation/HOST_TESTS.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
