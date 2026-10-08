import json,copy,hashlib
from pathlib import Path
import pytest
from pci_bench.io import atomic_csv,read_csv_strict
from pci_bench.crypto import build_witness,FIELD,integer
from pci_bench.crypto_check import samples
from pci_bench.crypto_generate import generate
from pci_bench.crypto_benchmark import signed_disclosure
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

class EncodingOnlyTestHasher:
    """Unit-test double for parsing ONLY. Never used by production proof code."""
    def hash(self,values):
        return int.from_bytes(hashlib.sha256(json.dumps(values).encode()).digest(),'big')%FIELD
    def commit(self,domain,values,salt=0):return self.hash([domain,values,salt])

def test_witness_metadata_not_sent_as_circuit_input():
    g,_=samples();w=build_witness(g,EncodingOnlyTestHasher(),run_tag=1,nonce=2,salt=3)
    assert 'expected' not in w['input'] and 'ok' not in w['input'] and 'record_hash' not in w['input']
    assert w['expected']['ok']=='1'
    assert len(w['input']['x'])==2

@pytest.mark.parametrize('value',[1.2,float('nan'),True,'1.2'])
def test_integer_does_not_truncate(value):
    with pytest.raises((TypeError,ValueError)):integer(value,'bad')

def test_witness_invalid_predicate_raises():
    _,cases=samples()
    for g in cases.values():
        with pytest.raises(ValueError):build_witness(g,EncodingOnlyTestHasher(),run_tag=1)

def test_witness_unsigned_bounds_enforced():
    g,_=samples();g['before']['counts'][0][0]=65536
    with pytest.raises(ValueError):build_witness(g,EncodingOnlyTestHasher(),run_tag=1,strict=False)

def test_exact_csv_schema_no_silent_discard(tmp_path):
    p=tmp_path/'a.csv'
    atomic_csv(p,[{'run_id':'x','value':1}]);assert len(read_csv_strict(p))==1
    with pytest.raises(ValueError):atomic_csv(p,[{'a':1},{'a':2,'b':3}])
    p.write_text('a,b\n1,2,3\n')
    with pytest.raises(ValueError,match='expected 2'):read_csv_strict(p)

def test_missing_or_duplicate_csv_header_rejected(tmp_path):
    p=tmp_path/'a.csv';p.write_text('a,a\n1,2\n')
    with pytest.raises(ValueError):read_csv_strict(p)
    p.write_text('')
    with pytest.raises(ValueError):read_csv_strict(p)

@pytest.mark.parametrize('n',[2,4,8,12])
def test_circuit_generation_schema(tmp_path,n):
    source=generate('twodoors',n,tmp_path)
    text=source.read_text();spec=json.loads(source.with_suffix('.spec.json').read_text())
    assert f'WarehouseTransition({n})' in text
    assert 'state.root === state_root' in text
    assert 'actions.root === action_root' in text
    assert 'rules.root === rules_root' in text
    assert 'capLT[r].out === 0' in text
    assert 'Num2Bits(16)' in text
    assert len(spec['public_names'])==10

def test_actual_signature_and_monitor():
    g,_=samples();r=signed_disclosure(g,Ed25519PrivateKey.generate())
    assert r['valid'] is True and r['proof_json_bytes']==64
