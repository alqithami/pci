#!/usr/bin/env python3
"""Create a separate blinded-action prototype; never edit the frozen source.

Only the action commitment changes (private field salt and distinct domain 1004).
Public rule commitments remain public. This repair blocks the old deterministic
small-action dictionary, but is not a proof of application privacy or sensing.
"""
import argparse,hashlib,json,shutil
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
base=a.base.resolve();out=a.out.resolve()
expected={'crypto.py':'491ed24de682fb2db4a930e1c76fbca8377ebcb007fc3779cfd37239aa547008','crypto_generate.py':'abf1eb115f11e6cf9bb6e7f7a2a80e526cb4299fc0061dbf1ed3a632c28a0dc7'}
for name,h in expected.items():
 assert hashlib.sha256((base/'pci_bench'/name).read_bytes()).hexdigest()==h,name
if out.exists():raise SystemExit('Refusing to overwrite an existing prototype directory')
out.mkdir(parents=True)
for name in ('pci_bench','tests','configs'):
 shutil.copytree(base/name,out/name,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
(out/'crypto').mkdir();(out/'validation').mkdir()
for name in ('primitives.circom','worker.cjs','package.json','package-lock.json'):
 if (base/'crypto'/name).is_file():shutil.copy2(base/'crypto'/name,out/'crypto'/name)
for name in ('node_modules','bin'):
 if (base/'crypto'/name).exists():(out/'crypto'/name).symlink_to(base/'crypto'/name,target_is_directory=True)
# The public phase-1 transcript is reused by identity, never any old circuit key.
(out/'crypto/setup').mkdir()
for name in ('powersOfTau28_hez_final_16.ptau','verified.json'):
 if (base/'crypto/setup'/name).is_file():(out/'crypto/setup'/name).symlink_to(base/'crypto/setup'/name)

def change(file,old,new):
 t=file.read_text();assert t.count(old)==1,(file.name,old,t.count(old));file.write_text(t.replace(old,new))
gen=out/'pci_bench/crypto_generate.py';host=out/'pci_bench/crypto.py'
change(gen,'    signal input salt;','    signal input salt; signal input action_salt;')
change(gen,'component actions=VectorCommit(4+N,1002);actions.salt <== 0;',
 'component actions=VectorCommit(4+N,1004);actions.salt <== action_salt;')
change(host,'nonce=None,salt=None,strict=True','nonce=None,salt=None,action_salt=None,strict=True')
change(host,"    if not 0<=salt<FIELD:raise ValueError('Invalid state commitment salt')",
 "    if not 0<=salt<FIELD:raise ValueError('Invalid state commitment salt')\n    action_salt=secrets.randbelow(FIELD) if action_salt is None else integer(action_salt,'action_salt')\n    if not 0<=action_salt<FIELD:raise ValueError('Invalid action commitment salt')")
change(host,'action_root=worker.commit(1002,action_values)','action_root=worker.commit(1004,action_values,action_salt)')
change(host,"'capacities':capacities,'action':actions,'salt':salt,","'capacities':capacities,'action':actions,'salt':salt,'action_salt':action_salt,")
manifest={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.rglob('*')) if f.is_file() and f.suffix in ('.py','.circom','.cjs') and not f.is_symlink()}
(out/'PROTOTYPE.json').write_text(json.dumps({'status':'PREPARED_NOT_VALIDATED','variant':'blinded-action-domain1004','base_source_hashes':expected,'files':manifest,'scope':'new circuit keys required; not the recorded experimental protocol; no new learning results'},indent=2)+'\n')
print('BLINDED_VARIANT_PREPARED',out)
