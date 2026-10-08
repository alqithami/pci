import hashlib,json,tarfile
from pathlib import Path
from . import ROOT,BASE
from .io import sha256,atomic_json
r=ROOT/'results/full'
d=json.loads((r/'SUITE_COMPLETE.json').read_text())
if d['status']!='COMPLETE':raise RuntimeError('Incomplete study')
paths=[]
for folder in ('pci_followup','tests','docs','results/full','validation','upstream'):
    for f in (ROOT/folder).rglob('*'):
        if f.is_file() and not set(f.parts).intersection({'__pycache__','.pytest_cache'}) and f.name not in ('.lock',):paths.append((f,str(f.relative_to(ROOT))))
for name in ('protocol.json','run_all.sh','launch_server.sh','requirements.txt','requirements-lock.txt','rware_installed_hashes.json','SOURCE_MANIFEST.json','LOCAL_VALIDATION.md'):
    if (ROOT/name).exists():paths.append((ROOT/name,name))
for f in (BASE/'pci_bench').glob('*.py'):paths.append((f,'frozen_base/pci_bench/'+f.name))
for f in (BASE/'crypto').glob('*.c*'):paths.append((f,'frozen_base/crypto/'+f.name))
for folder in ('configs','tests','crypto/generated'):
    for f in (BASE/folder).rglob('*'):
        if f.is_file() and not set(f.parts).intersection({'__pycache__','.pytest_cache'}):paths.append((f,'frozen_base/'+str(f.relative_to(BASE))))
for name in ('crypto/package.json','crypto/package-lock.json'):
    paths.append((BASE/name,'frozen_base/'+name))
for f in (BASE/'crypto/build').rglob('*'):
    if f.is_file() and f.suffix in ('.json','.r1cs','.sym','.wasm','.zkey'):paths.append((f,'frozen_base/'+str(f.relative_to(BASE))))
for s in (0,1):
    path=BASE/'results/deadline/runs'/f'pci_twodoors_l0.2_s{s}'
    for name in ('checkpoint.pt','config.json','TRAIN_COMPLETE.json'):
        f=path/name;paths.append((f,'frozen_checkpoints/'+path.name+'/'+name))
manifest={name:sha256(f) for f,name in paths};m=ROOT/'bundle_manifest.json'
atomic_json(m,{'files':manifest,'original_archive_sha256':json.loads((ROOT/'protocol.json').read_text())['original_archive_sha256']})
out=ROOT/'PCI_followup_results.tar.gz'
with tarfile.open(out,'w:gz') as tar:
    for f,name in paths:tar.add(f,arcname=name,recursive=False)
    tar.add(m,arcname='bundle_manifest.json')
with tarfile.open(out,'r:gz') as tar:
    for name,expected in manifest.items():
        stream=tar.extractfile(name);h=hashlib.sha256()
        for b in iter(lambda:stream.read(1<<20),b''):h.update(b)
        if h.hexdigest()!=expected:raise RuntimeError('Archive mismatch '+name)
out.with_suffix(out.suffix+'.sha256').write_text(sha256(out)+'  '+out.name+'\n')
print('FOLLOWUP_ARCHIVE_VERIFIED',out,'files',len(manifest),flush=True)
