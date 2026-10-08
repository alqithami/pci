#!/usr/bin/env python3
"""Archive results, code, keys and checksums; deliberately exclude the large public PTAU."""
import argparse,hashlib,json,tarfile
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pci_bench.io import sha256,atomic_json
p=argparse.ArgumentParser();p.add_argument('results',nargs='?',default='results/deadline');p.add_argument('--partial',action='store_true');a=p.parse_args()
r=(ROOT/a.results).resolve()
if not (r/'plan.json').is_file():raise SystemExit('No plan.json at specified results directory')
if not a.partial and not (r/'SUITE_COMPLETE.json').exists():raise SystemExit('Incomplete run. Use --partial only to share diagnostics.')
paths=[]
for folder in [r,ROOT/'pci_bench',ROOT/'tests',ROOT/'configs',ROOT/'scripts',ROOT/'docs',ROOT/'crypto/build',ROOT/'crypto/generated']:
    if folder.exists():
        for f in folder.rglob('*'):
            if f.is_file() and not set(f.parts).intersection({'__pycache__','.pytest_cache'}) and f.name not in ('.suite.lock','bundle_manifest.json'):
                paths.append(f)
for rel in ['requirements.txt','requirements-lock.txt','crypto/package.json','crypto/package-lock.json','crypto/primitives.circom','crypto/worker.cjs','run_all.sh','setup_ubuntu.sh']:
    f=ROOT/rel
    if f.exists():paths.append(f)
paths=sorted(set(paths));manifest={}
def arc(f):
    if f.is_relative_to(r):return str(Path('results')/f.relative_to(r))
    return str(Path('source')/f.relative_to(ROOT))
for f in paths:manifest[arc(f)]=sha256(f)
meta=r/'bundle_manifest.json';atomic_json(meta,{'files':manifest,'partial':a.partial,
    'note':'Includes synthetic private witnesses for reproducibility; not for use with confidential real-world input.'})
out=ROOT/('PCI_'+r.name+('_partial' if a.partial else '_results')+'.tar.gz')
with tarfile.open(out,'w:gz') as tar:
    for f in paths:tar.add(f,arcname=arc(f),recursive=False)
    tar.add(meta,arcname='bundle_manifest.json')
digest=sha256(out);out.with_suffix(out.suffix+'.sha256').write_text(f'{digest}  {out.name}\n')
# Check every archived byte stream against the manifest now, not merely existence.
with tarfile.open(out,'r:gz') as tar:
    for name,expected in manifest.items():
        h=hashlib.sha256();stream=tar.extractfile(name)
        for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
        if h.hexdigest()!=expected:raise SystemExit(f'Archive verification failed: {name}')
print('ARCHIVE_VERIFIED:',out);print('SHA256:',digest);print('Files:',len(paths))
