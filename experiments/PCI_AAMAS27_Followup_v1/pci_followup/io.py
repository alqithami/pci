import hashlib,json,os,platform,importlib.metadata
from pathlib import Path
from . import ROOT,BASE
from pci_bench.io import atomic_json,atomic_csv,read_csv_strict,sha256,canonical_hash

def sources():
    out={}
    for directory in ('pci_followup','tests','docs'):
        for p in sorted((ROOT/directory).rglob('*')):
            if p.is_file() and not set(p.parts).intersection({'__pycache__','.pytest_cache'}):out[str(p.relative_to(ROOT))]=sha256(p)
    for name in ('protocol.json','run_all.sh','launch_server.sh','requirements.txt'):
        if (ROOT/name).exists():out[name]=sha256(ROOT/name)
    out.update({'frozen_base/'+p.name:sha256(p) for p in sorted((BASE/'pci_bench').glob('*.py'))})
    return out

def versions():
    return {x:importlib.metadata.version(x) for x in ('torch','numpy','scipy','rware','gymnasium','networkx')}

def signature(config):return canonical_hash({'config':config,'sources':sources(),'versions':versions()})

def immutable_json(path,obj):
    p=Path(path)
    if p.exists() and json.loads(p.read_text())!=obj:raise RuntimeError(f'Refusing to mix versions: {p}')
    if not p.exists():atomic_json(p,obj)

def atomic_torch(path,obj):
    import torch
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,path)
