"""Atomic writes, explicit schemas, and provenance. No tolerant CSV parsing."""
from __future__ import annotations
import csv, hashlib, json, os, platform, subprocess, tempfile
from pathlib import Path
import importlib.metadata as md


def atomic_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def atomic_csv(path, rows, fields=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows and fields is None: raise ValueError(f'Refusing to write schema-less empty CSV: {path}')
    fields=list(fields or rows[0].keys())
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='raise'); w.writeheader()
        for row in rows:
            if set(row)!=set(fields): raise ValueError(f'Schema changed at {path}: {set(row)^set(fields)}')
            w.writerow(row)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def source_hashes(root):
    """Exact import closure of this separately versioned training runtime."""
    root=Path(root)
    names=["vectorized_admission.py"] + ["pci_risk_train/"+n for n in
        ("__init__.py","environment.py","model.py","objectives.py","trainer.py","io.py")]
    return {n:sha256(root/n) for n in sorted(names)}


def system_meta():
    """Separate imported runtime identity from distribution-index metadata.

    A distribution record can describe a different installation than the module
    actually imported on sys.path. Only the actual imported versions enter the
    training signature's ``versions`` field. Distribution records remain useful
    diagnostics and are labeled separately rather than asserted as runtime facts.
    """
    import sys
    import numpy as np
    import torch
    imported={
        "numpy":{"version":str(np.__version__),"file":str(np.__file__)},
        "torch":{"version":str(torch.__version__),"file":str(torch.__file__)},
    }
    actual_versions={name:value["version"] for name,value in imported.items()}
    actual_versions["python"]=platform.python_version()
    distribution_versions={}
    for name in ("torch","numpy","scipy","matplotlib","pytest"):
        try:distribution_versions[name]=md.version(name)
        except md.PackageNotFoundError:distribution_versions[name]=None
    discrepancies=[{"module":name,"imported_version":value["version"],
                    "distribution_metadata_version":distribution_versions[name],
                    "imported_file":value["file"]}
                   for name,value in imported.items()
                   if distribution_versions[name]!=value["version"]]
    return {"python":platform.python_version(),"python_executable":sys.executable,
            "python_full_version":sys.version,"platform":platform.platform(),
            "machine":platform.machine(),"cpu_count":os.cpu_count(),
            "versions":actual_versions,"imported_modules":imported,
            "distribution_metadata_versions":distribution_versions,
            "module_distribution_version_discrepancies":discrepancies,
            "signature_version_scope":"Actual imported NumPy/Torch and running Python versions; distribution-index metadata is diagnostic only"}


def read_csv_strict(path):
    path=Path(path)
    with path.open(newline='',encoding='utf-8') as f:
        r=csv.reader(f,strict=True)
        try: header=next(r)
        except StopIteration: raise ValueError(f'Empty CSV: {path}')
        if len(set(header))!=len(header) or not all(header):raise ValueError(f'Invalid header: {path}')
        rows=[]
        for row in r:
            if len(row)!=len(header):raise ValueError(f'{path}:{r.line_num}: expected {len(header)} columns, got {len(row)}')
            rows.append(dict(zip(header,row)))
    return rows


def atomic_torch(path, value):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("wb") as f:
        torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def atomic_npz(path, arrays, *, verify_existing=False):
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if not verify_existing:
            raise RuntimeError("Refusing to replace an existing closed NPZ: "+str(path))
        with np.load(path,allow_pickle=False) as old:
            if set(old.files)!=set(arrays) or any(not np.array_equal(old[k],v,equal_nan=False) for k,v in arrays.items()):
                raise RuntimeError("Uncommitted retained rollout differs on exact resume: "+str(path))
        return sha256(path)
    tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    return sha256(path)
