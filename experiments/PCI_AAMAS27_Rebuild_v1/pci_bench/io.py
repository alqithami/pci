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
    root=Path(root)
    result = {str(p.relative_to(root)):sha256(p) for directory in ('pci_bench','crypto','configs','scripts','tests')
            for p in sorted((root/directory).rglob('*')) if p.is_file() and not set(p.relative_to(root).parts).intersection({'__pycache__','node_modules','vendor','bin','build','setup','generated'})}
    for name in ('run_all.sh','setup_ubuntu.sh','requirements.txt','pytest.ini'):
        if (root/name).exists():result[name]=sha256(root/name)
    return result


def system_meta():
    versions={}
    for name in ('torch','numpy','scipy','matplotlib','pytest'):
        try:versions[name]=md.version(name)
        except md.PackageNotFoundError:versions[name]=None
    return {'python':platform.python_version(),'platform':platform.platform(),'machine':platform.machine(),
            'cpu_count':os.cpu_count(),'versions':versions}


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
