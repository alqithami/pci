"""Read-only held-body receipt/source helpers; no scientific inference."""
import argparse,csv,hashlib,importlib.util,json,math,sys,os,stat
from contextvars import ContextVar
from collections import defaultdict
from pathlib import Path
sys.dont_write_bytecode=True
PINNED_SOURCE='e7fbf20f649ce40a21d42d644756cf519dd6fd2717564c5bbcf0eb0793473ed7'
PROFILES=('capacity_tight','nominal','quota_tight','rule_shift')
COUNTS=('physical_replacements','institutional_replacements','total_replacements','deliveries','pickups','joint_steps')

def need(x,message):
    if not x:raise ValueError(message)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
_SNAPSHOT=ContextVar('input_snapshot',default=None)
class InputSnapshot:
    def __init__(self):self.files={}
    @staticmethod
    def stamp(st):return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    def fresh(self,path):
        p=Path(path).absolute()
        need(p.resolve()==p,'SYMLINK_INPUT_PATH:'+str(p))
        before=p.lstat();need(stat.S_ISREG(before.st_mode),'NONREGULAR_INPUT:'+str(p))
        fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            first=os.fstat(fd);need(self.stamp(first)==self.stamp(before),'INPUT_CHANGED_BEFORE_READ:'+str(p))
            with os.fdopen(fd,'rb',closefd=False) as f:body=f.read()
            last=os.fstat(fd);after=p.lstat()
            need(self.stamp(first)==self.stamp(last)==self.stamp(after) and len(body)==first.st_size,'INPUT_CHANGED_DURING_READ:'+str(p))
        finally:os.close(fd)
        return body,self.stamp(last)
    def read(self,path):
        p=Path(path).absolute()
        if p not in self.files:
            body,stamp=self.fresh(p);self.files[p]=(body,stamp,hashlib.sha256(body).hexdigest())
        return self.files[p][0]
    def check(self):
        for p,(body,stamp,hashed) in self.files.items():
            current,now=self.fresh(p)
            need(now==stamp and current==body,'INPUT_CHANGED_AFTER_READ:'+str(p))
def read_bytes(p):
    snapshot=_SNAPSHOT.get()
    return snapshot.read(p) if snapshot is not None else Path(p).read_bytes()
def sha(p):return hashlib.sha256(read_bytes(p)).hexdigest()
def unique_pairs(pairs):
    result={}
    for k,v in pairs:
        need(k not in result,'DUPLICATE_JSON_KEY:'+k);result[k]=v
    return result

def load(p):
    p=Path(p);need(p.is_file() and not p.is_symlink(),'MISSING_REGULAR_INPUT:'+str(p))
    return json.loads(read_bytes(p),object_pairs_hook=unique_pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('NONFINITE_JSON:'+x)))

def load_protocol(source):
    source=Path(source)
    paths={p.relative_to(source).as_posix():sha(p) for folder in ('runtime','protocol') for p in sorted((source/folder).rglob('*.py')) if '__pycache__' not in p.parts}
    need(digest(paths)==PINNED_SOURCE,'UNQUALIFIED_SOURCE_MAP')
    sys.path.insert(0,str(source/'protocol'))
    for name in ('bridge_design','build_study','frontier'):
        spec=importlib.util.spec_from_file_location(name,source/'protocol'/f'{name}.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return paths,sys.modules['build_study'],sys.modules['frontier']

def keys_exact(rows,keys,fields,label):
    found=[tuple(row[k] for k in fields) for row in rows]
    need(len(found)==len(set(found)) and set(found)==set(keys),'INCOMPLETE_OR_DUPLICATE_'+label)
    return {k:r for k,r in zip(found,rows)}

