#!/usr/bin/env python3
"""Bounded, source-bound training -> complete evaluation -> retained audit.

This Linux controller has no cloud API, resource-creation or authorization
capability. Scientific execution requires independently supplied, root-reviewed
source/design, CUDA-readiness and launch records. Engineering execution is
CPU-only and cannot produce a scientific completion record.
"""
from __future__ import annotations
import argparse
import ctypes
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import time

SCHEMA = 'PCI-FAIR-COHORT-CONTROLLER-CONFIG-v1'
STAGES = ('engineering','development','confirmation','replication','estimator_ablation')
PID_STATE_DEAD = ('Z','X')

def need(condition, message):
    if not condition: raise RuntimeError(message)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def digest(value): return hashlib.sha256(canonical(value)).hexdigest()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def read_bound(path):
    """Parse and hash the same bytes from one stable regular held inode."""
    p=Path(path)
    def fingerprint(s):
        return (s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC|os.O_NONBLOCK),'rb') as held:
        before=os.fstat(held.fileno());named_before=p.lstat()
        need(stat.S_ISREG(before.st_mode) and before.st_nlink>0 and
             fingerprint(before)==fingerprint(named_before),'JSON_HELD_INODE_PATH_MISMATCH:'+str(p))
        body=held.read()
        after=os.fstat(held.fileno());named_after=p.lstat()
        need(fingerprint(before)==fingerprint(after)==fingerprint(named_after),
             'JSON_HELD_INODE_CHANGED_DURING_READ:'+str(p))
    return json.loads(body),hashlib.sha256(body).hexdigest()

def read(path): return read_bound(path)[0]

def atomic(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.writing')
    with t.open('wb') as f:
        f.write(canonical(value)+b'\n');f.flush();os.fsync(f.fileno())
    os.replace(t,p)
    fd=os.open(p.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

_CLOSED_HELPERS={}

def closed(path,value,*,runtime_root=None):
    """Use the qualified anonymous-inode immutable publisher, then read back.

    A public pathname must never expose a partially written immutable receipt.
    Mutable operational state uses atomic() and is not a completion authority.
    """
    helper_path=Path(runtime_root or Path(__file__).resolve().parent)/'evaluation_fair/common.py'
    expected_source=sha(helper_path);key=(str(helper_path),expected_source)
    if key not in _CLOSED_HELPERS:
        name='_cohort_immutable_io_'+hashlib.sha256(str(helper_path).encode()).hexdigest()
        spec=importlib.util.spec_from_file_location(name,helper_path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        need(sha(helper_path)==expected_source,'IMMUTABLE_PUBLISHER_CHANGED_DURING_IMPORT')
        _CLOSED_HELPERS[key]=module.closed_json
    result=_CLOSED_HELPERS[key](path,value)
    need(sha(path)==hashlib.sha256(canonical(value)+b'\n').hexdigest(),
         'IMMUTABLE_RECEIPT_READBACK_DIFFERS:'+str(path))
    return result

def exact_path(value, *, existing=False):
    p=Path(value)
    need(p.is_absolute() and '..' not in p.parts,'ABSOLUTE_NARROW_PATH_REQUIRED')
    need(str(p) not in ('/','/tmp','/home','/home/ubuntu','/workspace'),'NARROW_PATH_REQUIRED')
    need(p.resolve()==p,'SYMLINK_OR_NONCANONICAL_PATH:'+str(p))
    if existing:need(p.exists(),'PATH_MISSING:'+str(p))
    return p

def safe_relative(value):
    p=Path(value)
    need(not p.is_absolute() and p.parts and '..' not in p.parts,'SAFE_RELATIVE_PATH_REQUIRED')
    return p

def source_map(project):
    project=Path(project)
    return {p.relative_to(project).as_posix():sha(p)
            for folder in ('runtime','protocol')
            for p in sorted((project/folder).rglob('*.py'))
            if '__pycache__' not in p.parts}

_PID_NAMESPACE=os.readlink('/proc/self/ns/pid')
_NSPID_LINE=next(x for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('NSpid:'))
_PID_DEPTH=len(_NSPID_LINE.split())-2

def _proc_identity(proc_pid):
    """Keep namespace signal PIDs distinct from the mounted /proc view."""
    p=Path('/proc')/str(proc_pid)
    try:
        fields=(p/'stat').read_text().rsplit(')',1)[1].split()
        values={}
        for line in (p/'status').read_text().splitlines():
            if line.startswith(('NSpid:','NSpgid:','NSsid:')):
                key,body=line.split(':',1);values[key]=[int(x) for x in body.split()]
        if any(len(values.get(k,[]))<=_PID_DEPTH for k in ('NSpid','NSpgid','NSsid')):return None
        if os.readlink(p/'ns/pid')!=_PID_NAMESPACE:return None
        # cwd/cmdline are diagnostic metadata, not process identity. Linux
        # releases fs/mm during normal exit before /proc/stat necessarily says
        # Z; a missing cwd must not erase a verified namespace/birth identity.
        # Namespace, PID/group/session and stat reads above remain mandatory.
        try: cwd=os.readlink(p/'cwd') if fields[0]not in PID_STATE_DEAD else None
        except (FileNotFoundError,ProcessLookupError):cwd=None
        try: cmdline=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')[:4096]
        except (FileNotFoundError,ProcessLookupError):cmdline=None
        return {'pid':values['NSpid'][_PID_DEPTH],'proc_pid':int(proc_pid),
                'birth_ticks':int(fields[19]),'state':fields[0],
                'proc_ppid':int(fields[1]),'pgid':values['NSpgid'][_PID_DEPTH],
                'proc_pgid':int(fields[2]),'session':values['NSsid'][_PID_DEPTH],
                'pid_namespace':_PID_NAMESPACE,
                'cwd':cwd,'cmdline':cmdline}
    except FileNotFoundError:return None

def identity(pid):
    if pid==os.getpid():return _proc_identity(int(os.readlink('/proc/self')))
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            line=next((x for x in (p/'status').read_text().splitlines() if x.startswith('NSpid:')),None)
            values=[int(x) for x in line.split()[1:]] if line else []
            if len(values)<=_PID_DEPTH or values[_PID_DEPTH]!=pid:continue
            row=_proc_identity(int(p.name))
            if row is not None:return row
        except (PermissionError,FileNotFoundError,ProcessLookupError):continue
    return None

def boot_id():return Path('/proc/sys/kernel/random/boot_id').read_text().strip()

def group_members(pgid,proc_pgid):
    result=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            stat=(p/'stat').read_text().rsplit(')',1)[1].split()
            if int(stat[2])!=proc_pgid or stat[0]in PID_STATE_DEAD:continue
            r=_proc_identity(int(p.name))
            if r is None:
                # A worker may exit after the first /proc/stat read. Recheck
                # that same process before classifying a missing identity as
                # a namespace fault. A still-live unresolved member continues
                # to fail closed; it is never adopted or signalled here.
                latest=(p/'stat').read_text().rsplit(')',1)[1].split()
                if latest[0]in PID_STATE_DEAD or int(latest[2])!=proc_pgid:continue
            need(r is not None and r['pgid']==pgid,'OWNED_PROCESS_GROUP_NAMESPACE_CHANGED')
        except (FileNotFoundError,ProcessLookupError):continue
        if r and r['pgid']==pgid and r['state'] not in PID_STATE_DEAD:result.append(r)
    return result

def check_gpu(uuid):
    r=subprocess.run(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],
                     capture_output=True,text=True,timeout=15)
    need(r.returncode==0,'GPU_QUERY_FAILED')
    rows=[x.strip() for x in r.stdout.splitlines() if x.strip()]
    need(len(rows)==1 and rows[0].split(',')[0].strip()==uuid
         and rows[0].split(',',1)[1].strip()=='NVIDIA L4',
         'EXACT_OWNED_L4_READINESS_REQUIRED')

def _child_parent_death_signal(expected_parent):
    # subprocess invokes this in the single-threaded parent before exec.
    if os.getppid()!=expected_parent:os._exit(125)
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(1,signal.SIGTERM,0,0,0)!=0:os._exit(125)
    if os.getppid()!=expected_parent:os._exit(125)

def configuration(raw):
    c=dict(raw)
    required=('schema','run_id','role','stage','study_domain','device','project_root',
              'case_manifest','out','deadline_unix','per_case_seconds','stage_seconds',
              'max_workers','evaluation_workers','worlds_per_profile','expected_cells',
              'planning_bytes_per_policy','planning_evaluation_bytes','disk_min_free_bytes',
              'disk_reserve_fraction')
    need(all(k in c for k in required),'EXPLICIT_CONTROLLER_CONFIGURATION_REQUIRED')
    need(c['schema']==SCHEMA and c['role'] in ('engineering','scientific'),'CONTROLLER_SCHEMA_OR_ROLE')
    need(c['stage'] in STAGES and c['study_domain'] in ('warehouse','rware'),'STAGE_DOMAIN')
    need(re.fullmatch(r'[A-Za-z0-9_-]{1,100}',c['run_id']) is not None,'SAFE_RUN_ID')
    need(c['device']==('cpu' if c['role']=='engineering' else 'cuda'),'CPU_ENGINEERING_OR_CUDA_SCIENCE')
    need((c['stage']=='engineering')==(c['role']=='engineering'),'STAGE_ROLE_ISOLATION')
    for k in ('max_workers','evaluation_workers'):
        need(type(c[k])is int and 1<=c[k]<=4,'WORKER_CEILING_1_TO_4')
    need(type(c['worlds_per_profile'])is int and 1<=c['worlds_per_profile']<=50,'WORLD_COUNT')
    for k in ('deadline_unix','per_case_seconds','stage_seconds'):
        need(type(c[k])in (int,float) and math.isfinite(c[k]) and c[k]>0,'FINITE_TIME_BOUND:'+k)
    need(c['deadline_unix']>time.time(),'EXPIRED_NEW_ABSOLUTE_DEADLINE')
    for k in ('planning_bytes_per_policy','planning_evaluation_bytes','disk_min_free_bytes'):
        need(type(c[k])is int and c[k]>=0,'NONNEGATIVE_DISK_BYTES:'+k)
    need(type(c['disk_reserve_fraction'])in (int,float) and 0<=c['disk_reserve_fraction']<=5,'DISK_RESERVE_FRACTION')
    project=exact_path(c['project_root'],existing=True)
    manifest=exact_path(c['case_manifest'],existing=True)
    out=exact_path(c['out'])
    need(not out.is_relative_to(project/'runtime') and not out.is_relative_to(project/'protocol'),
         'OUTPUT_MUST_BE_OUTSIDE_EXECUTABLE_CODE')
    need(not manifest.is_relative_to(out),'INPUT_MANIFEST_OUTSIDE_OUTPUT')
    need(sys.platform.startswith('linux'),'LINUX_PROC_IDENTITY_REQUIRED')
    return c,project,manifest,out

def validate_ledger(c, manifest_path, *, bound_manifest=None):
    m,_=bound_manifest if bound_manifest is not None else read_bound(manifest_path)
    need(m.get('schema')=='PCI-COMPLETE-CASE-MANIFEST-v1' and m.get('stage')==c['stage'],'MANIFEST_SCHEMA_STAGE')
    rows=m.get('cases')
    need(type(rows)is list and rows and m.get('case_count')==len(rows),'COMPLETE_NONEMPTY_CASE_LEDGER')
    need(len({r['case_id'] for r in rows})==len(rows),'DUPLICATE_CASE_ID')
    expected=set()
    for cell in c['expected_cells']:
        need(cell['domain']==c['study_domain'] and 1<=cell['n_agents']<=32,'EXPECTED_CELL_DOMAIN_N')
        ids=cell['candidate_ids'];seeds=cell['training_seeds']
        need(len(ids)==len(set(ids)) and len(seeds)==len(set(seeds)) and ids and seeds,'EXACT_CANDIDATE_SEED_LISTS')
        for candidate in ids:
            need(re.fullmatch(r'[A-Za-z0-9_-]{1,80}',candidate) is not None,'SAFE_CANDIDATE')
            for seed in seeds:
                key=(cell['domain'],cell['n_agents'],candidate,seed)
                need(key not in expected,'DUPLICATE_EXPECTED_CELL');expected.add(key)
    actual=set();configs={}
    role='engineering' if c['role']=='engineering' else ('development' if c['stage']=='development' else 'confirmatory')
    for row in rows:
        need(re.fullmatch(r'[A-Za-z0-9_-]{1,100}',row['case_id']) is not None,'SAFE_CASE_ID')
        path=manifest_path.parent/safe_relative(row['path'])
        need(path.resolve()==path and path.is_file(),'CASE_CONFIG_PATH:'+row['case_id'])
        cfg,config_sha256=read_bound(path)
        need(config_sha256==row['sha256'],'CASE_CONFIG_IDENTITY:'+row['case_id'])
        need(cfg.get('run_id')==row['case_id'] and cfg.get('run_role')==role and row['run_role']==role,'CASE_RUN_ID_ROLE')
        need(cfg.get('device')==c['device'] and cfg.get('torch_threads',1)==1,'CASE_DEVICE_THREAD_BOUND')
        key=(cfg['domain'],cfg['n_agents'],row['candidate_id'],cfg['seed'])
        need(key not in actual,'DUPLICATE_CASE_COORDINATE');actual.add(key)
        need(row['total_steps']==cfg['total_steps'] and row['training_seed']==cfg['seed']
             and row['domain']==cfg['domain'] and row['n_agents']==cfg['n_agents'],'CASE_DESCRIPTOR_MISMATCH')
        need(row['family']==cfg['method'],'METHOD_FAMILY_MISMATCH')
        need(cfg['total_steps']>0,'POSITIVE_BUDGET_REQUIRED')
        if c['role']=='engineering':need(cfg['total_steps']<=8192,'ENGINEERING_ONLY_SMALL_CASE_BUDGET')
        else:need(cfg.get('protocol_sha256')==m.get('protocol_canonical_sha256'),'SCIENTIFIC_PROTOCOL_BINDING')
        configs[row['case_id']]={'path':str(path),'configuration':cfg,'descriptor':row}
    need(actual==expected,'MISSING_OR_UNEXPECTED_CASE_COORDINATES')
    return m,configs

def validate_authority(c, project, manifest_path, current_sources, *, bound_manifest=None):
    """Read supplied records; this function never creates authorizations."""
    manifest,mh=bound_manifest if bound_manifest is not None else read_bound(manifest_path)
    smh=digest(current_sources)
    if c['role']=='engineering':
        return {'engineering_only':True,'source_map_sha256':smh,'case_manifest_sha256':mh}
    for k in ('source_design_lock','launch_lock','readiness_receipt'):
        need(k in c,'SCIENTIFIC_EXTERNAL_RECORD_REQUIRED:'+k)
        exact_path(c[k],existing=True)
    design,design_hash=read_bound(c['source_design_lock'])
    launch,launch_hash=read_bound(c['launch_lock'])
    ready,ready_hash=read_bound(c['readiness_receipt'])
    need(design.get('schema')=='PCI-COMPLETE-SOURCE-DESIGN-LOCK-v1'
         and design.get('scientific_launch_authorized')is True and design.get('engineering_complete')is True,
         'ROOT_REVIEWED_SOURCE_DESIGN_LOCK_REQUIRED')
    need(design.get('source_sha256')==current_sources and design.get('source_map_sha256')==smh,'DESIGN_EXECUTION_SOURCE_MAP_MISMATCH')
    need(design.get('protocol_canonical_sha256')==manifest.get('protocol_canonical_sha256'),'DESIGN_MANIFEST_PROTOCOL_MISMATCH')
    need('protocol_file' in c,'SCIENTIFIC_PROTOCOL_FILE_REQUIRED')
    protocol_path=exact_path(c['protocol_file'],existing=True)
    protocol,protocol_hash=read_bound(protocol_path)
    need(digest(protocol)==design['protocol_canonical_sha256'],'EXECUTING_PROTOCOL_BODY_MISMATCH')
    need(launch.get('schema')=='PCI-FAIR-SCIENTIFIC-LAUNCH-v1' and launch.get('scientific_launch_authorized')is True,
         'INDEPENDENT_ROOT_LAUNCH_LOCK_REQUIRED')
    comparisons={'controller_config_sha256':digest(c),'case_manifest_sha256':mh,
                 'source_design_lock_sha256':design_hash,
                 'readiness_receipt_sha256':ready_hash,
                 'source_map_sha256':smh,'out':c['out'],'deadline_unix':c['deadline_unix']}
    need(all(launch.get(k)==v for k,v in comparisons.items()),'ROOT_LAUNCH_BINDING_MISMATCH')
    need(ready.get('schema')=='PCI-FAIR-CUDA-READINESS-v1' and ready.get('status')=='PASS'
         and ready.get('source_map_sha256')==smh and ready.get('all_required_methods_pass')is True,
         'ACTUAL_CUDA_METHOD_READINESS_REQUIRED')
    need(ready.get('host_boot_id')==boot_id()==launch.get('host_boot_id'),'READINESS_HOST_BOOT_MISMATCH')
    need(ready.get('gpu_uuid')==launch.get('gpu_uuid') and ready.get('device')=='cuda','READINESS_GPU_BINDING_MISMATCH')
    need(type(ready.get('runtime_versions'))is dict and ready['runtime_versions'],'READINESS_RUNTIME_VERSIONS_REQUIRED')
    need(ready.get('python_executable')==sys.executable and
         ready.get('python_executable_sha256')==sha(Path(sys.executable).resolve()),
         'READINESS_INTERPRETER_IDENTITY_CHANGED')
    coverage=ready.get('covered_training_shapes')
    need(type(coverage)is list and coverage,'TYPED_CUDA_READINESS_COVERAGE_REQUIRED')
    fields=('domain','n_agents','method','risk_estimator','shield_policy')
    covered={tuple(row.get(k) for k in fields) for row in coverage}
    for row in manifest['cases']:
        cfg,config_sha256=read_bound(manifest_path.parent/safe_relative(row['path']))
        need(config_sha256==row['sha256'],'READINESS_CASE_CONFIG_IDENTITY:'+row['case_id'])
        shape=tuple(cfg.get(k,'exact' if k in ('risk_estimator','shield_policy') else None) for k in fields)
        need(shape in covered,'CUDA_READINESS_MISSING_TRAINING_SHAPE:'+repr(shape))
    sys.path.insert(0,str(project/'runtime'))
    from training.pci_risk_train.fair_trainer import fair_system_meta
    actual_runtime=fair_system_meta(c['study_domain'])
    need(ready['runtime_versions']==actual_runtime['versions'],'READINESS_RUNTIME_VERSIONS_CHANGED')
    need(ready.get('dependency_source_sha256')==actual_runtime['dependency_source_sha256'],
         'READINESS_NATIVE_DEPENDENCY_SOURCE_CHANGED')
    check_gpu(ready['gpu_uuid'])
    return {'engineering_only':False,**comparisons,'host_boot_id':boot_id(),'gpu_uuid':ready['gpu_uuid'],
            'protocol_sha256':design['protocol_canonical_sha256'],
            'external_record_sha256':{c['source_design_lock']:design_hash,c['launch_lock']:launch_hash,
                c['readiness_receipt']:ready_hash,str(protocol_path):protocol_hash}}

class Controller:
    def __init__(self,value,config_path):
        self.c,self.project,self.manifest_path,self.out=configuration(value)
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            os.environ[name]='1'
        import torch
        torch.set_num_threads(1)
        self.config_path=exact_path(config_path,existing=True)
        config_body,self.config_file_sha256=read_bound(self.config_path)
        need(digest(config_body)==digest(self.c),'CONFIG_INPUT_CHANGED')
        bound_manifest=read_bound(self.manifest_path)
        self.manifest,self.configs=validate_ledger(self.c,self.manifest_path,bound_manifest=bound_manifest)
        self.sources=source_map(self.project)
        self.authority=validate_authority(self.c,self.project,self.manifest_path,self.sources,
                                          bound_manifest=bound_manifest)
        self.out.mkdir(parents=True,exist_ok=True)
        need(self.out.resolve()==self.out,'OUTPUT_SYMLINK_CHANGED')
        self.lock=(self.out/'COHORT.lock').open('a+')
        fcntl.flock(self.lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.binding={'controller_configuration':self.c,'controller_configuration_file_sha256':self.config_file_sha256,
                      'case_manifest_sha256':bound_manifest[1],
                      'source_sha256':self.sources,'authority':self.authority}
        closed(self.out/'EXECUTION_LOCK.json',self.binding,runtime_root=self.project/'runtime')
        self.state_path=self.out/'COHORT_STATE.json'
        self.s=read(self.state_path) if self.state_path.exists() else {
            'schema':'PCI-FAIR-COHORT-STATE-v1','run_id':self.c['run_id'],'status':'STARTED',
            'started_unix':time.time(),'controller_pid':os.getpid(),
            'controller_birth_ticks':identity(os.getpid())['birth_ticks'],'host_boot_id':boot_id(),
            'deadline_unix':self.c['deadline_unix'],'completed_training':{},'events':[]}
        need(self.s['deadline_unix']==self.c['deadline_unix'] and self.s['host_boot_id']==boot_id(),'RESTART_HOST_OR_DEADLINE_CHANGED')
        # A restarted controller must not adopt an unobserved live old child.
        for old in self.s.get('active_processes',[]):
            need(old.get('pid_namespace')==_PID_NAMESPACE,
                 'RECORDED_CHILD_PID_NAMESPACE_CHANGED_REQUIRES_REVIEW')
            r=identity(old['pid'])
            need(r is None or r['state']in PID_STATE_DEAD or r['birth_ticks']!=old['birth_ticks'],
                 'RECORDED_PREVIOUS_CHILD_STILL_LIVE_REQUIRES_REVIEW')
        current=identity(os.getpid())
        need(current is not None,'CONTROLLER_PROCESS_IDENTITY_REQUIRED')
        previous={k:self.s.get(k) for k in ('controller_pid','controller_birth_ticks','controller_proc_pid')}
        self.s.setdefault('controller_invocations',[]).append({
            'at_unix':time.time(),'previous_identity':previous,'identity':current})
        self.s.update(controller_pid=current['pid'],controller_birth_ticks=current['birth_ticks'],
                      controller_proc_pid=current['proc_pid'],pid_namespace=current['pid_namespace'])
        self.active={};self.last_source_check=0.
        self.interrupted=None
        for sig in (signal.SIGTERM,signal.SIGINT):
            signal.signal(sig,self._interrupt)
        self.save('PREFLIGHT_COMPLETE')

    def _interrupt(self,sig,_frame):self.interrupted=signal.Signals(sig).name

    def save(self,status,**updates):
        self.s.update(updates);self.s['status']=status;self.s['updated_unix']=time.time()
        self.s['active_processes']=[v['record'] for v in self.active.values()]
        self.s['events'].append({'status':status,'at_unix':self.s['updated_unix']})
        atomic(self.state_path,self.s)

    def guard(self, *, force_sources=False):
        need(not self.interrupted,'CONTROLLER_INTERRUPTED:'+str(self.interrupted))
        need(time.time()<self.c['deadline_unix'],'NEW_ABSOLUTE_COHORT_DEADLINE')
        need(self.s['host_boot_id']==boot_id(),'HOST_REBOOTED')
        need(self.out.resolve()==self.out and not self.out.is_symlink(),'OUTPUT_PATH_CHANGED')
        need(sha(self.config_path)==self.config_file_sha256,'CONTROLLER_CONFIGURATION_CHANGED_DURING_RUN')
        need(sha(self.manifest_path)==self.binding['case_manifest_sha256'],'CASE_MANIFEST_CHANGED_DURING_RUN')
        if force_sources or time.monotonic()-self.last_source_check>5:
            need(source_map(self.project)==self.sources,'EXECUTABLE_SOURCE_CHANGED_DURING_RUN')
            for case in self.configs.values():
                need(sha(case['path'])==case['descriptor']['sha256'],
                     'CASE_CONFIGURATION_CHANGED_DURING_RUN:'+case['descriptor']['case_id'])
            for path,expected_hash in self.authority.get('external_record_sha256',{}).items():
                need(sha(path)==expected_hash,'EXTERNAL_AUTHORITY_RECORD_CHANGED_DURING_RUN:'+path)
            self.last_source_check=time.monotonic()
        if force_sources and self.c['role']=='scientific':
            check_gpu(self.authority['gpu_uuid'])
        need(shutil.disk_usage(self.out).free>=self.c['disk_min_free_bytes'],'DISK_MINIMUM_RESERVE_EXHAUSTED')

    def admit_disk(self,remaining):
        free=shutil.disk_usage(self.out).free
        plan=remaining*self.c['planning_bytes_per_policy']+self.c['planning_evaluation_bytes']
        required=self.c['disk_min_free_bytes']+math.ceil(plan*(1+self.c['disk_reserve_fraction']))
        need(free>=required,'INSUFFICIENT_COMPLETE_COHORT_DISK_ADMISSION:'+str({'free':free,'required':required,'remaining':remaining}))
        self.s['latest_disk_admission']={'free_bytes':free,'required_bytes':required,'remaining_training':remaining}

    def spawn(self,key,args,*,kind,limit,case_id=None):
        self.guard(force_sources=True)
        need(key not in self.active,'DUPLICATE_CHILD_KEY')
        log=self.out/'logs'/(key+'.log');log.parent.mkdir(exist_ok=True)
        f=log.open('ab',buffering=0)
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
            MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
            PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
        expected_parent=os.getpid()
        bootstrap=[sys.executable,'-B',str(self.project/'runtime/cohort_fair.py'),
                   '--child-bootstrap',str(expected_parent),'--',*args]
        p=subprocess.Popen(bootstrap,cwd=self.project/'runtime',env=env,stdin=subprocess.DEVNULL,
            stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        f.close()
        born=None
        for _ in range(100):
            born=identity(p.pid)
            if born:break
            if p.poll()is not None:break
            time.sleep(.01)
        need(born is not None and born['pgid']==p.pid and born['session']==p.pid,'CHILD_IDENTITY_NOT_RECORDED')
        record={**born,'kind':kind,'key':key,'case_id':case_id,'argv':args,
                'started_unix':time.time(),'limit_seconds':limit,'known_members':{str(p.pid):born['birth_ticks']}}
        self.active[key]={'process':p,'record':record};self.save('CHILD_STARTED')
        return key

    def observe(self):
        self.guard()
        finished=[]
        for key,item in list(self.active.items()):
            p=item['process'];r=item['record']
            actual=identity(r['pid'])
            if actual and actual['state']not in PID_STATE_DEAD:
                need(actual['birth_ticks']==r['birth_ticks'] and actual['pgid']==r['pgid']
                     and actual['session']==r['session'],'CHILD_IDENTITY_CHANGED')
            members=group_members(r['pgid'],r['proc_pgid'])
            for member in members:
                # While the recorded session leader remains alive, its
                # process group is exclusively the session we created.
                if actual and actual['birth_ticks']==r['birth_ticks']:
                    r['known_members'][str(member['pid'])]=member['birth_ticks']
                elif r['known_members'].get(str(member['pid']))!=member['birth_ticks']:
                    # A short-lived multiprocessing leader may spawn its
                    # tracker between observations. Do not adopt or signal an
                    # unrecorded PID after leader exit. Permit only a bounded
                    # passive EOF drain, and require an empty group for success.
                    r.setdefault('unattributed_after_exit',{})[str(member['pid'])]=member
            need(time.time()-r['started_unix']<r['limit_seconds'],'OWNED_CHILD_OPERATIONAL_TIMEOUT:'+key)
            code=p.poll()
            if code is not None:
                if members:
                    # multiprocessing's resource tracker may need a brief
                    # EOF/shutdown interval after the session leader exits.
                    r.setdefault('leader_exit_seen_unix',time.time())
                    need(time.time()-r['leader_exit_seen_unix']<3,
                         'CHILD_EXIT_LEFT_LIVE_PROCESS_GROUP:'+key)
                    continue
                finished.append((key,code,r))
                del self.active[key]
        return finished

    def cleanup_owned(self):
        reports=[]
        for key,item in list(self.active.items()):
            p=item['process'];saved=item['record'];actual=identity(saved['pid'])
            members=group_members(saved['pgid'],saved['proc_pgid'])
            targets=[]
            for member in members:
                if saved['known_members'].get(str(member['pid']))==member['birth_ticks']:
                    targets.append(member)
                elif actual and actual['birth_ticks']==saved['birth_ticks'] and actual['session']==saved['session']:
                    # Observed under the still-authentic session leader.
                    targets.append(member)
                else:reports.append({'key':key,'unknown_member_not_signalled':member})
            for sig in (signal.SIGTERM,signal.SIGKILL):
                for old in targets:
                    cur=identity(old['pid'])
                    if cur and cur['birth_ticks']==old['birth_ticks'] and cur['state']not in PID_STATE_DEAD:
                        try:os.kill(cur['pid'],sig)
                        except ProcessLookupError:pass
                try:p.wait(timeout=3 if sig==signal.SIGTERM else 2)
                except subprocess.TimeoutExpired:pass
                if not group_members(saved['pgid'],saved['proc_pgid']):break
            remaining=group_members(saved['pgid'],saved['proc_pgid'])
            reports.append({'key':key,'remaining_members':remaining,'returncode':p.poll()})
            del self.active[key]
        return reports

    def wait_stage(self,key):
        while key in self.active:
            finished=self.observe()
            for _,code,r in finished:
                need(code==0,'STAGE_EXIT_FAILED:'+r['key']+':'+str(code))
                self.s.setdefault('closed_processes',[]).append(r|{'returncode':code})
                self.save('STAGE_PROCESS_CLOSED')
            if key in self.active:time.sleep(.2)

    def verify_training(self,case_id):
        sys.path.insert(0,str(self.project/'runtime'))
        from evaluation_fair.actor import verify_training_policy
        cfg,done,_=verify_training_policy(self.out/'training'/case_id)
        want=self.configs[case_id]['configuration']
        # configuration() expands declared runtime defaults in config.json.
        from training.pci_risk_train.fair_trainer import configuration as training_configuration
        need(cfg==training_configuration(want),'CLOSED_TRAINING_CONFIGURATION_DIFFERS')
        need(done['status']=='COMPLETE' and done['joint_steps']==want['total_steps'],'INCOMPLETE_TRAINING_CASE')
        need(done['run_role']==want['run_role'],'TRAINING_ROLE_MISMATCH')
        self.s['completed_training'][case_id]={'train_complete_sha256':sha(self.out/'training'/case_id/'TRAIN_COMPLETE.json'),
            'closed_manifest_sha256':done['closed_manifest_sha256'],'signature':done['signature'],
            'joint_steps':done['joint_steps']}
        return done

    def training(self):
        order=[r['case_id'] for r in self.manifest['cases']]
        pending=[]
        for case_id in order:
            directory=self.out/'training'/case_id
            need(directory.resolve()==directory and not directory.is_symlink(),'TRAINING_OUTPUT_SYMLINK:'+case_id)
            p=directory/'TRAIN_COMPLETE.json'
            if p.exists():self.verify_training(case_id)
            else:pending.append(case_id)
        self.admit_disk(len(pending));self.save('TRAINING_COHORT')
        cursor=0
        while cursor<len(pending) or self.active:
            self.guard()
            self.admit_disk(len(pending)-cursor+len(self.active))
            while cursor<len(pending) and len(self.active)<self.c['max_workers']:
                case_id=pending[cursor];cursor+=1
                directory=self.out/'training'/case_id
                need(directory.resolve()==directory and not directory.is_symlink(),'TRAINING_OUTPUT_SYMLINK:'+case_id)
                args=[sys.executable,'-B','-m','training.pci_risk_train.fair_trainer',
                      '--config',self.configs[case_id]['path'],'--out',str(self.out/'training'/case_id)]
                self.spawn('train_'+case_id,args,kind='training',limit=self.c['per_case_seconds'],case_id=case_id)
            for key,code,r in self.observe():
                need(code==0,'TRAINING_CHILD_FAILED:'+str(r['case_id'])+':'+str(code))
                self.verify_training(r['case_id']);self.guard(force_sources=True)
                self.s.setdefault('closed_processes',[]).append(r|{'returncode':code})
                self.save('TRAINING_CASE_CLOSED')
            if cursor<len(pending) or self.active:time.sleep(.2)
        need(set(self.s['completed_training'])==set(order),'ALL_PRESCHEDULED_POLICIES_REQUIRED')
        need(not self.active,'TRAINING_WORKERS_NOT_CLOSED')
        self.admit_disk(0)
        self.save('ALL_FINAL_POLICIES_CLOSED')

    def evaluation(self):
        descriptors=[{'directory':r['case_id'],'candidate_id':r['candidate_id'],
            'family':r['family'],'training_seed':r['training_seed']} for r in self.manifest['cases']]
        closed(self.out/'POLICY_DESCRIPTORS.json',descriptors,runtime_root=self.project/'runtime')
        manifest=self.out/'EVALUATION_MANIFEST.json'
        args=[sys.executable,'-B','-m','evaluation_fair.engine','prepare',
              '--training-root',str(self.out/'training'),'--policy-descriptors',str(self.out/'POLICY_DESCRIPTORS.json'),
              '--stage',self.c['stage'],'--worlds-per-profile',str(self.c['worlds_per_profile']),'--output',str(manifest)]
        if self.c['role']=='scientific':
            args+=['--protocol-sha256',self.manifest['protocol_canonical_sha256'],
                   '--training-case-manifest',str(self.manifest_path)]
        self.spawn('evaluation_prepare',args,kind='evaluation_prepare',limit=self.c['stage_seconds']);self.wait_stage('evaluation_prepare')
        self.guard(force_sources=True)
        evalout=self.out/'evaluation'
        need(evalout.resolve()==evalout and not evalout.is_symlink(),'EVALUATION_OUTPUT_SYMLINK')
        need((self.project/'runtime/evaluation_fair/parallel.py').is_file(),'QUALIFIED_PARALLEL_EVALUATOR_REQUIRED')
        args=[sys.executable,'-B','-m','evaluation_fair.parallel','evaluate',
              '--training-root',str(self.out/'training'),'--manifest',str(manifest),
              '--out',str(evalout),'--max-workers',str(self.c['evaluation_workers'])]
        self.spawn('evaluation_collect',args,kind='evaluation',limit=self.c['stage_seconds']);self.wait_stage('evaluation_collect')
        args=[sys.executable,'-B','-m','evaluation_fair.parallel','audit',
              '--training-root',str(self.out/'training'),'--out',str(evalout),
              '--max-workers',str(self.c['evaluation_workers'])]
        self.spawn('evaluation_audit',args,kind='audit',limit=self.c['stage_seconds']);self.wait_stage('evaluation_audit')
        complete,complete_hash=read_bound(evalout/'EVALUATION_COMPLETE.json')
        audit,audit_hash=read_bound(evalout/'RETAINED_RECORD_AUDIT.json')
        em=read(manifest)
        need(complete['status']=='COMPLETE' and audit['status']=='PASS','COMPLETE_EVALUATION_AND_AUDIT_REQUIRED')
        need(complete['all_policy_count']==len(self.configs)==audit['policy_count'],'ALL_POLICY_COUNT_REQUIRED')
        need(complete['all_episode_count']==len(em['cases'])==audit['episode_count'],'ALL_EPISODE_COUNT_REQUIRED')
        need(complete['joint_steps']==audit['joint_steps'] and complete['agent_decisions']==audit['agent_decisions'],'EVALUATION_AUDIT_COUNTS_DIFFER')
        self.save('COMPLETE_EVALUATION_AUDITED',evaluation_complete_sha256=complete_hash,
            retained_record_audit_sha256=audit_hash)
        if self.c['stage']=='development':
            args=[sys.executable,'-B',str(self.project/'protocol/frontier.py'),
                  '--development-records',str(evalout/'development_selection_rows.json'),
                  '--output',str(self.out/'DEVELOPMENT_SELECTION.json'),'--study-domain',self.c['study_domain']]
            if not (self.out/'DEVELOPMENT_SELECTION.json').exists():
                self.spawn('development_selection',args,kind='selection',limit=min(300,self.c['stage_seconds']));self.wait_stage('development_selection')
            selection=read(self.out/'DEVELOPMENT_SELECTION.json')
            need(selection['status']=='DEVELOPMENT_ONLY_SELECTION' and selection['primary_test_data_opened']is False,'DEVELOPMENT_ONLY_SELECTION_REQUIRED')
            # A stale/tampered selection receipt cannot choose configurations
            # for a later confirmation phase merely by copying a data digest.
            module_path=self.project/'protocol/frontier.py'
            sys.path.insert(0,str(module_path.parent))
            spec=importlib.util.spec_from_file_location('_cohort_frontier_'+sha(module_path),module_path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            records_path=evalout/'development_selection_rows.json'
            records,records_hash=read_bound(records_path)
            expected=module.select(records,source_sha256=records_hash,
                                   study_domain=self.c['study_domain'])
            need(canonical(selection)==canonical(expected),'DEVELOPMENT_SELECTION_RECOMPUTATION_DIFFERS')
        return complete,audit

    def run(self):
        try:
            self.training();complete,audit=self.evaluation();self.guard(force_sources=True)
            need(not self.active,'OWNED_WORKER_CLOSURE_REQUIRED')
            result={'schema':'PCI-FAIR-COHORT-COMPLETE-v1',
                'status':'COMPLETE_ENGINEERING_COHORT' if self.c['role']=='engineering' else 'COMPLETE_AUDITED_COHORT',
                'scientific_performance_claim':False if self.c['role']=='engineering' else None,
                'run_id':self.c['run_id'],'stage':self.c['stage'],'study_domain':self.c['study_domain'],
                'source_map_sha256':digest(self.sources),'case_manifest_sha256':self.binding['case_manifest_sha256'],
                'execution_lock_sha256':sha(self.out/'EXECUTION_LOCK.json'),
                'trained_policies':len(self.configs),'training_joint_steps':sum(r['joint_steps'] for r in self.s['completed_training'].values()),
                'evaluated_episodes':complete['all_episode_count'],'evaluation_joint_steps':audit['joint_steps'],
                'evaluation_agent_decisions':audit['agent_decisions'],'all_owned_worker_processes_closed':True,
                'analysis_sha256':sha(self.out/'evaluation/ANALYSIS.json'),
                'retained_record_audit_sha256':self.s['retained_record_audit_sha256'],
                'completed_unix':time.time(),'absolute_deadline_unix':self.c['deadline_unix']}
            final_path=self.out/'STUDY_COMPLETE.json'
            if final_path.exists():
                previous=read(final_path)
                need({k:v for k,v in previous.items() if k!='completed_unix'}==
                     {k:v for k,v in result.items() if k!='completed_unix'},'PRIOR_COMPLETION_CHANGED')
                result=previous
            # Revalidate canonical immutable bytes even on a closed resume;
            # parsing alone would accept a rewritten receipt with whitespace.
            closed(final_path,result,runtime_root=self.project/'runtime')
            self.save(result['status'])
            return result
        except BaseException as error:
            cleanup=self.cleanup_owned()
            self.save('INCOMPLETE_REQUIRES_REVIEW',error=repr(error),cleanup_reports=cleanup,
                      no_partial_cohort_success_claim=True)
            raise

def main():
    if len(sys.argv)>=5 and sys.argv[1]=='--child-bootstrap':
        parent=int(sys.argv[2]);need(sys.argv[3]=='--','BOOTSTRAP_ARGUMENT_BOUNDARY')
        args=sys.argv[4:];need(args,'BOOTSTRAP_EXECUTABLE_REQUIRED')
        _child_parent_death_signal(parent)
        os.execv(args[0],args)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    a=p.parse_args()
    result=Controller(read(a.config),a.config).run()
    print(json.dumps(result,sort_keys=True))

if __name__=='__main__':main()

