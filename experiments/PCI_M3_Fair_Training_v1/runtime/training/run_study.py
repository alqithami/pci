"""Source-bound M2 planning, live GPU readiness gates and owned-process control.

plan/check/self-test import no Torch and execute no scientific trajectory. GPU
paths are deliberately untested until run on the freshly verified owned L4.
No command starts/stops a cloud instance or changes its networking. A supplied
PASS receipt cannot authorize a launch: run recomputes the executable gates.
"""
from __future__ import annotations
import argparse, ast, copy, csv, datetime as dt, gzip, hashlib, io, json, math
import os, platform, re, shutil, signal, socket, subprocess, sys, tarfile, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('task','fixed','adaptive','semantic','consensus','directrisk')
PROFILES = ('nominal','quota_tight','capacity_tight','rule_shift')
OWNED_ID = '02b7_e5856ec3-7b46-4c9d-957f-67689272d52f'
OWNED_NAME = 'chmarl-render-l4'
OWNED_PROFILE = 'gx3-16x80x1l4'
MAX_WORKERS = 4
CEILING = 8*3600
FORECAST_MAX = 6*3600
FIXTURE_STEPS = 32768
TOLS = dict(logits_atol=2e-5, probabilities_atol=2e-6, values_atol=2e-5,
            risk_atol=2e-6, auxiliary_atol=2e-6, gradient_atol=2e-5, gradient_rtol=1e-3)
HASH_RE = re.compile('[0-9a-f]{64}')

class GateError(ValueError): pass

def require(condition, message):
    if not condition: raise GateError(message)

def global_budget(not_after_unix=None,*,started_unix):
    """An optional wall-clock cutoff may only shorten the existing ceiling."""
    require(type(started_unix) in (int,float) and math.isfinite(started_unix),'FINITE_CONTROLLER_START_REQUIRED')
    if not_after_unix is None:return float(CEILING)
    require(type(not_after_unix) in (int,float) and math.isfinite(not_after_unix),'FINITE_NOT_AFTER_UNIX_REQUIRED')
    remaining=not_after_unix-started_unix
    require(math.isfinite(remaining) and remaining>0,'NOT_AFTER_UNIX_EXPIRED')
    return min(float(CEILING),remaining)

def remaining_global_budget(deadline,*,now_monotonic=None):
    now=time.monotonic() if now_monotonic is None else now_monotonic
    require(math.isfinite(deadline) and math.isfinite(now) and deadline>now,'GLOBAL_WALL_CEILING')
    return deadline-now

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)

def encoded(value):
    return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()

def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()

def sha(path):
    p=Path(path);before=p.stat()
    require(p.is_file() and not p.is_symlink(),'REGULAR_FILE_REQUIRED:'+str(p))
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    after=p.stat()
    require((before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
            (after.st_size,after.st_mtime_ns,after.st_ctime_ns),'FILE_CHANGED_DURING_READ:'+str(p))
    return h.hexdigest()

def read_json(path):
    def pairs(items):
        result={}
        for k,v in items:
            require(k not in result,'DUPLICATE_JSON_KEY:'+k);result[k]=v
        return result
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,
        parse_constant=lambda x:(_ for _ in ()).throw(GateError('NONFINITE_JSON:'+x)))

def relative(root,name):
    require(type(name) is str and name and not Path(name).is_absolute() and
            '..' not in Path(name).parts and '\\' not in name,'SAFE_RELATIVE_PATH_REQUIRED')
    p=(Path(root)/name).resolve()
    require(p.is_relative_to(Path(root).resolve()),'PATH_ESCAPE')
    return p

def sync_dir(path):
    fd=os.open(Path(path),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def write_json(path,value,*,replace=False):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    require(replace or not p.exists(),'REFUSE_EXISTING_FILE:'+str(p))
    temporary=p.with_name(p.name+'.writing')
    with temporary.open('xb') as f:f.write(encoded(value));f.flush();os.fsync(f.fileno())
    os.replace(temporary,p);sync_dir(p.parent)

def source_closure(root,protocol):
    root=Path(root).resolve();protocol=Path(protocol).resolve()
    require(protocol.is_relative_to(root),'PROTOCOL_MUST_BE_PROJECT_LOCAL')
    names=set()
    for folder in ('training','evaluation','training_controls','evaluation_controls'):
        for p in (root/folder).rglob('*.py'):
            if '__pycache__' not in p.parts:
                ast.parse(p.read_text(),filename=str(p));names.add(p.relative_to(root).as_posix())
    names.update(('training/GPU_GATE_SPEC.md','training_controls/fixture_actor.npz',
        'training_controls/fixture_inputs.npz','training_controls/FIXTURE_INPUT_ANCHORS.json',
        'kernel_controls/KERNEL_FROZEN_INDEX.json','kernel_controls/KERNEL_VALIDATION.json',
        'kernel_controls/independent_review.json',protocol.relative_to(root).as_posix()))
    for p in (root/'design').glob('M2_SCIENTIFIC_DESIGN*.json'):
        names.add(p.relative_to(root).as_posix())
    required={'training/run_study.py','training/vectorized_admission.py',
        'training/pci_risk_train/trainer.py','training/pci_risk_train/model.py',
        'training/pci_risk_train/environment.py','training/pci_risk_train/objectives.py',
        'training/pci_risk_train/io.py','training_controls/check_training.py',
        'evaluation/engine.py','evaluation/retained_audit.py','evaluation/analysis.py'}
    require(required<=names,'INCOMPLETE_RUNTIME_CLOSURE:'+repr(sorted(required-names)))
    return {n:sha(relative(root,n)) for n in sorted(names)}

def validate_protocol(protocol,*,frozen=False):
    require(protocol.get('schema') in ('PCI-M2-MATCHED-RISK-TRAINING-PROTOCOL-v1',
            'PCI-M2-MATCHED-RISK-TRAINING-PROTOCOL-DRAFT-v1'),'PROTOCOL_SCHEMA')
    if frozen:
        require(protocol['schema']=='PCI-M2-MATCHED-RISK-TRAINING-PROTOCOL-v1' and
                protocol.get('scientific_design_frozen') is True,'SCIENTIFIC_DESIGN_NOT_FROZEN')
    train,evaluation=protocol['training_cases'],protocol['evaluation_cases']
    require(type(train) is list and len(train)==60,'EXACT_SIXTY_TRAINING_CASES')
    require(type(evaluation) is list and len(evaluation)==1200,'EXACT_1200_EVALUATION_CASES')
    seeds=list(range(41001,41011))
    require({(r['method'],r['training_seed']) for r in train}==
            {(m,s) for m in METHODS for s in seeds},'FULL_FIXED_TRAINING_CROSS')
    ids=[r['case_id'] for r in train]
    require(len(set(ids))==60 and all(re.fullmatch('[a-zA-Z0-9_-]+',n) for n in ids),'UNIQUE_SAFE_TRAINING_IDS')
    case_map={r['case_id']:r for r in train}
    initialization={}
    for r in train:
        c=r['config'];s=r['training_seed'];cluster=s-41001
        require(r['cluster_index']==cluster and c['method']==r['method'] and c['seed']==c['init_seed']==s,'CONFIG_CASE_IDENTITY')
        require(c['run_id']==r['case_id'] and c['run_role']=='scientific' and c['device']=='cuda','SCIENTIFIC_CUDA_CASE_CONFIG')
        require(c['total_steps']==262144 and c['n_envs']==8 and c['rollout_steps']==128 and
                c['hidden']==64 and c['update_epochs']==4 and c['minibatch_groups']==128 and
                c['torch_threads']==1 and c['enforce'] is True,'EXACT_COMMON_TRAINING_BUDGET')
        require(c['env_profile_ids']==[p for p in PROFILES for _ in range(2)] and
                c['profiles']==protocol['profiles'],'BALANCED_FOUR_PROFILE_TRAINING')
        require(c['env_seeds']==[840000000+100*cluster+i for i in range(8)] and
                c['policy_seed']==840100000+cluster and c['minibatch_seed']==840200000+cluster,'FIXED_FRESH_TRAINING_STREAMS')
        require(c['checkpoint_every_updates']==8 and c['deterministic_algorithms'] is True,'FIXED_RETENTION_AND_DETERMINISM')
        shared=copy.deepcopy(c)
        for k in ('run_id','method'):shared.pop(k)
        require(initialization.setdefault(s,shared)==shared,'MATCHED_WITHIN_SEED_CONFIGS')
    keys=set();streams={}
    for r in evaluation:
        k=(r['method'],r['training_seed'],r['profile_id'],r['world_index'])
        require(k not in keys,'DUPLICATE_EVALUATION_CELL');keys.add(k)
        require(r['training_case_id'] in case_map,'UNKNOWN_FINAL_POLICY')
        parent=case_map[r['training_case_id']]
        require(parent['method']==r['method'] and parent['training_seed']==r['training_seed'],'EVALUATION_PARENT_IDENTITY')
        require(r['horizon']==256 and r['cluster_index']==r['training_seed']-41001,'FIXED_EVALUATION_HORIZON')
        stream=(r['world_seed'],r['policy_sampling_seed']);sk=(r['training_seed'],r['world_index'])
        require(streams.setdefault(sk,stream)==stream,'MATCHED_EXOGENOUS_EVALUATION_STREAMS')
        require(stream==(850000000+100*r['cluster_index']+r['world_index'],
                        850100000+100*r['cluster_index']+r['world_index']),'FIXED_FRESH_EVALUATION_STREAMS')
    require(keys=={(m,s,p,w) for m in METHODS for s in seeds for p in PROFILES for w in range(5)},'FULL_FIXED_EVALUATION_CROSS')
    require(len({r['case_id'] for r in evaluation})==1200,'UNIQUE_EVALUATION_IDS')
    return train,evaluation

def make_plan(root,protocol_path,expected_sha=None):
    root=Path(root).resolve();p=Path(protocol_path).resolve();h=sha(p)
    require(root/'training/run_study.py'==Path(__file__).resolve(),'EXECUTING_CONTROLLER_SOURCE_PATH_MISMATCH')
    require(expected_sha is None or expected_sha==h,'PROTOCOL_DIGEST_MISMATCH')
    protocol=read_json(p);train,evaluation=validate_protocol(protocol)
    closure=source_closure(root,p)
    ledger=[]
    for r in train:
        ledger.append({'kind':'training','case_id':r['case_id'],'method':r['method'],
            'seed':r['training_seed'],'config_sha256':digest(r['config']),
            'config_body_sha256':hashlib.sha256(encoded(r['config'])).hexdigest()})
    ledger += [{'kind':'evaluation','case_id':r['case_id'],'training_case_id':r['training_case_id'],
                'case_sha256':digest(r)} for r in evaluation]
    return dict(schema='PCI-M2-CONTROLLER-PLAN-v1',status='NOT_GPU_LAUNCH_READY',
        protocol_path=p.relative_to(root).as_posix(),protocol_sha256=h,
        scientific_design_frozen=protocol.get('scientific_design_frozen') is True,
        source_sha256=closure,source_map_sha256=digest(closure),case_ledger=ledger,
        case_ledger_sha256=digest(ledger),training_cases=60,evaluation_cases=1200,
        training_joint_steps=15728640,evaluation_joint_steps=307200,
        max_workers=MAX_WORKERS,global_deadline_seconds=CEILING,max_forecast_seconds=FORECAST_MAX,
        pending_live_gates=['fresh IBM authentication and operator ownership/idle-host observation',
            'source-bound CPU executable controls','live CPU/CUDA numerical comparison and actual CUDA profiler events',
            'two waves of four concurrent complete-path engineering jobs plus retained evaluation/audit',
            'compressed full-cohort disk forecast','all sixty closed final models before evaluation'],
        scientific_worlds_or_training_started=False,
        scope='Read-only source/configuration/seed validation. This plan never authorizes GPU launch.')

def worker_environment():
    env=dict(os.environ)
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        env[k]='1'
    env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    env['PYTHONDONTWRITEBYTECODE']='1'
    return env

def _proc_entry(path,kind='group'):
    namespace=os.readlink(path/'ns/pid')
    if namespace!=os.readlink('/proc/self/ns/pid'):raise ProcessLookupError('Different PID namespace')
    status={line.split(':',1)[0]:line.split(':',1)[1].split()
            for line in (path/'status').read_text().splitlines() if ':' in line}
    suffix=(path/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=int(status.get('NSpid',status['Pid'])[-1]),kind=kind,
        birth_ticks=int(suffix[19]),process_group=int(status.get('NSpgid',[suffix[2]])[-1]),
        session=int(status.get('NSsid',[suffix[3]])[-1]),state=suffix[0],
        uid=int(status['Uid'][0]),pid_namespace=namespace)

def process_record(pid,kind='group'):
    # /proc can be mounted from an ancestor PID namespace. Match namespace and
    # NSpid rather than assuming its directory number equals our visible PID.
    first=Path('/proc')/str(pid)
    try:
        record=_proc_entry(first,kind)
        if record['pid']==pid:return record
    except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or path==first:continue
        try:record=_proc_entry(path,kind)
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        if record['pid']==pid:return record
    raise ProcessLookupError('No same-namespace process for PID '+str(pid))

def owned_group_members(record):
    try:
        leader=process_record(record['pid'])
        if leader['birth_ticks']!=record['birth_ticks']:return []
    except (FileNotFoundError,ProcessLookupError):pass
    result=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:r=_proc_entry(p)
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        if r['process_group']!=record['pid'] or r['state']=='Z':continue
        require(r['session']==record['pid'] and r['uid']==record['uid'] and
                r['birth_ticks']>=record['birth_ticks'],'OWNED_GROUP_IDENTITY_UNSAFE_TO_SIGNAL')
        result.append(r)
    return result

def register_owned(process):
    record=process_record(process.pid)
    require(record['process_group']==record['session']==process.pid,'NEW_OWNED_SESSION_REQUIRED')
    process._pci_owned_record=record
    path=os.environ.get('PCI_M2_OWNED_PID_REGISTRY')
    if path:
        with Path(path).open('a') as f:f.write(canonical(record)+'\n');f.flush();os.fsync(f.fileno())
    return record

def terminate_owned(processes):
    """Close our created sessions, including children after their leader exits."""
    records=[];unknown=[]
    for p in processes:
        record=getattr(p,'_pci_owned_record',None)
        if record is None:
            try:record=process_record(p.pid)
            except (FileNotFoundError,ProcessLookupError):
                # Popen still owns this unreaped child. Its signal methods poll
                # before signaling, avoiding reuse of an already reaped PID.
                if p.poll() is None:
                    p.terminate()
                    try:p.wait(timeout=3)
                    except subprocess.TimeoutExpired:p.kill();p.wait(timeout=3)
                unknown.append(p.pid)
                continue
        records.append(record)
    actions=[]
    for sig,grace in ((signal.SIGINT,20.),(signal.SIGTERM,5.),(signal.SIGKILL,3.)):
        live=[r for r in records if owned_group_members(r)]
        if not live:break
        for r in live:
            try:os.killpg(r['pid'],sig);actions.append({'process_group':r['pid'],'signal':sig.name})
            except ProcessLookupError:pass
        end=time.monotonic()+grace
        while time.monotonic()<end and any(owned_group_members(r) for r in live):time.sleep(.1)
    for p in processes:
        try:p.wait(timeout=.1)
        except subprocess.TimeoutExpired:pass
    require(not any(owned_group_members(r) for r in records),'OWNED_GROUP_CLEANUP_FAILED')
    require(not unknown,'OWNED_SESSION_IDENTITY_UNKNOWN_AFTER_CHILD_CLEANUP:'+repr(unknown))
    return actions

def watchdog(parent_pid,parent_birth,registry,deadline_monotonic,receipt):
    """Separate OS process: the deadline covers inline CUDA and filesystem I/O."""
    require(parent_pid==os.getppid(),'WATCHDOG_CAN_ONLY_SUPERVISE_ITS_OWN_PARENT')
    parent=process_record(parent_pid,kind='pid')
    require(parent['birth_ticks']==parent_birth,'WATCHDOG_PARENT_BIRTH_IDENTITY')
    def parent_alive():
        try:r=process_record(parent_pid)
        except (FileNotFoundError,ProcessLookupError):return False
        return r['birth_ticks']==parent_birth and r['state']!='Z'
    while time.monotonic()<deadline_monotonic and parent_alive():time.sleep(.2)
    exceeded=time.monotonic()>=deadline_monotonic
    records=[]
    for line in Path(registry).read_text().splitlines():
        if line:records.append(json.loads(line))
    actions=[]
    for sig,grace in ((signal.SIGINT,20.),(signal.SIGTERM,5.),(signal.SIGKILL,3.)):
        groups=[r for r in records if owned_group_members(r)]
        for r in groups:
            try:os.killpg(r['pid'],sig);actions.append({'process_group':r['pid'],'signal':sig.name})
            except ProcessLookupError:pass
        if exceeded and parent_alive():
            try:os.kill(parent_pid,sig);actions.append({'parent_pid':parent_pid,'signal':sig.name})
            except ProcessLookupError:pass
        end=time.monotonic()+grace
        while time.monotonic()<end:
            if not any(owned_group_members(r) for r in groups) and (not exceeded or not parent_alive()):break
            time.sleep(.1)
        if not any(owned_group_members(r) for r in records) and (not exceeded or not parent_alive()):break
    write_json(receipt,dict(schema='PCI-M2-OWNED-OS-WATCHDOG-v1',
        status='GLOBAL_DEADLINE_FAILURE' if exceeded else 'PARENT_EXIT_CLEANUP',
        deadline_exceeded=exceeded,actions=actions,
        surviving_owned_groups=[r['pid'] for r in records if owned_group_members(r)],
        parent_survives=parent_alive(),scope='Only recorded created sessions and the exact birth-identified parent PID.'))

def start_watchdog(output,deadline_seconds,*,absolute_deadline_monotonic=None):
    if absolute_deadline_monotonic is not None:
        remaining_global_budget(absolute_deadline_monotonic)
    output=Path(output);registry=output/'OWNED_PROCESS_GROUPS.jsonl'
    with registry.open('x') as f:f.flush();os.fsync(f.fileno())
    os.environ['PCI_M2_OWNED_PID_REGISTRY']=str(registry)
    parent=process_record(os.getpid(),kind='pid')
    log=(output/'WATCHDOG.log').open('xb')
    watchdog_deadline=(time.monotonic()+deadline_seconds if absolute_deadline_monotonic is None
                       else absolute_deadline_monotonic)
    remaining_global_budget(watchdog_deadline)
    p=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'_watchdog',
        '--parent-pid',str(os.getpid()),'--parent-birth',str(parent['birth_ticks']),
        '--registry',str(registry),'--deadline-monotonic',str(watchdog_deadline),
        '--output',str(output/'WATCHDOG_RECEIPT.json')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    return p,log

def stop_watchdog(process,stream):
    if process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=3)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    stream.flush();os.fsync(stream.fileno());stream.close()


def run_pool(jobs,root,logs,deadline,*,workers=4,on_complete=None):
    require(1<=workers<=MAX_WORKERS,'WORKER_CEILING')
    logs=Path(logs);logs.mkdir(parents=True,exist_ok=False)
    rows=[dict(case_id=j['case_id'],status='UNRUN',argv=j['argv']) for j in jobs]
    pending=list(range(len(jobs)));running={};all_procs=[];opened=[];peak=0;failed=None
    live_samples=[];observed_live_intervals=[]
    try:
        while pending or running:
            require(time.monotonic()<deadline,'GLOBAL_WALL_CEILING')
            while pending and len(running)<workers:
                i=pending.pop(0);row=rows[i]
                stdout=(logs/(row['case_id']+'.stdout.log')).open('xb')
                stderr=(logs/(row['case_id']+'.stderr.log')).open('xb');opened += [stdout,stderr]
                p=subprocess.Popen(jobs[i]['argv'],cwd=root,env=worker_environment(),
                    stdout=stdout,stderr=stderr,start_new_session=True)
                all_procs.append(p)
                row.update(status='RUNNING',pid=p.pid,start_monotonic=time.monotonic())
                register_owned(p)
                running[i]=(p,stdout,stderr)
                peak=max(peak,len(running))
            if len(running)==workers:
                sample_start=time.monotonic()
                pids=tuple(sorted(p.pid for p,_,_ in running.values()))
                if all(p.poll() is None for p,_,_ in running.values()):
                    sample=dict(pids=pids,start=sample_start,end=time.monotonic())
                    if live_samples and live_samples[-1]['pids']==pids:
                        lower=sample['start']-live_samples[-1]['end']
                        if lower>0:observed_live_intervals.append(dict(pids=pids,seconds_lower_bound=lower))
                    live_samples.append(sample)
            finished=[]
            for i,(p,stdout,stderr) in list(running.items()):
                code=p.poll()
                if code is None:continue
                stdout.flush();stderr.flush();os.fsync(stdout.fileno());os.fsync(stderr.fileno())
                stdout.close();stderr.close();finished.append(i)
                rows[i].update(exit_code=code,end_monotonic=time.monotonic())
                if code!=0:
                    rows[i]['status']='FAILED';failed=f'CASE_PROCESS_FAILED:{rows[i]["case_id"]}:{code}';break
                if owned_group_members(p._pci_owned_record):
                    rows[i]['status']='FAILED_OWNED_DESCENDANT_LEAK'
                    failed='OWNED_DESCENDANTS_SURVIVED_WORKER:'+rows[i]['case_id'];break
                if on_complete is not None:
                    try:on_complete(jobs[i],rows[i])
                    except BaseException:
                        rows[i]['status']='FAILED_RETENTION_OR_VALIDATION'
                        raise
                rows[i]['status']='COMPLETE_CLOSED'
            for i in finished:running.pop(i,None)
            if failed:raise GateError(failed)
            if running:time.sleep(.1)
        return dict(status='PASS',rows=rows,peak_parallel_workers=peak,owned_cleanup_actions=[],
            actual_live_concurrency_intervals=observed_live_intervals)
    except BaseException as exc:
        cleanup_error=None
        try:cleanup=terminate_owned(all_procs)
        except Exception as cleanup_exception:
            cleanup=[];cleanup_error=repr(cleanup_exception)
        for i,row in enumerate(rows):
            if row['status']=='RUNNING':row.update(status='STOP_REQUESTED_SESSION_UNVERIFIED' if cleanup_error else 'STOPPED_AFTER_COHORT_FAILURE',end_monotonic=time.monotonic())
            elif row['status']=='UNRUN':row['status']='UNRUN_AFTER_COHORT_FAILURE'
        result=dict(status='FAIL',error=repr(exc),rows=rows,peak_parallel_workers=peak,
                    owned_cleanup_actions=cleanup,all_owned_processes_closed=cleanup_error is None,
                    cleanup_error=cleanup_error)
        write_json(logs/'POOL_FAILED.json',result)
        raise GateError('POOL_FAILURE_RETAINED:'+str(logs/'POOL_FAILED.json')) from exc
    finally:
        for stream in opened:
            if not stream.closed:stream.close()

def run_one(argv,root,logs,deadline):
    return run_pool([{'case_id':'command','argv':argv}],root,logs,deadline,workers=1)

def check_ownership(record,*,now=None):
    require(record.get('schema')=='PCI-M2-OPERATOR-OWNED-GPU-OBSERVATION-v1','OWNERSHIP_OBSERVATION_SCHEMA')
    i=record['instance'];require(i['id']==OWNED_ID and i['name']==OWNED_NAME and
        i['profile_name']==OWNED_PROFILE and i['status']=='running','EXACT_OWNED_L4_IDENTITY_REQUIRED')
    now=time.time() if now is None else now
    for k in ('observed_at_utc','ibm_auth_verified_at_utc'):
        x=dt.datetime.fromisoformat(record[k].replace('Z','+00:00'))
        require(x.tzinfo is not None and -30<=now-x.timestamp()<=300,'STALE_OR_FUTURE_OPERATOR_OBSERVATION:'+k)
    require(record.get('other_gpu_jobs')==[],'OWNED_GPU_NOT_CONFIRMED_IDLE')
    require(record.get('scope')=='operator-observed-not-remote-attestation','OPERATOR_TRUST_BOUNDARY_REQUIRED')
    return record

def gpu_identity():
    def command(args):
        r=subprocess.run(args,capture_output=True,text=True,timeout=15)
        require(r.returncode==0,'NVIDIA_SMI_FAILED:'+r.stderr[-300:]);return r.stdout.strip()
    text=command(['nvidia-smi','--query-gpu=name,uuid,memory.total,memory.used','--format=csv,noheader,nounits'])
    rows=list(csv.reader(io.StringIO(text)))
    require(len(rows)==1 and len(rows[0])==4,'ONE_VISIBLE_PHYSICAL_GPU_REQUIRED')
    name,uuid,total,used=[x.strip() for x in rows[0]]
    require(re.search(r'\bL4\b',name) is not None,'EXACT_L4_REQUIRED_NOT_COAUTHOR_L40S')
    procs=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'])
    processes=[]
    if procs:
        for row in csv.reader(io.StringIO(procs)):
            require(len(row)==3,'NVIDIA_PROCESS_SCHEMA');processes.append({'pid':int(row[0].strip()),'process_name':row[1].strip(),'memory_mib':row[2].strip()})
    require(all(p['pid']==os.getpid() for p in processes),'FOREIGN_GPU_COMPUTE_PROCESS_PRESENT')
    return dict(name=name,uuid=uuid,total_memory_mib=float(total),used_memory_mib=float(used),
        observed_compute_processes=processes,hostname=socket.gethostname(),
        host_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())

def hot_output(root,output,ownership):
    root=Path(root).resolve();out=Path(output).resolve()
    require(not out.is_relative_to(root),'LIVE_OUTPUT_CANNOT_BE_ARTIFACT_SOURCE_TREE')
    bases=[Path('/tmp').resolve()]
    for value in ownership.get('owned_hot_output_roots',[]):
        b=Path(value).resolve()
        require(b.is_absolute() and str(b) not in ('/','/home','/workspace','/mnt'),'NARROW_OWNED_PERSISTENT_ROOT_REQUIRED')
        bases.append(b)
    require(any(out!=b and out.is_relative_to(b) for b in bases),'OUTPUT_NOT_UNDER_TMP_OR_DECLARED_OWNED_DISK')
    return out

def verify_inventory(folder,manifest):
    folder=Path(folder)
    expected=manifest['files']
    require(type(expected) is dict and expected,'NONEMPTY_CLOSED_MANIFEST')
    for name,item in expected.items():
        f=relative(folder,name)
        require(f.stat().st_size==item['bytes'] and sha(f)==item['sha256'],'CLOSED_BODY_MISMATCH:'+name)
    return expected

def validate_training_job(root,folder,case,*,fixture=False):
    folder=Path(folder);c=read_json(folder/'config.json');d=read_json(folder/'TRAIN_COMPLETE.json')
    expected=case['config']
    require(c==expected and d['config_sha256']==digest(expected),'EXACT_TRAINING_CONFIG')
    steps=expected['total_steps'];updates=steps//1024
    require(d['status']=='COMPLETE' and d['run_role']==('readiness_fixture' if fixture else 'scientific') and
            d['env_steps']==steps and d['updates']==updates,'COMPLETE_FIXED_TRAINING_BUDGET')
    require(d['method']==expected['method'] and d['seed']==expected['seed'] and d['run_id']==expected['run_id'],'FINAL_POLICY_IDENTITY')
    require(sha(folder/'CLOSED_FILES.json')==d['closed_manifest_sha256'],'CLOSED_MANIFEST_IDENTITY')
    require(sha(folder/'checkpoint.pt')==d['checkpoint_sha256'] and sha(folder/'final_export.npz')==d['portable_export_sha256'],'FINAL_MODEL_BODY')
    closed=verify_inventory(folder,read_json(folder/'CLOSED_FILES.json'))
    observed={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    require(observed==set(closed)|{'CLOSED_FILES.json','TRAIN_COMPLETE.json'},'EXACT_CLOSED_TRAINING_MEMBERSHIP')
    require({n for n in closed if n.startswith('rollouts/')}=={f'rollouts/update_{i:06}.npz' for i in range(1,updates+1)},'ALL_CLOSED_TRAINING_ROLLOUTS')
    require(d['counters']['oracle_states']==d['counters']['sampled_native_oracle_equal']==steps and
            d['counters']['oracle_candidate_rows']==steps*625,'ALL_SAMPLED_ORACLE_CHECKS')
    require(d['episodes']==steps//256 and sum(map(sum,d['phase_counts']))==steps,'EPISODE_AND_PHASE_ACCOUNTING')
    source_names=['vectorized_admission.py']+['pci_risk_train/'+n for n in
        ('__init__.py','environment.py','model.py','objectives.py','trainer.py','io.py')]
    require(d['sources']=={n:sha(Path(root)/'training'/n) for n in sorted(source_names)},'TRAINING_SOURCE_BODY_CLOSURE')
    # Import Torch only for an actual readiness/scientific execution path.
    import numpy as np
    for n in closed:
        f=relative(folder,n)
        with f.open('rb') as stream:os.fsync(stream.fileno())
        if n.endswith('.npz'):
            with np.load(f,allow_pickle=False) as z:
                require(bool(z.files),'NONEMPTY_RETAINED_ARRAY_FILE')
                for key in z.files:
                    a=z[key]
                    require(a.dtype.kind!='O' and (a.dtype.kind not in 'fc' or np.isfinite(a).all()),'FINITE_RETAINED_ARRAY:'+n+':'+key)
    sync_dir(folder)
    return d

def seal_training_job(folder,archive,deadline):
    folder,archive=Path(folder),Path(archive);archive.parent.mkdir(parents=True,exist_ok=True)
    require(not archive.exists(),'REFUSE_EXISTING_JOB_ARCHIVE')
    partial=archive.with_name(archive.name+'.partial')
    bodies={p.relative_to(folder).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)}
            for p in sorted(folder.rglob('*')) if p.is_file()}
    with partial.open('xb') as raw:
        with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as gz:
            with tarfile.open(fileobj=gz,mode='w|') as tar:
                for name,item in bodies.items():
                    require(time.monotonic()<deadline,'DEADLINE_DURING_JOB_RETENTION')
                    f=relative(folder,name);require(sha(f)==item['sha256'],'JOB_CHANGED_BEFORE_ARCHIVE')
                    info=tarfile.TarInfo(name);info.size=item['bytes'];info.mode=0o600;info.mtime=0
                    with f.open('rb') as stream:tar.addfile(info,stream)
        raw.flush();os.fsync(raw.fileno())
    with tarfile.open(partial,'r:gz') as tar:
        members=tar.getmembers();require(len(members)==len(bodies) and {m.name for m in members}==set(bodies),'JOB_ARCHIVE_MEMBERSHIP')
        for member in members:
            require(member.isfile(),'REGULAR_ARCHIVE_MEMBER')
            stream=tar.extractfile(member);h=hashlib.sha256()
            for b in iter(lambda:stream.read(1<<20),b''):h.update(b)
            require(member.size==bodies[member.name]['bytes'] and h.hexdigest()==bodies[member.name]['sha256'],'JOB_ARCHIVE_BODY')
    require(all(sha(relative(folder,n))==v['sha256'] for n,v in bodies.items()),'JOB_CHANGED_AFTER_ARCHIVE')
    os.replace(partial,archive);sync_dir(archive.parent)
    return dict(schema='PCI-M2-CLOSED-TRAINING-JOB-ARCHIVE-v1',status='PASS',
        archive_path=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),files=bodies)


def numerical_cuda_gate(root,out):
    import numpy as np
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'CUDA_NOT_AVAILABLE_AS_ONE_DEVICE')
    require(re.search(r'\bL4\b',torch.cuda.get_device_properties(0).name) is not None,'TORCH_DEVICE_NOT_OWNED_L4')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','DETERMINISTIC_CUBLAS_CONFIG_REQUIRED')
    sys.path.insert(0,str(root))
    from training.pci_risk_train.model import ContextPolicy
    from training.pci_risk_train.objectives import MODEL_METHOD,institutional_risk,auxiliary
    with np.load(Path(root)/'training_controls/fixture_actor.npz',allow_pickle=False) as z:state={k:torch.from_numpy(z[k].copy()) for k in z.files}
    with np.load(Path(root)/'training_controls/fixture_inputs.npz',allow_pickle=False) as z:
        features=z['features'].copy();costs=z['institutional_replacements'].copy()
    records=[]
    def quantities(method,device):
        with torch.random.fork_rng(devices=[0] if device=='cuda' else []):
            torch.manual_seed(860000901);model=ContextPolicy(MODEL_METHOD[method],64)
        model.load_state_dict(state,strict=True);model=model.to(device)
        x=torch.as_tensor(features,device=device);c=torch.as_tensor(costs,device=device)
        dist,value,heads=model(x)
        raw=torch.stack([head(torch.cat((x*model.masks[i],torch.zeros_like(x)),dim=-1)).squeeze(-1)
                         for i,head in enumerate(model.heads)],dim=-2).mean(-2)
        risks=institutional_risk(dist.probs,c);aux=auxiliary(method,dist,heads,x,c)
        actor_parameters=[p for head in model.heads for p in head.parameters()]
        risk_gradient=torch.autograd.grad(risks.mean(),actor_parameters,retain_graph=True)
        risk_vector=torch.cat([v.reshape(-1) for v in risk_gradient])
        actions=torch.arange(16,device=device).reshape(4,4)%5
        old=torch.as_tensor(np.log(np.full((4,4),.2,dtype=np.float32)),device=device)
        advantage=(torch.arange(16,device=device,dtype=torch.float32).reshape(4,4)-7.5)/8
        ratio=(dist.log_prob(actions)-old).exp()
        actor=torch.maximum(-advantage*ratio,-advantage*ratio.clamp(.8,1.2)).mean()
        loss=actor+.5*(.5*(value-.25).square().mean())-.02*dist.entropy().mean()+.2*aux
        loss.backward()
        gradient=torch.cat([p.grad.reshape(-1) for p in model.parameters()])
        result=dict(logits=raw,probabilities=dist.probs,values=value,risk=risks,
                    auxiliary=aux.reshape(1),loss=loss.reshape(1),risk_gradient=risk_vector,full_gradient=gradient)
        require(all(torch.isfinite(v).all().item() for v in result.values()),'FINITE_CPU_CUDA_NUMERICS')
        return {k:v.detach().cpu().clone() for k,v in result.items()}
    cpu={m:quantities(m,'cpu') for m in METHODS}
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        gpu={m:quantities(m,'cuda') for m in METHODS}
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(Path(out)/'actual_cuda_profile.json'))
    trace=read_json(Path(out)/'actual_cuda_profile.json')
    cuda_events=[e for e in trace['traceEvents'] if str(e.get('cat','')).lower()=='kernel']
    require(len(cuda_events)>0,'NO_ACTUAL_CUDA_KERNEL_EVENTS')
    tolerances={'logits':(TOLS['logits_atol'],0.),'probabilities':(TOLS['probabilities_atol'],0.),
        'values':(TOLS['values_atol'],0.),'risk':(TOLS['risk_atol'],0.),'auxiliary':(TOLS['auxiliary_atol'],0.),
        'loss':(2e-5,1e-4),'risk_gradient':(TOLS['gradient_atol'],TOLS['gradient_rtol']),
        'full_gradient':(TOLS['gradient_atol'],TOLS['gradient_rtol'])}
    for method in METHODS:
        errors={}
        for key,(atol,rtol) in tolerances.items():
            torch.testing.assert_close(cpu[method][key],gpu[method][key],atol=atol,rtol=rtol)
            errors[key]=float((cpu[method][key]-gpu[method][key]).abs().max())
        records.append({'method':method,'status':'PASS','max_absolute_errors':errors})
    runtime=dict(torch_version=torch.__version__,torch_cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),numpy_version=np.__version__,python=platform.python_version(),
        actual_device_name=torch.cuda.get_device_properties(0).name,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),tf32_matmul=False,tf32_cudnn=False)
    result=dict(schema='PCI-M2-ACTUAL-CPU-CUDA-NUMERICAL-GATE-v1',status='PASS',runtime=runtime,
        records=records,tolerances=TOLS,actual_cuda_kernel_events=len(cuda_events),
        profiler_trace_sha256=sha(Path(out)/'actual_cuda_profile.json'),
        scope='Actually executed fixed-input float32 CPU/CUDA forward, exact-risk and gradient comparisons. Tolerance agreement is not bitwise equality, trajectory equivalence or derivative-rounding certification.')
    write_json(Path(out)/'CUDA_NUMERICAL_GATE.json',result);return result


def fixture_cases(protocol):
    methods=('task','fixed','adaptive','semantic','consensus','directrisk','semantic','directrisk')
    template=protocol['training_cases'][0]['config'];result=[]
    for i,method in enumerate(methods):
        c=copy.deepcopy(template)
        c.update(method=method,run_id=f'engineering_{i:02}_{method}',run_role='readiness_fixture',
            total_steps=FIXTURE_STEPS,seed=860000000+i,init_seed=860000000+i,
            env_seeds=[860000100+8*i+j for j in range(8)],policy_seed=860000300+i,
            minibatch_seed=860000400+i,checkpoint_every_updates=8)
        result.append({'case_id':c['run_id'],'method':method,'training_seed':c['seed'],'config':c})
    return result

def timing_training_scale(protocol):
    cases=protocol['training_cases']
    require(type(FIXTURE_STEPS) is int and FIXTURE_STEPS>0,'POSITIVE_INTEGER_FIXTURE_STEP_BUDGET')
    require(len(cases)==60 and MAX_WORKERS==4,'EXACT_FORECAST_COHORT_AND_WORKER_BUDGET')
    budgets={c['config']['total_steps'] for c in cases}
    require(all(type(c['config']['total_steps']) is int for c in cases) and
            budgets=={262144} and 262144%FIXTURE_STEPS==0,'INTEGER_FULL_TO_FIXTURE_STEP_RATIO_REQUIRED')
    for c in cases:
        cfg=c['config'];steps_per_update=cfg['n_envs']*cfg['rollout_steps']
        require(FIXTURE_STEPS%steps_per_update==0 and FIXTURE_STEPS>=steps_per_update,
                'FIXTURE_MUST_CONTAIN_COMPLETE_UPDATES')
    return dict(full_training_steps_per_policy=262144,fixture_steps_per_policy=FIXTURE_STEPS,
        full_to_fixture_step_ratio=262144//FIXTURE_STEPS,
        complete_training_cases=len(cases),workers_per_wave=MAX_WORKERS,
        complete_training_waves=math.ceil(len(cases)/MAX_WORKERS))


def timing_evaluation_cases(fixtures):
    """Two fixed engineering blocks declared before any new trajectory."""
    blocks=[]
    for block in range(2):
        cases=[]
        for i,c in enumerate(fixtures):
            for j,profile in enumerate(PROFILES):
                label=f'engineering_eval_{i:02}_{profile}' if block==0 else f'engineering_eval_b01_{i:02}_{profile}'
                cases.append(dict(case_id=label,training_case_id=c['case_id'],method=c['method'],
                    training_seed=c['training_seed'],profile_id=profile,horizon=256,world_index=block,
                    world_seed=860000500+1000*block+4*i+j,
                    policy_sampling_seed=860000600+1000*block+4*i+j))
        require(len(cases)==32,'EXACT_32_ENGINEERING_EVALUATIONS_PER_BLOCK')
        blocks.append(cases)
    require(len({c['case_id'] for cases in blocks for c in cases})==64,
            'UNIQUE_64_ENGINEERING_EVALUATION_CASES')
    return blocks


def timing_evaluation_scale(protocol,blocks):
    require(len(blocks)==2 and all(b['complete_cases']==32 for b in blocks),
            'TWO_COMPLETE_32_CASE_PARALLEL_BLOCKS_REQUIRED')
    durations=[b['elapsed_with_retained_audit_seconds'] for b in blocks]
    require(all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in durations),
            'POSITIVE_FINITE_COMPLETE_PARALLEL_BLOCK_DURATIONS_REQUIRED')
    require(len(protocol['evaluation_cases'])==1200,'EXACT_1200_FORECAST_EVALUATION_CASES')
    slowest=max(durations);repeats=math.ceil(len(protocol['evaluation_cases'])/32)
    return dict(measured_blocks=2,cases_per_measured_block=32,workers_per_phase=4,
        measured_native_and_independent_audit_phases_per_block=2,
        slowest_complete_parallel_block_seconds=slowest,complete_block_repetitions=repeats,
        forecast_seconds=slowest*repeats,
        formula='max(measured complete native-plus-independent-audit block wall times) * ceil(1200 / 32)',
        process_startup_and_retention_included=True)


def timing_primary_inventory(root,output,cases,receipt,deadline):
    """Closed engineering bodies only; no world, actor forward or model change."""
    output=Path(output)
    expected={prefix+c['case_id']+suffix for c in cases for prefix,suffix in
              (('cases/','.json'),('traces/','.jsonl.gz'))}
    actual={p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}
    require(actual==expected,'EXACT_ENGINEERING_PRIMARY_MEMBERSHIP')
    files={}
    for name in sorted(expected):
        require(time.monotonic()<deadline,'DEADLINE_DURING_ENGINEERING_INVENTORY')
        f=relative(output,name)
        require(f.is_file() and not f.is_symlink() and f.stat().st_size>0,'CLOSED_ENGINEERING_PRIMARY_FILE')
        files[name]=dict(bytes=f.stat().st_size,sha256=sha(f))
    for c in cases:
        summary=read_json(output/'cases'/(c['case_id']+'.json'))
        require(summary['case']==c and summary['trace_sha256']==files[summary['trace_path']]['sha256'],
                'ENGINEERING_INVENTORY_CASE_IDENTITY')
    write_json(receipt,dict(schema='PCI-M2-ENGINEERING-PRIMARY-INVENTORY-v1',files=files,
        complete_cases=len(cases),source_sha256=source_closure(root,Path(root)/'design/M2_PROTOCOL.json')))
    return sha(receipt)


def timing_overhead_scale(protocol,training_scale,validation_seconds,inventory_seconds):
    require(len(validation_seconds)==8 and len(inventory_seconds)==2,
            'ALL_EIGHT_MODEL_AND_TWO_INVENTORY_OVERHEAD_MEASUREMENTS_REQUIRED')
    require(all(type(v) in (int,float) and math.isfinite(v) and v>0
                for v in validation_seconds+inventory_seconds),'POSITIVE_FINITE_OVERHEAD_MEASUREMENTS')
    model=max(validation_seconds)*len(protocol['training_cases'])*training_scale['full_to_fixture_step_ratio']
    repeats=math.ceil(len(protocol['evaluation_cases'])/32)
    inventory=max(inventory_seconds)*repeats*3
    reserve=300.0
    return dict(measured_training_revalidation_seconds=validation_seconds,
        slowest_model_revalidation_seconds=max(validation_seconds),
        model_validation_forecast_seconds=model,model_count=len(protocol['training_cases']),
        rollout_count_scale=training_scale['full_to_fixture_step_ratio'],
        model_validation_scope='Existing validate_training_job closes hashes and exact membership, fsyncs files and decompresses/scans every retained NPZ array; max of eight fixtures is repeated for all sixty models and multiplied by full/fixture rollout-count ratio, also overstating fixed model/source work.',
        measured_primary_inventory_seconds=inventory_seconds,
        slowest_primary_inventory_seconds=max(inventory_seconds),complete_block_repetitions=repeats,
        additional_full_primary_inventory_passes=3,primary_inventory_forecast_seconds=inventory,
        inventory_scope='Each measured 32-case closed-body hash/JSON/source-membership scan and external receipt write is repeated for ceil(1200/32) blocks and for three additional native-manifest, retained-audit precheck and analysis precheck passes.',
        unmeasured_coordinator_checkpoint_parse_and_analysis_reserve_seconds=reserve,
        reserve_scope='Positive 300-second engineering reserve for parent coordinator/source/provenance work, sixty checkpoint parses, analysis arithmetic and receipts. Those operations are not exactly benchmarked by the shortened engineering cohort.',
        forecast_seconds=model+inventory+reserve,
        limitation='A conservative readiness estimate under the measured hardware/filesystem/cache and workload conditions, not a proved upper bound; state-dependent cost and future contention may exceed it. The absolute global cutoff still binds.')


def timing_gate(root,protocol,protocol_sha,out,deadline):
    """Actually runs future engineering fixtures; never called by plan/check."""
    out=Path(out);out.mkdir(mode=0o700)
    scale=timing_training_scale(protocol)
    fixtures=fixture_cases(protocol);case_blocks=timing_evaluation_cases(fixtures)
    write_json(out/'ENGINEERING_CASE_PLAN.json',dict(schema='PCI-M2-PARALLEL-TIMING-ENGINEERING-PLAN-v1',
        role='ENGINEERING_ONLY_NOT_SCIENTIFIC_COHORT',protocol_sha256=protocol_sha,
        declared_before_new_training_or_evaluation_trajectories=True,training_cases=fixtures,
        evaluation_blocks=case_blocks,training_joint_steps=8*FIXTURE_STEPS,
        new_evaluation_cases=64,new_evaluation_joint_steps=16384,
        retained_audit_cases=64,retained_audit_new_native_task_worlds=0,
        max_workers_per_phase=4,case_retries=0,
        forecast_formula='training complete-path forecast + max(two complete 32-case native-plus-independent-audit parallel block wall times) * ceil(1200 / 32) + separately measured/scaled validation and three inventory passes + 300-second reserve'))
    configdir=out/'configs';configdir.mkdir()
    trainroot=out/'training';trainroot.mkdir();archives=out/'closed_training';seals={};summaries={}
    for c in fixtures:write_json(configdir/(c['case_id']+'.json'),c['config'])
    waves=[]
    for wave in range(2):
        subset=fixtures[4*wave:4*wave+4]
        jobs=[{'case_id':c['case_id'],'case':c,'argv':[sys.executable,'-B','-m',
            'training.pci_risk_train.trainer','--config',str(configdir/(c['case_id']+'.json')),
            '--out',str(trainroot/c['case_id'])]} for c in subset]
        def completed(job,row):
            c=job['case'];folder=trainroot/c['case_id']
            summaries[c['case_id']]=validate_training_job(root,folder,c,fixture=True)
            seals[c['case_id']]=seal_training_job(folder,archives/(c['case_id']+'.tar.gz'),deadline)
            write_json(out/(c['case_id']+'_CLOSED_RETENTION.json'),seals[c['case_id']])
        begin=time.monotonic()
        result=run_pool(jobs,root,out/f'wave_{wave}_logs',deadline,workers=4,on_complete=completed)
        elapsed=time.monotonic()-begin
        rows=result['rows'];overlap=sum(r['seconds_lower_bound'] for r in result['actual_live_concurrency_intervals'])
        require(result['peak_parallel_workers']==4 and overlap>0,'ACTUAL_FOUR_WAY_FULL_PATH_OVERLAP_REQUIRED')
        waves.append(dict(wave=wave,elapsed_seconds=elapsed,actual_four_way_overlap_seconds=overlap,
            rows=rows,peak_parallel_workers=4,live_observations=result['actual_live_concurrency_intervals']))
    require(len(summaries)==len(seals)==8,'ALL_EIGHT_ENGINEERING_JOBS_REQUIRED')
    validation_seconds=[]
    for c in fixtures:
        require(time.monotonic()<deadline,'DEADLINE_DURING_ENGINEERING_MODEL_REVALIDATION')
        validation_begin=time.monotonic()
        require(validate_training_job(root,trainroot/c['case_id'],c,fixture=True)==summaries[c['case_id']],
                'ENGINEERING_MODEL_REVALIDATION_IDENTITY')
        validation_seconds.append(time.monotonic()-validation_begin)
    sys.path.insert(0,str(root))
    from evaluation.engine import PROFILES as profiles
    from evaluation.parallel import run_phase
    bindings={};export_paths={}
    for c in fixtures:
        done=summaries[c['case_id']];folder=trainroot/c['case_id']
        bindings[c['case_id']]=dict(signature=done['signature'],checkpoint_sha256=done['checkpoint_sha256'],
            portable_export_sha256=done['portable_export_sha256'])
        export_paths[c['case_id']]=str(folder/'final_export.npz')
    episodes=[];evaluation_blocks=[];inventory_seconds=[]
    for block,cases in enumerate(case_blocks):
        require(time.monotonic()<deadline,'DEADLINE_DURING_ENGINEERING_EVALUATION')
        evalroot=out/f'evaluation_block_{block:02}';evalroot.mkdir()
        begin=time.monotonic()
        native=run_phase(root,evalroot,cases,bindings,protocol_sha,'native',
            out/f'evaluation_operations/block_{block:02}/native',deadline,
            fixture=True,expected_configs=profiles,workers=4,export_paths=export_paths)
        audited=run_phase(root,evalroot,cases,bindings,protocol_sha,'audit',
            out/f'evaluation_operations/block_{block:02}/audit',deadline,
            fixture=True,expected_configs=profiles,workers=4,export_paths=export_paths)
        require(native['summaries']==audited['summaries'],'ENGINEERING_CLOSED_EPISODE_AUDIT')
        elapsed=time.monotonic()-begin
        phase_observations={}
        for phase,result in (('native',native),('independent_retained_audit',audited)):
            observed=result['receipt']['process'];intervals=observed['actual_live_concurrency_intervals']
            overlap=sum(r['seconds_lower_bound'] for r in intervals)
            require(observed['peak_parallel_workers']==4 and overlap>0 and
                    all(len(set(r['pids']))==4 for r in intervals),
                    'ACTUAL_FOUR_WAY_EVALUATION_AND_AUDIT_OVERLAP_REQUIRED:'+phase)
            phase_observations[phase]=dict(actual_four_way_overlap_seconds=overlap,
                peak_parallel_workers=4,live_observations=intervals)
        require(len(native['summaries'])==32,'ALL_32_BLOCK_EPISODES_REQUIRED')
        inventory_begin=time.monotonic()
        inventory_sha=timing_primary_inventory(root,evalroot,cases,out/f'PRIMARY_INVENTORY_BLOCK_{block:02}.json',deadline)
        inventory_seconds.append(time.monotonic()-inventory_begin)
        evaluation_blocks.append(dict(block=block,complete_cases=32,joint_steps=8192,
            elapsed_with_retained_audit_seconds=elapsed,phases=phase_observations,
            primary_inventory_receipt_sha256=inventory_sha,primary_inventory_elapsed_seconds=inventory_seconds[-1]))
        episodes.extend(dict(block=block,case_id=summary['case']['case_id'],trace_bytes=summary['trace_bytes'],
            trace_sha256=summary['trace_sha256'],joint_steps=256) for summary in native['summaries'])
    require(len(episodes)==64,'ALL_64_ENGINEERING_EVALUATION_CASES_REQUIRED')
    # Each measured block includes both fresh Python process startup phases,
    # native/trace/case retention and a separate complete retained-record audit.
    slowest_wave=max(w['elapsed_seconds'] for w in waves)
    evaluation_scale=timing_evaluation_scale(protocol,evaluation_blocks)
    training_forecast=slowest_wave*scale['full_to_fixture_step_ratio']*scale['complete_training_waves']
    evaluation_forecast=evaluation_scale['forecast_seconds']
    overhead_scale=timing_overhead_scale(protocol,scale,validation_seconds,inventory_seconds)
    forecast=training_forecast+evaluation_forecast+overhead_scale['forecast_seconds']
    training_bytes_per_step=max((sum(v['bytes'] for v in seals[c['case_id']]['files'].values())+
        seals[c['case_id']]['archive_bytes'])/FIXTURE_STEPS for c in fixtures)
    eval_bytes_per_step=max(e['trace_bytes']/256 for e in episodes)
    # Keep both live files and their independently closed archives. Evaluation
    # and its future export copy are also budgeted twice. Five GiB is reserved.
    required_free=math.ceil(1.5*(training_bytes_per_step*15728640+
        2*eval_bytes_per_step*307200)+5*(1<<30))
    mount_error=disk_usage_error=None
    try:
        mount=subprocess.run(['findmnt','-n','-T',str(out),'-o','FSTYPE,SOURCE,TARGET'],capture_output=True,text=True,timeout=10)
        mount_returncode,mount_stdout,mount_stderr=mount.returncode,mount.stdout.strip(),mount.stderr.strip()
    except (OSError,subprocess.TimeoutExpired) as exc:
        mount_returncode,mount_stdout,mount_stderr=None,'',''
        mount_error=repr(exc)
    filesystem_type=mount_stdout.split()[0] if mount_stdout else None
    try:free=shutil.disk_usage(out).free
    except OSError as exc:free=None;disk_usage_error=repr(exc)
    measured=dict(schema='PCI-M2-ACTUAL-PARALLEL-FULL-PATH-TIMING-MEASUREMENTS-v3',status='MEASURED_NOT_PASSED',
        fixture_role='ENGINEERING_NOT_SCIENTIFIC_COHORT',fixture_training_jobs=8,
        fixture_training_joint_steps=8*FIXTURE_STEPS,fixture_evaluation_cases=64,
        fixture_evaluation_joint_steps=16384,waves=waves,evaluation=episodes,
        evaluation_blocks=evaluation_blocks,training_extrapolation=scale,evaluation_extrapolation=evaluation_scale,
        overhead_extrapolation=overhead_scale,
        engineering_case_plan_sha256=sha(out/'ENGINEERING_CASE_PLAN.json'),
        slowest_measured_four_worker_wave_seconds=slowest_wave,
        slowest_measured_parallel_block_with_retained_audit_seconds=evaluation_scale['slowest_complete_parallel_block_seconds'],
        complete_evaluation_cases=len(protocol['evaluation_cases']),
        training_full_path_forecast_seconds=training_forecast,
        evaluation_with_retained_audit_forecast_seconds=evaluation_forecast,
        cohort_validation_inventory_and_analysis_overhead_forecast_seconds=overhead_scale['forecast_seconds'],
        complete_scientific_forecast_seconds=forecast,maximum_allowed_forecast_seconds=FORECAST_MAX,
        storage=dict(free_bytes=free,required_free_bytes=required_free,filesystem_observation=mount_stdout,
            filesystem_type=filesystem_type,mount_returncode=mount_returncode,mount_stderr=mount_stderr,
            mount_observation_error=mount_error,disk_usage_error=disk_usage_error,
            measured_max_training_live_plus_archive_bytes_per_step=training_bytes_per_step,
            measured_max_compressed_evaluation_bytes_per_step=eval_bytes_per_step,
            multiplier=1.5,reserve_bytes=5*(1<<30)),
        scope='Measured concurrent engineering trajectory/update/checkpoint/compression/hash validation and CPU episode/reconciliation paths. The extrapolation is a readiness estimate; it does not guarantee future state-dependent cost, storage use, or wall time. The eight-hour controller ceiling and supplied absolute cutoff remain binding.')
    write_json(out/'TIMING_MEASUREMENTS.json',measured)
    require(math.isfinite(forecast) and forecast<=FORECAST_MAX,'FULL_PATH_FORECAST_EXCEEDS_SIX_HOURS')
    require(mount_returncode==0 and mount_stdout,'HOT_FILESYSTEM_OBSERVATION_REQUIRED')
    require(filesystem_type not in ('tmpfs','ramfs','overlay'),'PERSISTENT_OWNED_DISK_REQUIRED_NOT_RAM_OR_CONTAINER_LAYER')
    require(free is not None and free>=required_free,'INSUFFICIENT_OWNED_HOT_DISK_FOR_COMPLETE_COHORT')
    result=dict(measured,schema='PCI-M2-ACTUAL-PARALLEL-FULL-PATH-TIMING-v2',status='PASS',
        measurements_receipt_sha256=sha(out/'TIMING_MEASUREMENTS.json'))
    write_json(out/'FULL_PATH_TIMING_AND_STORAGE.json',result);return result


def evaluator_source_admission(root,plan,receipt):
    """Check evaluator pins and construct its adapter, without a task world."""
    root=Path(root).resolve();receipt=Path(receipt)
    names=('evaluation/engine.py','evaluation/reference/scalar_world.py',
           'training/pci_risk_train/environment.py')
    observed={n:sha(root/n) for n in names}
    result=dict(schema='PCI-M2-EVALUATOR-SOURCE-ADMISSION-v1',
        source_map_sha256=plan['source_map_sha256'],source_sha256=observed,
        adapter_only=True,native_world_constructors=0,native_task_transition_calls=0,
        training_GPU_proof_cloud_calls=0,
        scope='Source-map and engine/scalar/native pin checks plus WorldAdapter construction only. No Warehouse constructor, reset, native transition, trajectory, training or Torch forward is called.')
    try:
        require(all(plan['source_sha256'].get(n)==observed[n] for n in names),
                'EVALUATOR_ADMISSION_SOURCE_MAP_MISMATCH')
        sys.path.insert(0,str(root))
        from evaluation import engine
        require(Path(engine.__file__).resolve()==root/'evaluation/engine.py',
                'EVALUATOR_ENGINE_IMPORT_PATH')
        require(Path(engine.scalar.__file__).resolve()==root/'evaluation/reference/scalar_world.py',
                'EVALUATOR_SCALAR_IMPORT_PATH')
        adapter=engine.checked_adapter(root)
        require(Path(adapter.mod.__file__).resolve()==root/'training/pci_risk_train/environment.py',
                'EVALUATOR_NATIVE_IMPORT_PATH')
        require({n:sha(root/n) for n in names}==observed,'EVALUATOR_SOURCE_CHANGED_DURING_ADMISSION')
    except Exception as exc:
        write_json(receipt,dict(result,status='FAIL',error=repr(exc)))
        raise
    write_json(receipt,dict(result,status='PASS'))
    return read_json(receipt)


def executable_gates(root,protocol_path,expected_sha,ownership_path,out,deadline):
    plan=make_plan(root,protocol_path,expected_sha);protocol=read_json(protocol_path)
    validate_protocol(protocol,frozen=True)
    observed=check_ownership(read_json(ownership_path))
    device=gpu_identity()
    require(observed['host']['hostname']==device['hostname'] and
            observed['host']['boot_id']==device['host_boot_id'] and
            observed['gpu_uuid']==device['uuid'],'OPERATOR_RECORD_LOCAL_HOST_GPU_MISMATCH')
    out=Path(out);out.mkdir(mode=0o700)
    write_json(out/'INPUT_PLAN.json',plan)
    write_json(out/'OWNED_HOST_OBSERVATION.json',observed)
    write_json(out/'ACTUAL_IDLE_GPU_IDENTITY.json',device)
    write_json(out/'GATES_STARTED.json',dict(status='RUNNING_ENGINEERING_GATES',scientific_training_started=False,
        source_map_sha256=plan['source_map_sha256'],protocol_sha256=plan['protocol_sha256']))
    try:
        evaluator_source_admission(root,plan,out/'EVALUATOR_SOURCE_ADMISSION.json')
        control_script=Path(root)/'training_controls/check_training.py'
        receipt=out/'CPU_CONTROLS.json'
        run_one([sys.executable,'-B',str(control_script),'--output',str(receipt),
            '--deadline-seconds','85'],root,out/'cpu_control_logs',min(deadline,time.monotonic()+100))
        cpu=read_json(receipt)
        require(cpu['status']=='PASS' and cpu['sources_unchanged'] is True and
                cpu['fixture_source_sha256']==sha(control_script),'ACTUAL_SOURCE_BOUND_CPU_CONTROLS_REQUIRED')
        expected_sources={n:sha(Path(root)/'training'/n) for n in
            ['vectorized_admission.py']+['pci_risk_train/'+n for n in
             ('__init__.py','environment.py','model.py','objectives.py','trainer.py','io.py')]}
        require(cpu['sources']==dict(sorted(expected_sources.items())),'CPU_CONTROL_RUNTIME_CLOSURE')
        require(source_closure(root,protocol_path)==plan['source_sha256'],'SOURCE_CHANGED_DURING_CPU_CONTROLS')
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        numerical=numerical_cuda_gate(root,out)
        timing=timing_gate(root,protocol,plan['protocol_sha256'],out/'timing',min(deadline,time.monotonic()+3600))
        require(source_closure(root,protocol_path)==plan['source_sha256'],'SOURCE_CHANGED_DURING_GPU_GATES')
        device_after=gpu_identity()
        require(device_after['uuid']==device['uuid'],'GPU_IDENTITY_CHANGED_DURING_GATES')
        require(time.monotonic()+timing['complete_scientific_forecast_seconds']<=deadline,'FORECAST_DOES_NOT_FIT_REMAINING_GLOBAL_BUDGET')
        result=dict(schema='PCI-M2-EXECUTED-GPU-READINESS-GATE-v1',status='PASS',
            plan_source_map_sha256=plan['source_map_sha256'],protocol_sha256=plan['protocol_sha256'],
            actual_device=device_after,runtime=numerical['runtime'],
            evaluator_source_admission_sha256=sha(out/'EVALUATOR_SOURCE_ADMISSION.json'),
            cpu_receipt_sha256=sha(receipt),numerical_receipt_sha256=sha(out/'CUDA_NUMERICAL_GATE.json'),
            actual_parallel_timing_receipt_sha256=sha(out/'timing/FULL_PATH_TIMING_AND_STORAGE.json'),
            complete_scientific_forecast_seconds=timing['complete_scientific_forecast_seconds'],
            no_supplied_PASS_override=True,remote_attestation_claim=False,
            scope='Locally executed readiness on the operator-identified owned L4. This does not authorize another host, a changed source/configuration, an unbounded cohort, or a generic performance/safety claim.')
        write_json(out/'GPU_READINESS_PASS.json',result);return plan,result
    except BaseException as exc:
        write_json(out/'GPU_READINESS_FAILED.json',dict(schema='PCI-M2-GPU-READINESS-FAILURE-v1',
            status='NOT_GPU_LAUNCH_READY',error=repr(exc),scientific_training_started=False,
            all_scientific_training_cases_unrun=60,all_scientific_evaluation_cases_unrun=1200))
        raise


def controller(root,protocol_path,expected_sha,ownership_path,output,*,gate_only=False,not_after_unix=None):
    """One new cohort only. Per-policy exact checkpoints are retained on failure.

    Deliberately no automatic cohort retry or supplied-PASS reuse. This version
    refuses an existing output root. Exact checkpoint resume is supported by the
    separately tested trainer, but a future controller resume admission requires
    its own source/runtime/budget validation and is not silently enabled here.
    """
    root=Path(root).resolve();protocol_path=Path(protocol_path).resolve()
    for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[variable]='1'
    plan=make_plan(root,protocol_path,expected_sha);protocol=read_json(protocol_path)
    validate_protocol(protocol,frozen=True);ownership=check_ownership(read_json(ownership_path))
    output=hot_output(root,output,ownership)
    require(not output.exists(),'EXISTING_COHORT_ROOT_REFUSED_NO_AUTOMATIC_RETRY')
    started=time.monotonic();started_unix=time.time()
    effective_budget=global_budget(not_after_unix,started_unix=started_unix)
    deadline=started+effective_budget
    budget_record=dict(controller_started_unix=started_unix,
        requested_not_after_unix=not_after_unix,effective_global_budget_seconds=effective_budget,
        effective_deadline_unix=started_unix+effective_budget,
        global_ceiling_seconds=CEILING)
    output.mkdir(mode=0o700,parents=True)
    write_json(output/'HOT_ROOT_OWNERSHIP.json',dict(schema='PCI-M2-OWNED-HOT-ROOT-v1',
        study_id=protocol['study_id'],protocol_sha256=plan['protocol_sha256'],project_root=str(root),
        owned_hot_output_root=str(output),created_by_controller=True))
    watchdog_process,watchdog_stream=start_watchdog(output,remaining_global_budget(deadline),
        absolute_deadline_monotonic=deadline)
    rows=[dict(kind=r['kind'],case_id=r['case_id'],status='UNRUN') for r in plan['case_ledger']]
    ledger_path=output/'COHORT_CASE_STATUS.json'
    write_json(ledger_path,dict(status='UNRUN_GATES_PENDING',rows=rows))
    write_json(output/'FROZEN_INPUT_PLAN.json',plan)
    try:
        write_json(output/'CONTROLLER_STARTED.json',dict(
            schema='PCI-M2-CONTROLLER-STARTED-v2',status='RUNNING_ENGINEERING_GATES',
            protocol_sha256=plan['protocol_sha256'],source_map_sha256=plan['source_map_sha256'],
            scientific_training_started=False,**budget_record))
        plan,gate=executable_gates(root,protocol_path,expected_sha,ownership_path,output/'gate',deadline)
        if gate_only:
            write_json(output/'GATE_ONLY_COMPLETE.json',dict(status='GATES_PASSED_SCIENTIFIC_COHORT_UNRUN',
                readiness_sha256=sha(output/'gate/GPU_READINESS_PASS.json'),training_cases_run=0,evaluation_cases_run=0))
            return dict(status='GATES_PASSED_SCIENTIFIC_COHORT_UNRUN')
        configroot=output/'configs';configroot.mkdir();trainroot=output/'training';trainroot.mkdir()
        cases=protocol['training_cases'];seals={};done={}
        for c in cases:
            write_json(configroot/(c['case_id']+'.json'),c['config'])
            require(sha(configroot/(c['case_id']+'.json'))==hashlib.sha256(encoded(c['config'])).hexdigest(),'WRITTEN_CONFIG_BODY')
        lock=dict(schema='PCI-M2-SOURCE-BOUND-CONTROLLER-EXECUTION-LOCK-v1',
            scientific_launch_authorized=True,protocol_path=plan['protocol_path'],
            protocol_sha256=plan['protocol_sha256'],source_sha256=plan['source_sha256'],
            case_ledger=plan['case_ledger'],case_ledger_sha256=plan['case_ledger_sha256'],
            readiness_receipt_sha256=sha(output/'gate/GPU_READINESS_PASS.json'),actual_device=gate['actual_device'],
            worker_ceiling=4,started_unix=time.time(),**budget_record,
            no_outcome_stopping_or_case_replacement=True)
        write_json(output/'EXECUTION_LOCK.json',lock)
        def completed(job,row):
            c=job['case'];folder=trainroot/c['case_id']
            done[c['case_id']]=validate_training_job(root,folder,c)
            require(source_closure(root,protocol_path)==plan['source_sha256'],'SOURCE_CHANGED_DURING_SCIENCE')
            seals[c['case_id']]=seal_training_job(folder,output/'closed_training'/(c['case_id']+'.tar.gz'),deadline)
            write_json(output/'retention'/(c['case_id']+'.json'),seals[c['case_id']])
            next(r for r in rows if r['kind']=='training' and r['case_id']==c['case_id'])['status']='COMPLETE_CLOSED_VALIDATED'
            write_json(ledger_path,dict(status='TRAINING_RUNNING',rows=rows),replace=True)
        jobs=[dict(case_id=c['case_id'],case=c,argv=[sys.executable,'-B','-m','training.pci_risk_train.trainer',
            '--config',str(configroot/(c['case_id']+'.json')),'--out',str(trainroot/c['case_id'])]) for c in cases]
        result=run_pool(jobs,root,output/'training_logs',deadline,workers=4,on_complete=completed)
        write_json(output/'TRAINING_POOL_COMPLETE.json',result)
        require(len(done)==len(seals)==60,'ALL_SIXTY_CLOSED_TRAINING_MODELS_REQUIRED')
        initial={}
        for c in cases:
            d=done[c['case_id']];key=c['training_seed']
            pair=(d['initial_vector_sha256'],d['initial_actor_vector_sha256'])
            require(initial.setdefault(key,pair)==pair,'ALL_SIX_WITHIN_SEED_INITIALIZATIONS_EQUAL')
        sys.path.insert(0,str(root))
        from evaluation.engine import source_closure as evaluation_sources
        evallock=dict(schema='PCI-M2-EVALUATION-EXECUTION-LOCK-v1',scientific_launch_authorized=True,
            protocol_path=plan['protocol_path'],protocol_sha256=plan['protocol_sha256'],
            source_sha256=evaluation_sources(root),wall_ceiling_seconds=max(1.,deadline-time.monotonic()),
            owned_hot_output_root=str(output),owned_hot_output_marker_sha256=sha(output/'HOT_ROOT_OWNERSHIP.json'))
        write_json(output/'EVALUATION_EXECUTION_LOCK.json',evallock)
        evalout=output/'evaluation'
        run_one([sys.executable,'-B','-m','evaluation.engine','--project-root',str(root),
            '--training-root',str(trainroot),'--execution-lock',str(output/'EVALUATION_EXECUTION_LOCK.json'),
            '--output',str(evalout),'--operations-root',str(output/'evaluation_operations'),
            '--deadline-monotonic',str(deadline)],root,output/'evaluation_logs',deadline)
        run_one([sys.executable,'-B','-m','evaluation.retained_audit','--project-root',str(root),
            '--training-root',str(trainroot),'--protocol',str(protocol_path),'--output',str(evalout),
            '--receipt',str(output/'RETAINED_EVALUATION_AUDIT.json'),
            '--operations-root',str(output/'audit_operations'),'--deadline-monotonic',str(deadline)],
            root,output/'audit_logs',deadline)
        audit=read_json(output/'RETAINED_EVALUATION_AUDIT.json')
        require(audit['status']=='PASS' and audit['checked_cases']==1200 and
                audit['checked_joint_steps']==307200 and audit['model_inputs']==60,'ALL_1200_RETAINED_CASES_VALIDATED')
        for r in rows:
            if r['kind']=='evaluation':r['status']='COMPLETE_CLOSED_VALIDATED'
        write_json(ledger_path,dict(status='ALL_60_AND_1200_VALIDATED',rows=rows),replace=True)
        run_one([sys.executable,'-B','-m','evaluation.analysis','--protocol',str(protocol_path),
            '--output',str(evalout),'--audit',str(output/'RETAINED_EVALUATION_AUDIT.json'),
            '--audit-sha256',sha(output/'RETAINED_EVALUATION_AUDIT.json'),
            '--receipt',str(output/'ANALYSIS.json')],root,output/'analysis_logs',deadline)
        require(source_closure(root,protocol_path)==plan['source_sha256'],'FINAL_SOURCE_CLOSURE_CHANGED')
        require(time.monotonic()<=deadline,'GLOBAL_WALL_CEILING_AT_FINAL_CLOSURE')
        result=dict(schema='PCI-M2-COMPLETE-VALIDATED-COHORT-v1',status='COMPLETE_VALIDATED_COHORT',
            trained_policies=60,evaluated_cases=1200,training_joint_steps=15728640,
            evaluation_joint_steps=307200,protocol_sha256=plan['protocol_sha256'],
            execution_lock_sha256=sha(output/'EXECUTION_LOCK.json'),
            retained_audit_sha256=sha(output/'RETAINED_EVALUATION_AUDIT.json'),
            analysis_sha256=sha(output/'ANALYSIS.json'),elapsed_seconds=time.monotonic()-started,
            all_owned_worker_processes_closed=True,cloud_resource_state_changed_by_controller=False,**budget_record)
        write_json(output/'STUDY_COMPLETE.json',result);return result
    except BaseException as exc:
        # Pool failures retain their own per-case process rows. Scientific
        # evaluation's writer separately preserves its exact failed/unrun index.
        poolfile=output/'training_logs/POOL_FAILED.json'
        if poolfile.exists():
            process_rows=read_json(poolfile)['rows']
            statuses={r['case_id']:r['status'] for r in process_rows}
            for r in rows:
                if r['kind']=='training' and r['status']!='COMPLETE_CLOSED_VALIDATED':
                    r['status']=statuses.get(r['case_id'],'UNRUN_AFTER_COHORT_FAILURE')
        evaluation_root=output/'evaluation'
        failure_file=evaluation_root/'EVALUATION_FAILED.json'
        failure_receipt_errors={}
        try:evaluation_failure=read_json(failure_file) if failure_file.exists() else {}
        except Exception as receipt_error:
            evaluation_failure={};failure_receipt_errors[str(failure_file)]=repr(receipt_error)
        failed_eval_id=evaluation_failure.get('failed_case_id')
        evaluation_statuses={r['case_id']:r for r in evaluation_failure.get('case_statuses',[])}
        audit_failure_file=output/'audit_operations/PHASE_FAILED.json'
        try:audit_failure=read_json(audit_failure_file) if audit_failure_file.exists() else {}
        except Exception as receipt_error:
            audit_failure={};failure_receipt_errors[str(audit_failure_file)]=repr(receipt_error)
        audit_statuses={r['case_id']:r for r in audit_failure.get('case_statuses',[])}
        # A killed phase parent may leave only fsynced per-case operation records.
        for operation,ops,statuses in (('native',output/'evaluation_operations',evaluation_statuses),
                                       ('audit',output/'audit_operations',audit_statuses)):
            if not statuses and (ops/'case_status').is_dir():
                from evaluation.parallel import phase_ledger
                statuses.update({r['case_id']:r for r in phase_ledger(evaluation_root,ops,protocol['evaluation_cases'],operation)})
        for r in rows:
            if r['kind']=='evaluation' and r['status']=='UNRUN':
                summary=evaluation_root/'cases'/(r['case_id']+'.json')
                partial=evaluation_root/'traces'/(r['case_id']+'.jsonl.gz.partial')
                trace=evaluation_root/'traces'/(r['case_id']+'.jsonl.gz')
                declared=evaluation_statuses.get(r['case_id'])
                if declared is not None:
                    # Preserve the evaluator's complete parallel failed/stopped/
                    # unrun index, including started cases with no file yet.
                    r['evaluation_status']=declared
                    r['status']=declared['status']
                elif r['case_id']==failed_eval_id:
                    r['status']='FAILED_CURRENT_EVALUATION_CASE'
                elif summary.exists():
                    r['status']='RECORDED_UNAUDITED_AFTER_COHORT_FAILURE'
                elif partial.exists():
                    r['status']='FAILED_PARTIAL_EVALUATION_CASE'
                elif trace.exists():
                    r['status']='FAILED_CLOSED_TRACE_WITHOUT_CASE_SUMMARY'
                if r['case_id'] in audit_statuses:
                    audit_status=audit_statuses[r['case_id']]
                    r['retained_audit_status']=audit_status
                    r['status']='RECORDED_AUDITED_BEFORE_COHORT_FAILURE' if audit_status['status']=='COMPLETE_CLOSED' else 'RETAINED_AUDIT_'+audit_status['status']
        for r in rows:
            if r['status']=='UNRUN':r['status']='UNRUN_AFTER_COHORT_FAILURE'
        write_json(ledger_path,dict(status='FAILED_NO_COMPLETE_COHORT_ANALYSIS',rows=rows,
            failure_receipt_read_errors=failure_receipt_errors),replace=True)
        write_json(output/'STUDY_FAILED.json',dict(schema='PCI-M2-CONTROLLER-FAILURE-v1',
            status='FAIL_NO_COMPLETE_COHORT_CLAIM',error=repr(exc),elapsed_seconds=time.monotonic()-started,
            completed_training_models=len(locals().get('done',{})),
            case_replacement_or_outcome_selection=False,cloud_resource_state_changed_by_controller=False,
            exact_checkpoints_and_failed_unrun_records_preserved=True,
            failure_receipt_read_errors=failure_receipt_errors,**budget_record))
        raise
    finally:
        stop_watchdog(watchdog_process,watchdog_stream)
        os.environ.pop('PCI_M2_OWNED_PID_REGISTRY',None)


def scheduler_self_test(root,out):
    """Small synthetic process/parser controls only; no study runtime imports."""
    out=Path(out);require(not out.exists(),'NEW_SELF_TEST_DIRECTORY_REQUIRED');out.mkdir(parents=True)
    require('torch' not in sys.modules,'SELF_TEST_MUST_NOT_IMPORT_TORCH')
    p=Path(root)/'design/M2_PROTOCOL.json';plan=make_plan(root,p)
    protocol=read_json(p);broken=copy.deepcopy(protocol);broken['training_cases'].pop()
    try:validate_protocol(broken)
    except GateError:pass
    else:raise AssertionError('Incomplete cohort accepted')
    budget_now=1700000000.
    require(global_budget(started_unix=budget_now)==CEILING and
            global_budget(budget_now+2*CEILING,started_unix=budget_now)==CEILING,
            'DEFAULT_AND_LATER_CUTOFF_KEEP_EXISTING_CEILING')
    require(global_budget(budget_now+123.5,started_unix=budget_now)==123.5,
            'ABSOLUTE_CUTOFF_ONLY_SHORTENS_GLOBAL_BUDGET')
    for cutoff in (budget_now-1,budget_now,float('nan'),float('inf'),float('-inf')):
        try:global_budget(cutoff,started_unix=budget_now)
        except GateError:pass
        else:raise AssertionError('Expired or nonfinite cutoff accepted')
    scale=timing_training_scale(protocol)
    require(scale['full_to_fixture_step_ratio']==8 and scale['complete_training_waves']==15 and
            len(fixture_cases(protocol))==8 and all(c['config']['total_steps']==32768
            for c in fixture_cases(protocol)),'REVISED_ENGINEERING_BUDGET_AND_INTEGER_SCALE')
    try:run_pool([dict(case_id='tiny_cutoff',argv=[sys.executable,'-c','import time;time.sleep(60)'])],
                 root,out/'synthetic_tiny_cutoff',time.monotonic()+.15,workers=1)
    except GateError as exc:require(str(exc.__cause__)=='GLOBAL_WALL_CEILING','TINY_CUTOFF_MUST_REJECT_WORK')
    else:raise AssertionError('Tiny global budget accepted long synthetic work')
    tiny_failure=read_json(out/'synthetic_tiny_cutoff/POOL_FAILED.json')
    require(tiny_failure['all_owned_processes_closed'] and
            tiny_failure['rows'][0]['status']!='COMPLETE_CLOSED','TINY_CUTOFF_OWNED_WORKER_CLEANUP')
    good=[dict(case_id=f'synthetic_{i}',argv=[sys.executable,'-c','import time; time.sleep(0.15); print("SYNTHETIC_CONTROL")']) for i in range(6)]
    passed=run_pool(good,root,out/'synthetic_success',time.monotonic()+10,workers=4)
    require(len(passed['rows'])==6 and passed['peak_parallel_workers']==4 and
            all(r['status']=='COMPLETE_CLOSED' for r in passed['rows']) and
            bool(passed['actual_live_concurrency_intervals']),'SYNTHETIC_SUCCESS_SCHEDULER')
    failing=copy.deepcopy(good);failing[0]['argv']=[sys.executable,'-c','raise SystemExit(7)']
    try:run_pool(failing,root,out/'synthetic_failure',time.monotonic()+10,workers=4)
    except GateError:pass
    else:raise AssertionError('Nonzero worker exit accepted')
    failure=read_json(out/'synthetic_failure/POOL_FAILED.json')
    require(len(failure['rows'])==6 and failure['all_owned_processes_closed'] and
            any(r['status']=='UNRUN_AFTER_COHORT_FAILURE' for r in failure['rows']),'FAILED_UNRUN_ROWS_NOT_PRESERVED')
    # A failed leader can leave a live child in its owned process group.
    # The controller must close that group even after poll() reaped its leader.
    child_pid_file=out/'synthetic_descendant_pid.txt'
    script=('import subprocess,sys;from pathlib import Path;'
        'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]);'
        'Path('+repr(str(child_pid_file))+').write_text(str(p.pid));raise SystemExit(7)')
    try:run_pool([dict(case_id='descendant_leader',argv=[sys.executable,'-c',script])],
                 root,out/'synthetic_descendant',time.monotonic()+10,workers=1)
    except GateError:pass
    else:raise AssertionError('Descendant-leaking failed leader accepted')
    child_pid=int(child_pid_file.read_text())
    try:remaining=process_record(child_pid)
    except (FileNotFoundError,ProcessLookupError):remaining=None
    try:require(remaining is None or remaining['state']=='Z','OWNED_DESCENDANT_SURVIVED_CLEANUP')
    finally:
        if remaining is not None and remaining['state']!='Z':os.kill(child_pid,signal.SIGKILL)
    watchdog_dir=out/'synthetic_watchdog'
    watchdog_script=('import time;from pathlib import Path;'
        'from training.run_study import start_watchdog;'
        'p=Path('+repr(str(watchdog_dir))+');p.mkdir();'
        'w,s=start_watchdog(p,0.35);time.sleep(60)')
    try:run_pool([dict(case_id='watchdog_parent',argv=[sys.executable,'-c',watchdog_script])],
                 root,out/'synthetic_watchdog_logs',time.monotonic()+10,workers=1)
    except GateError:pass
    else:raise AssertionError('Synthetic OS watchdog did not interrupt its parent')
    wait_end=time.monotonic()+3
    while not (watchdog_dir/'WATCHDOG_RECEIPT.json').exists() and time.monotonic()<wait_end:time.sleep(.05)
    watchdog_receipt=read_json(watchdog_dir/'WATCHDOG_RECEIPT.json')
    require(watchdog_receipt['deadline_exceeded'] and not watchdog_receipt['parent_survives'] and
            watchdog_receipt['surviving_owned_groups']==[],'SYNTHETIC_OS_WATCHDOG_FAILURE')
    now=time.time();stamp=dt.datetime.fromtimestamp(now,dt.timezone.utc).isoformat()
    owner=dict(schema='PCI-M2-OPERATOR-OWNED-GPU-OBSERVATION-v1',
        instance=dict(id=OWNED_ID,name=OWNED_NAME,profile_name=OWNED_PROFILE,status='running'),
        observed_at_utc=stamp,ibm_auth_verified_at_utc=stamp,other_gpu_jobs=[],
        scope='operator-observed-not-remote-attestation')
    check_ownership(owner,now=now)
    for mutate in ('wrong_id','stale'):
        invalid=copy.deepcopy(owner)
        if mutate=='wrong_id':invalid['instance']['id']='02c7_775cfbdf-19c5-4111-8c90-15cc737ae7d1'
        else:invalid['observed_at_utc']=dt.datetime.fromtimestamp(now-301,dt.timezone.utc).isoformat()
        try:check_ownership(invalid,now=now)
        except GateError:pass
        else:raise AssertionError('Invalid ownership record accepted:'+mutate)
    require('torch' not in sys.modules,'SELF_TEST_IMPORTED_TORCH')
    result=dict(schema='PCI-M2-CONTROLLER-CPU-SYNTHETIC-CONTROLS-v1',status='PASS',
        controller_sha256=sha(Path(root)/'training/run_study.py'),protocol_sha256=plan['protocol_sha256'],
        exact_plan_counts=[60,1200],four_worker_synthetic_pool=True,
        first_failure_stops_owned_processes_and_retains_unrun_rows=True,
        observed_live_four_process_intervals=passed['actual_live_concurrency_intervals'],
        descendant_group_closed_after_leader_exit=True,hard_OS_watchdog_interrupted_synthetic_parent=True,
        stale_and_coauthor_ownership_records_rejected=True,
        default_global_budget_seconds=CEILING,later_cutoff_cannot_extend_ceiling=True,
        shortened_absolute_budget_verified=True,expired_and_nonfinite_cutoffs_rejected=True,
        positive_tiny_budget_rejects_work_and_closes_owned_processes=True,
        fixture_steps_per_job=FIXTURE_STEPS,training_extrapolation=scale,
        scientific_training_or_evaluation_cases_executed=0,torch_imported=False,
        GPU_or_cloud_or_device_calls=0,
        scope='CPU parser/config/source/ownership-schema and synthetic process scheduling only. The GPU profiler, CPU/CUDA comparison, full-path timing and scientific controller have not been executed by this test.')
    write_json(out/'CONTROLLER_CPU_SELF_TEST.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('plan','check','self-test','gpu-gate','run','_watchdog'))
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--protocol',type=Path,default=ROOT/'design/M2_PROTOCOL.json')
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--ownership-record',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--parent-pid',type=int)
    parser.add_argument('--parent-birth',type=int)
    parser.add_argument('--registry',type=Path)
    parser.add_argument('--deadline-monotonic',type=float)
    parser.add_argument('--not-after-unix',type=float,
        help='Optional absolute work cutoff; can only shorten the eight-hour global ceiling.')
    args=parser.parse_args()
    require(args.not_after_unix is None or args.command in ('gpu-gate','run'),
            'NOT_AFTER_UNIX_ONLY_FOR_EXECUTION_COMMANDS')
    if args.command=='_watchdog':
        require(None not in (args.parent_pid,args.parent_birth,args.registry,args.deadline_monotonic,args.output),'INTERNAL_WATCHDOG_ARGUMENTS')
        watchdog(args.parent_pid,args.parent_birth,args.registry,args.deadline_monotonic,args.output);return
    root=args.project_root.resolve();protocol=args.protocol.resolve()
    if args.command in ('plan','check'):
        result=make_plan(root,protocol,args.protocol_sha256)
        if args.output is not None:write_json(args.output,result)
        print(json.dumps({k:result[k] for k in ('status','scientific_design_frozen','protocol_sha256',
            'source_map_sha256','training_cases','evaluation_cases','max_workers','global_deadline_seconds')},sort_keys=True));return
    require(args.output is not None,'EXPLICIT_NEW_OUTPUT_PATH_REQUIRED')
    if args.command=='self-test':result=scheduler_self_test(root,args.output)
    else:
        require(args.protocol_sha256 is not None and HASH_RE.fullmatch(args.protocol_sha256),'EXPLICIT_FROZEN_PROTOCOL_HASH_REQUIRED')
        require(args.ownership_record is not None,'FRESH_OPERATOR_OWNERSHIP_OBSERVATION_REQUIRED')
        result=controller(root,protocol,args.protocol_sha256,args.ownership_record,args.output,
                          gate_only=args.command=='gpu-gate',not_after_unix=args.not_after_unix)
    print(json.dumps({k:result[k] for k in ('status','scientific_training_or_evaluation_cases_executed') if k in result},sort_keys=True))


if __name__=='__main__':main()
