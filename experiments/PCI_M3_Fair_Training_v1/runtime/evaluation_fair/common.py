"""Exact file identities and exclusive, closed trace publication."""
from __future__ import annotations
import dataclasses
import enum
import gzip
import hashlib
import json
import os
import copy
import ctypes
import stat
from contextlib import contextmanager
from pathlib import Path
import numpy as np


def plain(x):
    if dataclasses.is_dataclass(x): return plain(dataclasses.asdict(x))
    if isinstance(x, enum.Enum): return plain(x.value)
    if isinstance(x, np.ndarray): return plain(x.tolist())
    if isinstance(x, np.generic): return plain(x.item())
    if isinstance(x, dict): return {str(k): plain(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [plain(v) for v in x]
    if x is None or isinstance(x,(str,int,float,bool)): return x
    raise TypeError('Nonportable evidence type: '+str(type(x)))


def canonical(x): return json.dumps(plain(x),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x): return hashlib.sha256(canonical(x)).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def fsync_directory(path):
    fd=os.open(path,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def _anonymous_fd(directory):
    """Require the narrowly qualified Linux anonymous-inode capability."""
    if not hasattr(os,'O_TMPFILE'):raise RuntimeError('Anonymous O_TMPFILE publication is unsupported; no named-file fallback')
    try:return os.open(directory,os.O_TMPFILE|os.O_RDWR|os.O_CLOEXEC,0o600)
    except OSError as e:raise RuntimeError('Anonymous O_TMPFILE publication unavailable on target filesystem') from e


def _link_anonymous(fd,path):
    libc=ctypes.CDLL(None,use_errno=True)
    libc.linkat.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_int]
    libc.linkat.restype=ctypes.c_int
    # Linux's unprivileged O_TMPFILE route resolves this process's held FD.
    # AT_SYMLINK_FOLLOW applies to that procfs reference; the destination is
    # still installed with no replacement and no named growing staging file.
    source=os.fsencode('/proc/self/fd/'+str(fd))
    if libc.linkat(-100,source,-100,os.fsencode(os.path.abspath(path)),0x400)!=0:
        code=ctypes.get_errno()
        if code==17:raise FileExistsError(code,os.strerror(code),str(path))
        raise RuntimeError('Anonymous no-replace linkat publication unavailable: '+os.strerror(code))


def _verify_json_descriptor(held,path,expected,*,baseline=None):
    """Bind canonical bytes and pathname to one stable held regular inode."""
    p=Path(path);before=os.fstat(held.fileno()) if baseline is None else baseline
    def unchanged():
        after=os.fstat(held.fileno());named=p.lstat()
        fields=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if not stat.S_ISREG(after.st_mode) or not stat.S_ISREG(named.st_mode):
            raise RuntimeError('Closed JSON must be a regular nonsymlink file: '+str(p))
        if fields(before)!=fields(after) or fields(named)!=fields(after):
            raise RuntimeError('Closed JSON held inode/path changed during verification: '+str(p))
    unchanged()
    held.seek(0)
    if held.read()!=expected:raise RuntimeError('Closed JSON differs: '+str(p))
    unchanged()
    held.seek(0)
    if held.read()!=expected:raise RuntimeError('Closed JSON changed during canonical readback: '+str(p))
    unchanged()
    # Never return a digest obtained by reopening an unchecked pathname.
    return hashlib.sha256(expected).hexdigest()


def _verify_existing_json(path,expected):
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as held:
        return _verify_json_descriptor(held,path,expected)


def closed_json(path,value,*,exclusive=False):
    """Publish a complete fsynced JSON inode atomically, never replace it."""
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    b=canonical(value)+b'\n'
    if p.exists():
        if exclusive:raise FileExistsError('Immutable publication intent already exists: '+str(p))
        return _verify_existing_json(p,b)
    with os.fdopen(_anonymous_fd(p.parent),'w+b') as held:
        held.write(b);held.flush();os.fsync(held.fileno());held.seek(0)
        stat0=os.fstat(held.fileno())
        if held.read()!=b:raise RuntimeError('Private JSON readback differs')
        stat1=os.fstat(held.fileno())
        if (stat0.st_dev,stat0.st_ino,stat0.st_size,stat0.st_mtime_ns,stat0.st_ctime_ns)!=(stat1.st_dev,stat1.st_ino,stat1.st_size,stat1.st_mtime_ns,stat1.st_ctime_ns):
            raise RuntimeError('Private JSON changed during readback')
        try:_link_anonymous(held.fileno(),p)
        except FileExistsError:
            if exclusive:raise
            with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as winner:
                baseline=os.fstat(winner.fileno())
                result=_verify_json_descriptor(winner,p,b,baseline=baseline)
                fsync_directory(p.parent)
                _verify_json_descriptor(winner,p,b,baseline=baseline)
        else:
            baseline=os.fstat(held.fileno())
            result=_verify_json_descriptor(held,p,b,baseline=baseline)
            fsync_directory(p.parent)
            _verify_json_descriptor(held,p,b,baseline=baseline)
    return result


class ClosedTrace:
    """Anonymous inode, closed readback and no-replace atomic publication.

    The case-level completion JSON is published only after this writer closes.
    A file left after a crash without a matching completion must be quarantined
    by an operator; it is not silently overwritten or reclassified complete.
    """
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        if trace_exists_or_pending(self.path):raise RuntimeError('Existing/partial trace requires explicit recovery')
        self.intent_path=trace_intent_path(self.path)
        intent={'schema':'PCI-ANONYMOUS-TRACE-INTENT-v1','trace_filename':self.path.name,
                'publication':'O_TMPFILE_then_linkat_proc_self_fd_AT_SYMLINK_FOLLOW_no_replace',
                'crash_semantics':'Intent survives; unlinked partial body can be lost on process termination; never auto-resample'}
        self.intent_sha256=closed_json(self.intent_path,intent,exclusive=True)
        fd=_anonymous_fd(self.path.parent);self.raw=os.fdopen(fd,'wb')
        self.held=os.fdopen(os.dup(fd),'rb')
        self.gz=gzip.GzipFile(filename='',fileobj=self.raw,mode='wb',mtime=0)
        self.count=0;self.chain='0'*64;self.closed=False;self.seal=None;self.lifecycle={}
    def append(self,record):
        if self.closed: raise RuntimeError('Trace is already closed')
        self.chain=hashlib.sha256(bytes.fromhex(self.chain)+canonical(record)).hexdigest()
        self.gz.write(canonical({'record':record,'chain':self.chain})+b'\n');self.count+=1
    def close(self):
        if self.closed:
            if self.seal is None:raise RuntimeError('Aborted private trace cannot be published')
            verify_trace_bytes(self.path,self.seal)
            return copy.deepcopy(self.seal)
        self.gz.close();self.raw.flush();os.fsync(self.raw.fileno());self.raw.close();self.closed=True
        # Hold the same unnamed inode through readback and no-replace install.
        # No growing named file can be copied and later resurrected.
        with self.held as held:
            held.seek(0)
            stat0=os.fstat(held.fileno());h=hashlib.sha256()
            for block in iter(lambda:held.read(1<<20),b''):h.update(block)
            candidate={'sha256':h.hexdigest(),'bytes':stat0.st_size,
                       'records':self.count,'last_chain':self.chain,
                       'publication_intent_sha256':self.intent_sha256}
            held.seek(0);structure=verify_trace_structure(self.path,candidate,_held=held)
            stat1=os.fstat(held.fileno())
            if (stat0.st_dev,stat0.st_ino,stat0.st_size,stat0.st_mtime_ns,stat0.st_ctime_ns)!=(stat1.st_dev,stat1.st_ino,stat1.st_size,stat1.st_mtime_ns,stat1.st_ctime_ns):
                raise RuntimeError('Private trace changed while read back')
            staged=stat_identity(stat1);writer_identity=process_identity()
            held_descriptor={'fd':held.fileno(),'target':os.readlink('/proc/self/fd/'+str(held.fileno()))}
            _link_anonymous(held.fileno(),self.path);fsync_directory(self.path.parent)
            finalstat=self.path.stat()
            if (finalstat.st_dev,finalstat.st_ino)!=(stat1.st_dev,stat1.st_ino):
                raise RuntimeError('Published trace differs from held verified inode')
            verify_trace_bytes(self.path,candidate)
        self.seal=candidate;self.closed=True
        self.lifecycle={'publication':'closed_verified_anonymous_inode_then_linkat_proc_self_fd_AT_SYMLINK_FOLLOW_no_replace',
                        'intent_path':str(self.intent_path.relative_to(self.path.parent.parent)),
                        'publication_intent_sha256':self.intent_sha256,
                        'private_identity_at_close':staged,'published_identity':identity(self.path),
                        'writer_identity':writer_identity,'held_read_descriptor':held_descriptor,'structure':structure}
        return copy.deepcopy(self.seal)
    def __enter__(self): return self
    def __exit__(self,kind,value,tb):
        if kind is None:self.close()
        elif not self.closed:
            # Caught failures retain a closed aborted body separately. SIGKILL
            # can lose an unnamed partial body; its permanent intent still
            # prevents sampling a replacement or claiming a complete cohort.
            try:
                self.gz.close();self.raw.flush();os.fsync(self.raw.fileno());self.raw.close()
                aborted=self.path.parent.parent/'trace_aborts'/self.path.name
                aborted.parent.mkdir(exist_ok=True)
                _link_anonymous(self.held.fileno(),aborted);fsync_directory(aborted.parent)
            finally:
                if not self.raw.closed:self.raw.close()
                self.held.close();self.closed=True


def trace_exists_or_pending(path):
    p=Path(path)
    return p.exists() or trace_intent_path(p).exists() or any(p.parent.glob('.'+p.name+'.*.staging'))


def trace_intent_path(path):
    p=Path(path)
    return p.parent.parent/'trace_intents'/(p.name+'.intent.json')


def verify_trace_intent(path,seal):
    p=Path(path);intent=trace_intent_path(p)
    with os.fdopen(os.open(intent,os.O_RDONLY|os.O_NOFOLLOW),'rb') as held:
        before=os.fstat(held.fileno());data=held.read();after=os.fstat(held.fileno());named=intent.lstat()
        if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns) or (named.st_dev,named.st_ino)!=(after.st_dev,after.st_ino):
            raise RuntimeError('Immutable trace publication intent inode changed')
    if hashlib.sha256(data).hexdigest()!=seal.get('publication_intent_sha256'):
        raise RuntimeError('Immutable trace publication intent changed or missing')
    body=json.loads(data)
    if body.get('schema')!='PCI-ANONYMOUS-TRACE-INTENT-v1' or body.get('trace_filename')!=p.name:
        raise RuntimeError('Trace publication intent descriptor differs')


def identity(path):
    return stat_identity(Path(path).stat())


def stat_identity(s):
    return {'device':s.st_dev,'inode':s.st_ino,'links':s.st_nlink,'bytes':s.st_size,
            'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}


def process_identity():
    result={'pid':os.getpid(),'ppid':os.getppid(),'uid':os.getuid()}
    try:
        result['boot_id']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        result['birth_ticks']=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
        result['namespace_pids']=[int(x) for line in Path('/proc/self/status').read_text().splitlines()
                                 if line.startswith('NSpid:') for x in line.split()[1:]]
        result['proc_self_identity']=os.readlink('/proc/self')
        result['open_descriptors']={}
        for fd in sorted(Path('/proc/self/fd').iterdir()):
            try:result['open_descriptors'][fd.name]=os.readlink(fd)
            except FileNotFoundError:pass
    except FileNotFoundError:result['proc_scope']='unavailable_on_this_platform'
    return result


def _held_sha(stream):
    stream.seek(0);h=hashlib.sha256()
    for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    stream.seek(0);return h.hexdigest()


@contextmanager
def _verified_trace_descriptor(path,seal):
    """Hash, consume and recheck one held inode, never hash then reopen."""
    p=Path(path);verify_trace_intent(p,seal);fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as held:
        before=os.fstat(held.fileno())
        def unchanged():
            current=os.fstat(held.fileno());named=p.lstat()
            if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns,current.st_ctime_ns):
                raise RuntimeError('Held trace inode changed during consumption')
            if (named.st_dev,named.st_ino)!=(current.st_dev,current.st_ino) or p.is_symlink():
                raise RuntimeError('Held trace pathname changed identity')
        if before.st_size!=seal['bytes'] or _held_sha(held)!=seal['sha256']:raise RuntimeError('Trace byte identity mismatch')
        unchanged()
        try:yield held
        finally:
            unchanged()
            if _held_sha(held)!=seal['sha256']:raise RuntimeError('Trace bytes changed during consumption')
            unchanged()
            verify_trace_intent(p,seal)


def verify_trace_bytes(path,seal):
    with _verified_trace_descriptor(path,seal):pass


def verify_trace_structure(path,seal,*,_held=None):
    first=None;last=None;steps=0
    for record in (_read_trace_stream(_held,seal) if _held is not None else read_trace(path,seal)):
        if first is None:
            first=record
            if first.get('kind')!='header':raise RuntimeError('Closed trace header required')
        elif record.get('kind')=='step':
            if last is not None and last.get('kind')=='footer':raise RuntimeError('Step after footer')
            if record.get('step')!=steps:raise RuntimeError('Closed trace step sequence mismatch')
            steps+=1
        elif record.get('kind')!='footer' or last is not None and last.get('kind')=='footer':
            raise RuntimeError('Closed trace footer/record type mismatch')
        last=record
    if first is None or last.get('kind')!='footer':raise RuntimeError('Closed trace footer required')
    expected=first.get('case',{}).get('environment_config',{}).get('horizon')
    if expected is not None and steps!=expected:raise RuntimeError('Closed trace horizon incomplete')
    return {'header_and_footer':True,'steps':steps,'records':seal['records'],'last_chain':seal['last_chain']}


def read_trace(path,seal):
    with _verified_trace_descriptor(path,seal) as held:yield from _read_trace_stream(held,seal)


def _read_trace_stream(stream,seal):
    chain='0'*64;count=0
    with gzip.GzipFile(fileobj=stream,mode='rb') as f:
        for line in f:
            item=json.loads(line)
            if set(item)!= {'record','chain'}: raise RuntimeError('Trace row schema')
            chain=hashlib.sha256(bytes.fromhex(chain)+canonical(item['record'])).hexdigest()
            if chain!=item['chain']: raise RuntimeError('Trace chain mismatch')
            count+=1;yield item['record']
    if count!=seal['records'] or chain!=seal['last_chain'] or count<3:
        raise RuntimeError('Trace incomplete/empty/chain mismatch')


def runtime_source_map():
    root=Path(__file__).resolve().parents[1]
    return {p.relative_to(root).as_posix():sha(p) for folder in ('training','evaluation_fair')
            for p in sorted((root/folder).rglob('*.py')) if '__pycache__' not in p.parts}


def environment_state(env):
    """Portable state sufficient to reconcile pinned native replay.

    It includes task streams/queues and both wrapper/native generator states,
    rather than treating a light institutional snapshot as a complete world.
    """
    cfg=plain(env.config)
    common={'config':cfg,'t':int(env.t),'episode':int(env.episode),
            'positions':env.pos.tolist(),'counts':env.counts.tolist(),
            'wait':env.wait.tolist(),'delivered':env.delivered.tolist(),
            'wrapper_rng':plain(env.rng.bit_generator.state)}
    if hasattr(env,'native'):
        n=env.native
        agents=[]
        for a in n.agents:
            agents.append({'id':int(a.id),'x':int(a.x),'y':int(a.y),'direction':int(a.dir.value),
                'prev_x':plain(a.prev_x),'prev_y':plain(a.prev_y),
                'carrying_shelf':int(a.carrying_shelf.id) if a.carrying_shelf else None,
                'has_delivered':bool(a.has_delivered),'requested_action':plain(a.req_action),
                'cancelled_action':plain(a.canceled_action),'message':plain(a.message)})
        common.update(domain='rware',native_reset_seed=int(env.native_seed),
            native_rng=plain(n.np_random.bit_generator.state),native_agents=agents,
            native_shelves=[{'id':int(s.id),'x':int(s.x),'y':int(s.y)} for s in n.shelfs],
            native_request_queue=[int(s.id) for s in n.request_queue],
            native_grid=n.grid.tolist(),native_steps=int(n._cur_steps),
            native_inactive_steps=int(n._cur_inactive_steps))
    else:
        common.update(domain='warehouse',carrying=env.carrying.tolist(),
            goals=env.goals.tolist(),task_index=env.task_index.tolist(),
            task_streams=env.tasks.tolist(),destination_streams=env.destinations.tolist())
    return common
