"""Strict host interface: actual circomlibjs Poseidon and snarkjs proofs only."""
from __future__ import annotations
import copy, json, numbers, os, queue, secrets, shutil, subprocess, threading, time
from pathlib import Path
from .environment import monitor_record
from .crypto_generate import geometry_tag
from .io import atomic_json,sha256

FIELD=21888242871839275222246405745257275088548364400416034343698204186575808495617
ROOT=Path(__file__).resolve().parents[1]

class CryptoError(RuntimeError):pass

class NodeWorker:
    def __init__(self,timeout=600):
        node=shutil.which('node')
        if node is None:raise CryptoError('Node is required; run setup_ubuntu.sh')
        self.timeout=timeout;self.seq=0
        self.proc=subprocess.Popen([node,str(ROOT/'crypto/worker.cjs')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,text=True,bufsize=1,cwd=ROOT/'crypto')
        self.q=queue.Queue();self.errors=[]
        def reader():
            for line in self.proc.stdout:self.q.put(line)
            self.q.put(None)
        def err_reader():
            for line in self.proc.stderr:
                self.errors.append(line)
                if len(self.errors)>100:self.errors.pop(0)
        threading.Thread(target=reader,daemon=True).start();threading.Thread(target=err_reader,daemon=True).start()
        self.request('ping')

    def request(self,op,**data):
        self.seq+=1
        message={'id':self.seq,'op':op,**data}
        try:
            self.proc.stdin.write(json.dumps(message,separators=(',',':'),allow_nan=False)+'\n');self.proc.stdin.flush()
            line=self.q.get(timeout=self.timeout)
        except (BrokenPipeError,queue.Empty) as e:
            self.close();raise CryptoError(f'Node worker failed or exceeded {self.timeout}s: {"".join(self.errors)}') from e
        if line is None:raise CryptoError('Node worker exited: '+''.join(self.errors))
        try:response=json.loads(line)
        except json.JSONDecodeError as e:raise CryptoError(f'Unexpected worker stdout: {line[:500]}') from e
        if response.get('id')!=self.seq:raise CryptoError('Worker request ID mismatch')
        if response.get('ok') is not True:raise CryptoError(response.get('error','Unknown cryptographic failure'))
        return response['result']

    def hash(self,values):return int(self.request('hash',values=[str(integer(v,'hash input')) for v in values])['hash'])
    def commit(self,domain,values,salt=0):
        return int(self.request('commit',domain=str(domain),values=[str(integer(v,'commit input')) for v in values],salt=str(salt))['hash'])
    def close(self):
        if self.proc.poll() is None:
            try:self.proc.stdin.close();self.proc.wait(timeout=3)
            except (OSError,subprocess.TimeoutExpired):self.proc.kill();self.proc.wait()
    def __enter__(self):return self
    def __exit__(self,*_):self.close()


def integer(v,where):
    if isinstance(v,bool):raise ValueError(f'{where}: bool is not an integer encoding')
    if isinstance(v,numbers.Integral):return int(v)
    if isinstance(v,str) and v and v.lstrip('-').isdigit():return int(v)
    raise ValueError(f'{where}: expected exact integer, got {v!r}; do not truncate floats')


def unsigned(v,bits,where):
    v=integer(v,where)
    if not 0<=v<2**bits:raise ValueError(f'{where}={v}: outside unsigned {bits}-bit range')
    return v


def build_witness(record,worker,*,run_tag,nonce=None,salt=None,strict=True):
    rec=copy.deepcopy(record);n=len(rec['before']['positions'])
    if n!=rec['config']['n_agents']:raise ValueError('Agent shape does not match circuit configuration')
    for key in ('counts',):
        if len(rec['before'][key])!=n:raise ValueError('Invalid state dimensions')
    if len(rec['actions'])!=n or len(rec['quotas'])!=n or len(rec['capacities'])!=2:raise ValueError('Invalid action/rule dimensions')
    x=[];y=[];used=[];quotas=[];actions=[]
    for i in range(n):
        if len(rec['before']['positions'][i])!=2 or len(rec['before']['counts'][i])!=2 or len(rec['quotas'][i])!=2:
            raise ValueError(f'Agent {i}: expected two coordinates/resources')
        x.append(unsigned(rec['before']['positions'][i][0],5,f'x[{i}]'))
        y.append(unsigned(rec['before']['positions'][i][1],5,f'y[{i}]'))
        used.append([unsigned(v,16,f'used[{i}][{j}]') for j,v in enumerate(rec['before']['counts'][i])])
        quotas.append([unsigned(v,16,f'quota[{i}][{j}]') for j,v in enumerate(rec['quotas'][i])])
        a=unsigned(rec['actions'][i],3,f'action[{i}]')
        if a>=5:raise ValueError('Action outside 0..4')
        actions.append(a)
    capacities=[unsigned(v,8,f'capacity[{j}]') for j,v in enumerate(rec['capacities'])]
    checked=monitor_record(rec)
    if strict and not checked['compliant']:raise ValueError('Non-compliant transition: '+','.join(checked['errors']))
    episode=unsigned(rec['episode'],64,'episode');step=unsigned(rec['step'],64,'step');run_tag=unsigned(run_tag,128,'run_tag')
    nonce=secrets.randbits(128) if nonce is None else unsigned(nonce,128,'nonce')
    salt=secrets.randbelow(FIELD) if salt is None else integer(salt,'salt')
    if not 0<=salt<FIELD:raise ValueError('Invalid state commitment salt')
    version=1+int(rec['config']['rule_change'] and step>=rec['config']['horizon']//2)
    state_values=[run_tag,episode,step]+[v for i in range(n) for v in (x[i],y[i],*used[i])]
    action_values=[run_tag,episode,step,nonce]+actions
    rule_values=[version,n,geometry_tag(rec['config']['layout']),*capacities]+[v for row in quotas for v in row]
    state_root=worker.commit(1001,state_values,salt)
    action_root=worker.commit(1002,action_values)
    rules_root=worker.commit(1003,rule_values)
    record_hash=worker.hash([state_root,action_root,rules_root,nonce])
    expected={'ok':1,'record_hash':record_hash,'run_tag':run_tag,'episode':episode,'step':step,'nonce':nonce,
              'policy_version':version,'state_root':state_root,'action_root':action_root,'rules_root':rules_root}
    witness={'x':x,'y':y,'used':used,'quotas':quotas,'capacities':capacities,'action':actions,'salt':salt,
             **{k:v for k,v in expected.items() if k not in ('ok','record_hash')}}
    def encode(v):return [encode(a) for a in v] if isinstance(v,list) else str(v)
    return {'input':{k:encode(v) for k,v in witness.items()},'expected':{k:str(v) for k,v in expected.items()},
            'monitor':checked,'record':rec}


def artifact_paths(layout,n,protocol):
    name=f'transition_{layout}_n{n}'
    folder=ROOT/'crypto/build'/name
    manifest=folder/'manifest.json'
    if not manifest.exists():raise CryptoError(f'Build circuit first: {manifest} is absent')
    meta=json.loads(manifest.read_text())
    for name_,expected in meta['files'].items():
        if sha256(folder/name_)!=expected:raise CryptoError(f'Artifact mismatch: {folder/name_}')
    if protocol not in ('groth16','plonk'):raise ValueError(protocol)
    return {'wasm':str(folder/f'{name}_js/{name}.wasm'),'zkey':str(folder/f'{protocol}.zkey'),
            'vkey':str(folder/f'{protocol}.vkey.json'),'sym':str(folder/f'{name}.sym'),
            'manifest':meta,'folder':folder}


def public_vector(expected,paths):
    vk=json.loads(Path(paths['vkey']).read_text());n=int(vk['nPublic'])
    mapping={}
    for line in Path(paths['sym']).read_text().splitlines():
        pieces=line.split(',')
        if len(pieces)!=4:continue
        idx=int(pieces[1]);label=pieces[3]
        if label.startswith('main.') and label[5:] in expected and 1<=idx<=n:
            mapping[idx]=label[5:]
    if set(mapping)!=set(range(1,n+1)) or n!=len(expected):
        raise CryptoError(f'Public ABI mismatch: {mapping}, expected {list(expected)}')
    return [str(expected[mapping[i]]) for i in range(1,n+1)]

class AdmissionVerifier:
    """Expected record is supplied by the trusted simulator/registry, not prover."""
    def __init__(self,worker,paths,protocol):self.worker=worker;self.paths=paths;self.protocol=protocol;self.used=set()
    def accept(self,proof,public,expected):
        intended=public_vector(expected,self.paths)
        if list(map(str,public))!=intended:return False
        token=(expected['run_tag'],expected['episode'],expected['step'])
        if token in self.used:return False
        answer=self.worker.request('verify',protocol=self.protocol,vkey=self.paths['vkey'],proof=proof,publicSignals=public)
        if answer['valid'] is not True:return False
        self.used.add(token);return True


def prove_record(record,worker,paths,protocol,out,run_tag=1,verifier=None):
    start=time.perf_counter();built=build_witness(record,worker,run_tag=run_tag)
    bind_ms=(time.perf_counter()-start)*1000
    result=worker.request('prove',protocol=protocol,input=built['input'],wasm=paths['wasm'],zkey=paths['zkey'],vkey=paths['vkey'])
    expected=public_vector(built['expected'],paths)
    if result['valid'] is not True or result['publicSignals']!=expected:raise CryptoError('Proof or public binding rejected')
    if verifier is not None and not verifier.accept(result['proof'],result['publicSignals'],built['expected']):
        raise CryptoError('Admission rejected (binding, replay, or proof)')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'witness.json',built['input'])
    atomic_json(out/'expected_public.json',built['expected'])
    atomic_json(out/'statement.json',record)
    atomic_json(out/'proof.json',result['proof']);atomic_json(out/'public.json',result['publicSignals'])
    row={k:v for k,v in result.items() if k not in ('proof','publicSignals')}
    row['host_binding_ms']=bind_ms;row['wall_total_ms']=(time.perf_counter()-start)*1000
    row['proof_encoding']='snarkjs_json';row['protocol']=protocol;row['n_agents']=len(record['actions']);row['layout']=record['config']['layout']
    atomic_json(out/'measurement.json',row)
    return row,built,result
