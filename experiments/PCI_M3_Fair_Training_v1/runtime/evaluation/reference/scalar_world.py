"""Faithful private-world adapter with an independent scalar transition oracle.

The SNARK profile certifies a restricted rule ledger. These full task dynamics,
queue/frame checks and persisted RNG are private-executor obligations, not
SNARK-certified task progress or sensor/physical-world attestation.
"""
from __future__ import annotations
import copy, hashlib, importlib.util, json, math, numbers, sys
from collections import deque
from pathlib import Path
import numpy as np

REFERENCE_SHA256='cc2ba5ad437b517f9aa3ab96119ce3fa3dba2c51098a7ff2d77ef6977f3e059c'
MOVES=((0,0),(1,0),(-1,0),(0,-1),(0,1))
CFG={'layout','n_agents','horizon','quota_window','quota','capacity','rule_change'}
STATE={'config','episode','t','pos','counts','carrying','delivered','wait','tasks','destinations','task_index','goals','last','rng_state'}
PICKUPS=((2,2),(2,4),(2,8),(2,10));DEPOTS=((10,3),(10,9))

class WorldError(ValueError): pass

def canonical(value):
    """Reject non-JSON and nonfinite values rather than coercing arbitrary objects."""
    def check(x):
        if x is None or type(x) in (str,bool,int):return
        if type(x) is float:
            if not math.isfinite(x):raise WorldError('NONFINITE_JSON')
            return
        if type(x) is list:
            for y in x:check(y)
            return
        if type(x) is dict and all(type(k) is str for k in x):
            for y in x.values():check(y)
            return
        raise WorldError('NONCANONICAL_JSON_TYPE')
    check(value)
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)

def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()

def _uint(v,bits,label):
    if type(v) is not int or not 0<=v<2**bits:raise WorldError('INVALID_INTEGER:'+label)
    return v

def config(cfg):
    if type(cfg) is not dict or set(cfg)!=CFG:raise WorldError('CONFIG_SCHEMA')
    if cfg['layout'] not in ('open','twodoors','staggered') or type(cfg['layout']) is not str:raise WorldError('LAYOUT')
    for k in CFG-{'layout','rule_change'}:_uint(cfg[k],32,k)
    if type(cfg['rule_change']) is not bool:raise WorldError('RULE_CHANGE_TYPE')
    if not (1<=cfg['n_agents']<=8 and 2<=cfg['horizon']<=4096 and 1<=cfg['quota_window']<=cfg['horizon'] and 1<=cfg['quota']<2**16 and 1<=cfg['capacity']<=8):raise WorldError('CONFIG_RANGE')
    return copy.deepcopy(cfg)

def geometry(layout):
    walls={(x,y) for x in range(13) for y in range(13) if x in (0,12) or y in (0,12)}
    if layout=='twodoors':
        walls|={(6,y) for y in range(1,12) if y not in (3,9)};regions=((5,7,3,3),(5,7,9,9))
    elif layout=='open':
        walls|={(x,y) for y in (4,8) for x in (3,4,8,9)};regions=((5,7,3,3),(5,7,9,9))
    elif layout=='staggered':
        walls|={(4,y) for y in range(1,12) if y not in (3,9)}
        walls|={(8,y) for y in range(1,12) if y not in (5,9)};regions=((3,5,3,3),(7,9,5,5))
    else:raise WorldError('LAYOUT')
    return walls,regions

def _inside(p,r):return r[0]<=p[0]<=r[1] and r[2]<=p[1]<=r[3]

class _DeepJSONFallback(Exception):
    """Use the original checker/copier for aliased or deep container graphs."""

def _clone_canonical_json(value):
    """Validate strict JSON and copy alias-free, bounded-depth containers.

    This matches canonical's accepted primitive types and finite-number policy.
    Any repeated container, including a cycle, uses the original checker/copier.
    This preserves its aliasing, traversal and recursion behavior unchanged.
    """
    # deepcopy uses more Python frames per container than this fast path.
    # Under a deep caller stack, defer to it before changing its resource limit.
    frame=sys._getframe()
    caller_depth=0
    while frame is not None:
        caller_depth+=1
        frame=frame.f_back
    if sys.getrecursionlimit()-caller_depth<512:
        raise _DeepJSONFallback
    memo = {}
    active = set()
    large_integers = []
    def clone(x):
        kind = type(x)
        if x is None or kind in (str, bool):
            return x
        if kind is int:
            # json.dumps also enforces Python's configured integer digit limit.
            # Normal bounded state integers never need decimal conversion.
            if x.bit_length() > 2000:
                large_integers.append(x)
            return x
        if kind is float:
            if not math.isfinite(x):
                raise WorldError('NONFINITE_JSON')
            return x
        if kind not in (list, dict) or (kind is dict and any(type(k) is not str for k in x)):
            raise WorldError('NONCANONICAL_JSON_TYPE')
        key = id(x)
        if key in memo:
            raise _DeepJSONFallback
        if len(active) >= 128:
            raise _DeepJSONFallback
        active.add(key)
        if kind is list:
            out = []
            memo[key] = out
            for item in x:
                out.append(clone(item))
        else:
            out = {}
            memo[key] = out
            for name, item in x.items():
                out[name] = clone(item)
        active.remove(key)
        return out
    result = clone(value)
    # canonical first validates every primitive, then invokes json.dumps.  Keep
    # its error precedence when a tree has both a huge int and an invalid value.
    for integer in large_integers:
        str(integer)
    return result

def validate_snapshot(s):
    if type(s) is not dict or set(s)!=STATE:raise WorldError('WORLD_SCHEMA')
    c=config(s['config']);n=c['n_agents'];h=c['horizon']
    _uint(s['episode'],64,'episode');_uint(s['t'],32,'t')
    if s['t']>h:raise WorldError('HORIZON')
    def matrix(name,cols,bits):
        v=s[name]
        if type(v) is not list or len(v)!=n or any(type(r) is not list or len(r)!=cols for r in v):raise WorldError('WORLD_SHAPE:'+name)
        for i,r in enumerate(v):
            for j,x in enumerate(r):_uint(x,bits,f'{name}:{i}:{j}')
    matrix('pos',2,5);matrix('counts',2,16);matrix('tasks',h+2,8);matrix('destinations',h+2,8);matrix('goals',2,5)
    for name in ('delivered','wait','task_index'):
        v=s[name]
        if type(v) is not list or len(v)!=n:raise WorldError('WORLD_SHAPE:'+name)
        for i,x in enumerate(v):_uint(x,32,f'{name}:{i}')
    if type(s['carrying']) is not list or len(s['carrying'])!=n or any(type(x) is not bool for x in s['carrying']):raise WorldError('CARRYING_TYPE')
    walls,_=geometry(c['layout'])
    if len({tuple(p) for p in s['pos']})!=n or any(not (0<=p[0]<13 and 0<=p[1]<13) or tuple(p) in walls for p in s['pos']):raise WorldError('WORLD_POSITIONS')
    if any(x>=4 for r in s['tasks'] for x in r) or any(x>=2 for r in s['destinations'] for x in r):raise WorldError('TASK_RANGE')
    if any(x>h for x in s['task_index']):raise WorldError('TASK_INDEX_RANGE')
    for i in range(n):
        goal=DEPOTS[s['destinations'][i][s['task_index'][i]]] if s['carrying'][i] else PICKUPS[s['tasks'][i][s['task_index'][i]]]
        if list(goal)!=s['goals'][i]:raise WorldError('GOAL_FRAME')
        if s['delivered'][i]!=s['task_index'][i]:raise WorldError('DELIVERY_INDEX_FRAME')
    rng=s['rng_state']
    if type(rng) is not dict or set(rng)!={'bit_generator','state','has_uint32','uinteger'} or rng['bit_generator']!='PCG64':raise WorldError('RNG_SCHEMA')
    if type(rng['state']) is not dict or set(rng['state'])!={'state','inc'}:raise WorldError('RNG_STATE')
    for k in ('state','inc'):_uint(rng['state'][k],128,'rng:'+k)
    _uint(rng['has_uint32'],1,'has_uint32');_uint(rng['uinteger'],32,'uinteger')
    if s['last'] is not None and type(s['last']) is not dict:raise WorldError('LAST_SCHEMA')
    try:
        return _clone_canonical_json(s)
    except _DeepJSONFallback:
        # Preserve the original recursion/resource behavior on deep JSON.
        canonical(s)
        return copy.deepcopy(s)

def _physical(old,raw):
    nxt=copy.deepcopy(raw);rejected=[False]*len(old)
    for _ in range(len(old)+1):
        bad=[False]*len(old)
        for i in range(len(old)):
            for j in range(i):
                if nxt[i]==nxt[j] or (nxt[i]==old[j] and nxt[j]==old[i]):
                    bad[i]|=nxt[i]!=old[i];bad[j]|=nxt[j]!=old[j]
        if not any(bad):break
        for i,b in enumerate(bad):
            if b:nxt[i]=old[i][:];rejected[i]=True
    if len({tuple(p) for p in nxt})!=len(old):raise WorldError('PHYSICAL_RESOLUTION')
    return nxt,rejected

def _rule_metrics(s,nxt):
    c=s['config'];_,rs=geometry(c['layout']);q=c['quota']-int(c['rule_change'] and s['t']>=c['horizon']//2);q=max(1,q)
    entries=[[int(_inside(nxt[i],r) and not _inside(s['pos'][i],r)) for r in rs] for i in range(c['n_agents'])]
    bad=[any(s['counts'][i][r]+entries[i][r]>q for r in range(2)) for i in range(c['n_agents'])]
    excess=[max(sum(int(_inside(p,r)) for p in nxt)-c['capacity'],0) for r in rs]
    return entries,bad,excess,q

def admitted_actions(s,proposal):
    s=validate_snapshot(s);n=s['config']['n_agents'];walls,rs=geometry(s['config']['layout'])
    if type(proposal) is not list or len(proposal)!=n or any(type(a) is not int or not 0<=a<5 for a in proposal):raise WorldError('ACTION_SCHEMA')
    raw=[];wall=[]
    for p,a in zip(s['pos'],proposal):
        d=MOVES[a];t=[p[0]+d[0],p[1]+d[1]];b=tuple(t) in walls
        wall.append(b);raw.append(p[:] if b else t)
    physical,collision=_physical(s['pos'],raw);_,raw_bad,raw_excess,q=_rule_metrics(s,physical)
    if any(v>q for r in s['counts'] for v in r):raise WorldError('PREEXISTING_OVERBUDGET')
    nxt=copy.deepcopy(physical)
    for _ in range(2*n+2):
        entries,qbad,excess,_=_rule_metrics(s,nxt);reject=qbad[:]
        for r in range(2):
            entrants=[i for i in range(n) if entries[i][r]]
            order=sorted(entrants,key=lambda i:(s['wait'][i],-s['delivered'][i],i))
            for i in order[:excess[r]]:reject[i]=True
        if not any(reject):
            if any(excess):raise WorldError('INFEASIBLE_CAPACITY')
            break
        for i,b in enumerate(reject):
            if b:nxt[i]=s['pos'][i][:]
        nxt,_=_physical(s['pos'],nxt)
    else:raise WorldError('CORRECTION_DID_NOT_CONVERGE')
    acts=[MOVES.index((t[0]-p[0],t[1]-p[1])) for p,t in zip(s['pos'],nxt)]
    return acts,nxt,wall,collision,raw_bad,raw_excess

def _distance(start,goal,walls):
    q=deque([(tuple(start),0)]);seen={tuple(start)}
    while q:
        p,d=q.popleft()
        if p==tuple(goal):return d
        for dx,dy in MOVES[1:]:
            t=(p[0]+dx,p[1]+dy)
            if 0<=t[0]<13 and 0<=t[1]<13 and t not in walls and t not in seen:seen.add(t);q.append((t,d+1))
    raise WorldError('UNREACHABLE_GOAL')

def scalar_effect(before,proposal):
    """Independent implementation: never calls Warehouse.transition or its helpers."""
    s=validate_snapshot(before);c=s['config'];n=c['n_agents']
    if s['t']>=c['horizon']:raise WorldError('RESET_REQUIRED')
    acts,nxt,wall,collision,qbad,excess=admitted_actions(s,proposal);walls,rs=geometry(c['layout'])
    entries,_,_,quota=_rule_metrics(s,nxt)
    try:
        out=_clone_canonical_json(s)
    except _DeepJSONFallback:
        out=copy.deepcopy(s)
    out['pos']=nxt
    out['counts']=[[s['counts'][i][r]+entries[i][r] for r in range(2)] for i in range(n)]
    reward=[];delivered=pickups=0;moved=[]
    for i in range(n):
        move=nxt[i]!=s['pos'][i];moved.append(move);reach=nxt[i]==s['goals'][i]
        de=reach and s['carrying'][i];pi=reach and not s['carrying'][i];delivered+=int(de);pickups+=int(pi)
        out['delivered'][i]+=int(de);out['wait'][i]+=1
        if de:out['wait'][i]=0;out['task_index'][i]+=1;out['carrying'][i]=False;out['goals'][i]=list(PICKUPS[s['tasks'][i][out['task_index'][i]]])
        elif pi:out['carrying'][i]=True;out['goals'][i]=list(DEPOTS[s['destinations'][i][s['task_index'][i]]])
        r=float(de)+0.2*int(pi)+0.05*(_distance(s['pos'][i],s['goals'][i],walls)-_distance(nxt[i],s['goals'][i],walls))-0.01*int(move)
        reward.append(r)
    rec={'config':copy.deepcopy(c),'episode':s['episode'],'step':s['t'],'before':{'positions':copy.deepcopy(s['pos']),'counts':copy.deepcopy(s['counts'])},'actions':acts,'quotas':[[quota,quota] for _ in range(n)],'capacities':[c['capacity'],c['capacity']],'after_positions':copy.deepcopy(nxt)}
    out['last']=rec;out['t']+=1
    if out['t']%c['quota_window']==0:out['counts']=[[0,0] for _ in range(n)]
    costs=[float(wall[i])+float(collision[i])+float(qbad[i])+float(sum(excess))/n for i in range(n)]
    metrics={'delivered':delivered,'pickups':pickups,'task_reward':float(np.asarray(reward,dtype=np.float64).sum()),'proposed_compliant':int(not(any(wall) or any(collision) or any(qbad) or any(excess))),'executed_compliant':1,'wall_attempts':sum(wall),'collision_attempts':sum(collision),'quota_violations':sum(qbad),'capacity_excess':sum(excess),'interventions':sum(a!=b for a,b in zip(proposal,acts)),'actual_vertex_collisions':n-len({tuple(p) for p in nxt}),'move_count':sum(moved),'max_wait':max(out['wait'])}
    result={'reward':[float(np.float32(x)) for x in reward],'cost':[float(np.float32(x)) for x in costs],'done':out['t']==c['horizon'],'metrics':metrics}
    validate_snapshot(out)
    return {'before':s,'proposal':proposal[:],'admitted_actions':acts,'after':out,'result':result,'record':rec}

class WorldAdapter:
    def __init__(self,reference_source):
        p=Path(reference_source);data=p.read_bytes()
        if hashlib.sha256(data).hexdigest()!=REFERENCE_SHA256:raise WorldError('REFERENCE_SOURCE_HASH')
        self.source_sha256=REFERENCE_SHA256
        name='_pci_successor_reference_'+REFERENCE_SHA256[:12]
        if name in sys.modules:self.mod=sys.modules[name]
        else:
            spec=importlib.util.spec_from_file_location(name,p);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);self.mod=mod
    def snapshot(self,env):
        def normalize(x):
            if isinstance(x,np.ndarray):return normalize(x.tolist())
            if isinstance(x,np.generic):return normalize(x.item())
            if hasattr(x,'__dataclass_fields__'):
                from dataclasses import asdict
                return normalize(asdict(x))
            if isinstance(x,tuple):return [normalize(v) for v in x]
            if isinstance(x,list):return [normalize(v) for v in x]
            if isinstance(x,dict):return {str(k):normalize(v) for k,v in x.items()}
            return x
        return validate_snapshot(normalize(env.state_dict()))
    def restore(self,snapshot):
        s=validate_snapshot(snapshot);env=self.mod.Warehouse(self.mod.EnvConfig(**s['config']),seed=0)
        state=copy.deepcopy(s);state['config']=self.mod.EnvConfig(**s['config'])
        for k in ('pos','counts','delivered','wait','tasks','destinations','task_index','goals'):state[k]=np.asarray(state[k],dtype=np.int64)
        state['carrying']=np.asarray(state['carrying'],dtype=bool);env.load_state_dict(state)
        if self.snapshot(env)!=s:raise WorldError('RESTORE_ROUNDTRIP')
        return env
    def initial(self,cfg,seed):
        cfg=config(cfg);_uint(seed,128,'seed');env=self.mod.Warehouse(self.mod.EnvConfig(**cfg),seed)
        return self.snapshot(env)
    def propose(self,snapshot,proposal):
        before=validate_snapshot(snapshot);env=self.restore(before)
        _,reward,cost,done,metrics=env.transition(proposal,enforce=True)
        out={'before':before,'proposal':list(proposal),'admitted_actions':env.last['actions'],'after':self.snapshot(env),'result':{'reward':[float(v) for v in reward],'cost':[float(v) for v in cost],'done':bool(done),'metrics':metrics},'record':copy.deepcopy(env.last)}
        expected=scalar_effect(before,list(proposal))
        if canonical(out)!=canonical(expected):raise WorldError('NATIVE_SCALAR_EFFECT_DISAGREEMENT')
        return out
    def validate_effect(self,payload,expected_before=None):
        if type(payload) is not dict or set(payload)!={'before','proposal','admitted_actions','after','result','record'}:raise WorldError('PRIVATE_EFFECT_SCHEMA')
        if expected_before is not None and canonical(payload['before'])!=canonical(validate_snapshot(expected_before)):raise WorldError('PRIVATE_PREDECESSOR_MISMATCH')
        expected=scalar_effect(payload['before'],payload['proposal'])
        if canonical(payload)!=canonical(expected):raise WorldError('PRIVATE_EFFECT_MISMATCH')
        # Also require agreement with the pinned actual native transition implementation.
        native=self.propose(payload['before'],payload['proposal'])
        if canonical(native)!=canonical(expected):raise WorldError('NATIVE_EFFECT_MISMATCH')
        return expected
