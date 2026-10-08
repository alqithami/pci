"""Independent RWARE 2.0.0 dynamics with an explicitly documented extension.

The upstream step/reset are called unchanged. Public task/map features and dense
training-only reward shaping are additions, shared across all compared methods.
RWARE institutions limit simultaneous *entries*, not standing occupancy, into
the two service-zone halves, plus per-agent 64-step entry quotas. RWARE is NOT
cryptographically certified by this study's custom-warehouse circuit.
"""
from dataclasses import dataclass,replace
import copy
import numpy as np
from pci_bench.environment import EnvConfig,Warehouse,D

@dataclass(frozen=True)
class ExternalConfig:
    layout:str='tiny'
    n_agents:int=4
    horizon:int=256
    quota_window:int=64
    quota:int=3
    capacity:int=2
    rule_change:bool=False

class RwareInstitution:
    def __init__(self,config,seed):
        from rware.warehouse import Warehouse as Native,RewardType
        self.config=ExternalConfig(**config) if isinstance(config,dict) else config
        if self.config.layout not in ('tiny','small'):raise ValueError('RWARE layout must be tiny or small')
        self.rng=np.random.default_rng(seed);self.seed=int(seed);self.episode=-1
        self.native=Native(shelf_columns=3,column_height=8,shelf_rows=1 if self.config.layout=='tiny' else 2,
            n_agents=self.n,msg_bits=0,sensor_range=1,request_queue_size=2*self.n,
            max_inactivity_steps=None,max_steps=self.config.horizon,reward_type=RewardType.INDIVIDUAL)
        self.reset()
    @property
    def n(self):return self.config.n_agents
    @property
    def pos(self):return np.array([(a.x,a.y) for a in self.native.agents],dtype=np.int64)
    def reset(self):
        self.episode+=1;self.t=0
        self.native.reset(seed=int(self.rng.integers(0,2**31-1)))
        self.counts=np.zeros((self.n,2),dtype=np.int64)
        self.delivered=np.zeros(self.n,dtype=np.int64);self.wait=np.zeros(self.n,dtype=np.int64)
        self.last=None
        return self.features()
    def rules(self):
        quota=max(1,self.config.quota-int(self.config.rule_change and self.t>=self.config.horizon//2))
        return np.full((self.n,2),quota),np.full(2,self.config.capacity)
    def inside(self,pos):
        p=np.asarray(pos);h,w=self.native.grid_size
        service=p[:,1]>=h-3
        return np.stack((service&(p[:,0]<w//2),service&(p[:,0]>=w//2)),-1)
    def _target(self,agent):
        requested={s.id for s in self.native.request_queue}
        if agent.carrying_shelf is not None:
            if agent.carrying_shelf.id in requested:candidates=[tuple(g) for g in self.native.goals]
            else:
                h,w=self.native.grid_size
                candidates=[(x,y) for y in range(h) for x in range(w)
                    if not self.native._is_highway(x,y) and
                    (self.native.grid[1,y,x]==0 or self.native.grid[1,y,x]==agent.carrying_shelf.id)]
        else:candidates=[(s.x,s.y) for s in self.native.request_queue]
        if not candidates:raise RuntimeError('No native task/return destination')
        return min(candidates,key=lambda p:(abs(p[0]-agent.x)+abs(p[1]-agent.y),p[1],p[0]))
    @staticmethod
    def _potential(pos,direction,target):
        dx,dy=target[0]-pos[0],target[1]-pos[1]
        delta={0:(0,-1),1:(0,1),2:(-1,0),3:(1,0)}[int(direction)]
        aligned=(delta[0]*dx+delta[1]*dy)>0
        return abs(dx)+abs(dy)+(.25 if (dx or dy) and not aligned else 0.)
    def _candidate(self,agent,action):
        # Pure read-only kinematic prediction; upstream step remains authoritative.
        wrap=[0,3,1,2];d=agent.dir.value;x,y=agent.x,agent.y
        if action==2:d=wrap[(wrap.index(d)-1)%4]
        if action==3:d=wrap[(wrap.index(d)+1)%4]
        dx,dy={0:(0,-1),1:(0,1),2:(-1,0),3:(1,0)}[agent.dir.value]
        xx=x+dx if action==1 else x;yy=y+dy if action==1 else y
        h,w=self.native.grid_size;bad=not(0<=xx<w and 0<=yy<h)
        xx=max(0,min(w-1,xx));yy=max(0,min(h-1,yy))
        if action==1 and agent.carrying_shelf is not None and (xx,yy)!=(x,y):
            sid=self.native.grid[1,yy,xx];aid=self.native.grid[0,yy,xx]
            if sid and not(aid and self.native.agents[aid-1].carrying_shelf is not None):bad=True
        if bad:xx,yy=x,y
        return (xx,yy),d,bad
    def features(self):
        f=np.zeros((self.n,5,D),dtype=np.float32);q,cap=self.rules();old=self.pos;now=self.inside(old)
        occupancy=now.sum(0);h,w=self.native.grid_size;requested={s.id for s in self.native.request_queue}
        for i,a in enumerate(self.native.agents):
            goal=self._target(a);loaded=a.carrying_shelf is not None
            d0=self._potential((a.x,a.y),a.dir.value,goal)
            for ac in range(5):
                nxt,d,bad=self._candidate(a,ac);entered=self.inside([nxt])[0]&~now[i]
                near=any(j!=i and tuple(p)==nxt for j,p in enumerate(old))
                useful_toggle=ac==4 and ((not loaded and self.native.grid[1,a.y,a.x] in requested)
                    or (loaded and a.carrying_shelf.id not in requested and not self.native._is_highway(a.x,a.y)))
                progress=d0-self._potential(nxt,d,goal)+float(useful_toggle)
                deliver=loaded and a.carrying_shelf.id in requested and nxt in self.native.goals
                f[i,ac]=[progress,float(bad),float(near),nxt[0]-a.x,nxt[1]-a.y,*entered.astype(float),
                    *((self.counts[i]+entered>q[i]).astype(float)),float(deliver),float(loaded),
                    (self.t%self.config.quota_window)/self.config.quota_window,
                    *np.maximum(q[i]-self.counts[i],0)/q[i],min(self.wait[i]/self.config.horizon,1),
                    (self.delivered[i]+1)/(self.delivered.sum()+self.n),a.x/(w-1),a.y/(h-1),
                    goal[0]/(w-1),goal[1]/(h-1),d0/(h+w),*occupancy/np.maximum(cap,1)]
        if not np.isfinite(f).all():raise RuntimeError('Invalid RWARE feature')
        return f
    def _predicate(self,old,new):
        entry=self.inside(new)&~self.inside(old);q,cap=self.rules()
        qbad=(self.counts+entry>q).any(-1);excess=np.maximum(entry.sum(0)-cap,0)
        return entry,qbad,excess
    def transition(self,actions,*,enforce=False,certifier=None):
        if certifier is not None:raise ValueError('No SNARK relation for RWARE is claimed')
        a=np.asarray(actions)
        if a.shape!=(self.n,) or a.dtype.kind not in 'iu' or ((a<0)|(a>4)).any():raise ValueError('Invalid native action')
        if self.t>=self.config.horizon:raise RuntimeError('reset required')
        old=self.pos;before_count=self.counts.copy();targets=[self._target(x) for x in self.native.agents]
        oldpotential=np.array([self._potential(p,x.dir.value,g) for p,x,g in zip(old,self.native.agents,targets)])
        before_carry=[x.carrying_shelf.id if x.carrying_shelf else None for x in self.native.agents]
        requested={s.id for s in self.native.request_queue};old_dirs=[x.dir.value for x in self.native.agents]
        walls=np.array([self._candidate(x,int(ac))[2] for x,ac in zip(self.native.agents,a)])
        if enforce:
            preview=copy.deepcopy(self.native);native_result=preview.step(a.tolist())
        else:
            preview=self.native;native_result=preview.step(a.tolist())
        raw_pos=np.array([(x.x,x.y) for x in preview.agents]);entry,qbad,excess=self._predicate(old,raw_pos)
        phys_rejected=np.array([int(ac)==1 and (old[i]==raw_pos[i]).all() and not walls[i] for i,ac in enumerate(a)])
        raw_ok=not(walls.any() or phys_rejected.any() or qbad.any() or excess.any())
        final=a.copy()
        if enforce:
            if (self.counts>self.rules()[0]).any():raise RuntimeError('Invalid admitted quota history')
            for _ in range(2*self.n+2):
                reject=qbad.copy()
                for r in range(2):
                    if excess[r]>0:
                        entrants=np.flatnonzero(entry[:,r]);reject[entrants[-int(excess[r]):]]=True
                if not reject.any():break
                final[reject]=0
                preview=copy.deepcopy(self.native);native_result=preview.step(final.tolist())
                raw2=np.array([(x.x,x.y) for x in preview.agents]);entry,qbad,excess=self._predicate(old,raw2)
            if qbad.any() or excess.any():raise RuntimeError('RWARE admission did not converge')
            # Native dynamics are run on the original object, not replaced by a simulator mock.
            expected_pos=np.array([(x.x,x.y) for x in preview.agents])
            native_result=self.native.step(final.tolist())
            if not np.array_equal(self.pos,expected_pos):raise AssertionError('Preview and execution differ')
        new=self.pos;actual_entry,actual_qbad,actual_excess=self._predicate(old,new)
        # Cost uses the original proposed action's physically realized entry effects.
        raw_entry,raw_qbad,raw_excess=self._predicate(old,raw_pos)
        costs=walls.astype(float)+phys_rejected.astype(float)+raw_qbad.astype(float)+float(raw_excess.sum())/self.n
        native_rewards=np.asarray(native_result[1],dtype=float)
        if not np.allclose(native_rewards,np.rint(native_rewards)) or (native_rewards<0).any():raise AssertionError('Unexpected native INDIVIDUAL reward')
        delivered=np.rint(native_rewards).astype(np.int64)
        after_carry=[x.carrying_shelf.id if x.carrying_shelf else None for x in self.native.agents]
        pickups=np.array([b is None and c in requested for b,c in zip(before_carry,after_carry)])
        drops=np.array([b is not None and b not in requested and c is None for b,c in zip(before_carry,after_carry)])
        newpotential=np.array([self._potential(p,x.dir.value,g) for p,x,g in zip(new,self.native.agents,targets)])
        active=(old!=new).any(-1)|np.array([d!=x.dir.value for d,x in zip(old_dirs,self.native.agents)])
        reward=native_rewards+.2*pickups+.05*drops+.05*(oldpotential-newpotential)-.01*active
        self.counts+=actual_entry.astype(np.int64);self.delivered+=delivered;self.wait+=1;self.wait[delivered>0]=0
        actual_unique=len({tuple(p) for p in new})==self.n
        if not actual_unique:raise AssertionError('Native collision invariant failed')
        metrics={'delivered':int(delivered.sum()),'pickups':int(pickups.sum()),'task_reward':float(reward.sum()),
            'proposed_compliant':int(raw_ok),'executed_compliant':int(not actual_qbad.any() and not actual_excess.any()),
            'wall_attempts':int(walls.sum()),'collision_attempts':int(phys_rejected.sum()),
            'quota_violations':int(raw_qbad.sum()),'capacity_excess':int(raw_excess.sum()),
            'interventions':int(np.sum(a!=final)),'actual_vertex_collisions':0,
            'move_count':int((old!=new).any(-1).sum()),'max_wait':int(self.wait.max())}
        self.last={'step':self.t,'positions_before':old.tolist(),'positions_after':new.tolist(),
            'counts_before':before_count.tolist(),'counts_after':self.counts.tolist(),'actions':a.tolist(),
            'admitted_actions':final.tolist(),'native_rewards':native_rewards.tolist(),'metrics':metrics}
        self.t+=1
        if self.t%self.config.quota_window==0:self.counts.fill(0)
        return self.features(),reward.astype(np.float32),costs.astype(np.float32),self.t==self.config.horizon,metrics
    def state_dict(self):return copy.deepcopy(self.__dict__)
    def load_state_dict(self,state):self.__dict__.update(copy.deepcopy(state))

def make_env(config,seed):
    if config['domain']=='custom':return Warehouse(EnvConfig(**config['env']),int(seed))
    if config['domain']=='rware':return RwareInstitution(config['env'],int(seed))
    raise ValueError(config['domain'])
