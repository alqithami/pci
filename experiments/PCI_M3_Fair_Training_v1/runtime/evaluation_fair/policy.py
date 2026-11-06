"""Independent inverse-CDF sampling and scalar stage reconciliation."""
from __future__ import annotations
import copy
import numpy as np
import torch
from training.pci_risk_train.environment_adapters import make_oracle
from training.pci_risk_train.shield_policy import ExactJointShield,SequentialSafeShield
from .common import digest


def inverse_cdf(probabilities, uniform):
    p=np.asarray(probabilities,dtype=np.float64)
    if p.ndim!=1 or not np.isfinite(p).all() or np.any(p<0) or p.sum()<=0 or not 0<=uniform<1:
        raise ValueError('Probability vector and uniform in[0,1) required')
    p=p/p.sum();cdf=np.cumsum(p);cdf[-1]=1.
    return int(np.searchsorted(cdf,uniform,side='right'))


def choose(logits,probabilities,oracle,uniforms,config,*,check_torch=True):
    """Every method consumes N+1 retained PCG64 uniforms per joint step.

    Product and sequential policies use agent-indexed first N uniforms.
    Exact conditioned joint policies use the last uniform. Equal random-stream
    consumption is not a claim of identical trajectories under different policies.
    """
    p=np.asarray(probabilities,dtype=np.float64);n,a=p.shape
    u=np.asarray(uniforms,dtype=np.float64)
    if a!=5 or u.shape!=(n+1,) or np.any((u<0)|(u>=1)):raise ValueError('Policy sampling dimensions')
    method=config['method'];details={'sampler':'product_inverse_CDF','uniform_convention':'N agent uniforms plus one joint uniform'}
    if method!='shieldaware':
        selected=np.array([inverse_cdf(p[i],u[i]) for i in range(n)],dtype=np.int64)
        logp=float(np.log(p[np.arange(n),selected]).sum())
    elif config['shield_policy']=='exact':
        if a**n>65536:raise ValueError('Exact evaluation shield exceeds enumeration cap')
        grid=np.indices((a,)*n,dtype=np.int64).reshape(n,-1).T
        valid=np.asarray(oracle.admissible(grid[None],require_physical=True))[0]
        if not valid.any():raise RuntimeError('Empty exact admissible support')
        # Condition in logit space: a raw float32 probability may underflow
        # even though the finite-logit conditional shield has valid support.
        weights=np.asarray(logits,dtype=np.float64)[np.arange(n)[None,:],grid].sum(-1)
        weights[~valid]=-np.inf;maximum=weights[valid].max()
        q=np.exp(weights-maximum);q/=q.sum()
        k=inverse_cdf(q,u[-1]);selected=grid[k];logp=float(np.log(q[k]))
        details.update(sampler='exact_conditioned_joint_inverse_CDF',support=int(valid.sum()),
                       flat_index=k,distribution_sha256=digest(q.tolist()))
        if check_torch:
            ref=make_oracle(oracle.snapshots,domain=config['domain'])
            with torch.no_grad():
                policy=ExactJointShield(torch.from_numpy(np.asarray(logits,dtype=np.float32))[None],ref)
                qt=policy.probs[0].numpy().astype(float)
                if not np.allclose(q,qt,atol=2e-6,rtol=2e-5):raise RuntimeError('Independent exact shield distribution mismatch')
                lp=float(policy.log_prob(torch.from_numpy(selected[None])).item())
            if abs(lp-logp)>1e-4:raise RuntimeError('Independent exact shield likelihood mismatch')
    else:
        selected=np.zeros(n,dtype=np.int64);logp=0.;masks=[];conditionals=[]
        order=[(int(oracle.snapshots[0]['t'])+j)%n for j in range(n)]
        for i in order:
            proposals=np.broadcast_to(selected,(5,n)).copy();proposals[:,i]=np.arange(5)
            mask=np.asarray(oracle.admissible(proposals[None],require_physical=True))[0]
            if not mask.any():raise RuntimeError('Empty sequential completion support')
            local=np.asarray(logits[i],dtype=np.float64).copy();local[~mask]=-np.inf
            q=np.exp(local-local[mask].max());q/=q.sum()
            selected[i]=inverse_cdf(q,u[i]);logp+=float(np.log(q[selected[i]]))
            masks.append(mask.tolist());conditionals.append(q.tolist())
        details.update(sampler='sequential_safe_completion_inverse_CDF',order=order,
                       conditional_masks=masks,conditional_probabilities=conditionals)
        if check_torch:
            ref=make_oracle(oracle.snapshots,domain=config['domain'])
            with torch.no_grad():
                policy=SequentialSafeShield(torch.from_numpy(np.asarray(logits,dtype=np.float32))[None],ref)
                lp=float(policy.log_prob(torch.from_numpy(selected[None])).item())
            if abs(lp-logp)>1e-4:raise RuntimeError('Independent sequential shield likelihood mismatch')
    return selected,logp,details


def scalar_stages(env,actions):
    """Native scalar reference independent of batched institutional code.

    RWARE physical previews use disposable copies of the pinned upstream
    simulator. The scalar entry-quota/capacity loop is implemented here,
    independently of its batched adapter. No preview mutates the live world.
    """
    a=np.asarray(actions,dtype=np.int64);n=env.n
    if not hasattr(env,'native'):
        candidate,_=env.candidates(a)
        physical,_=env.physical_resolve(candidate)
        admitted=env.shield(candidate)
        pa=np.asarray(env.actions_for_positions(physical),dtype=np.int64)
        ia=np.asarray(env.actions_for_positions(admitted),dtype=np.int64)
        commands=ia
    else:
        old=env.pos.copy();h,w=env.native.grid_size
        def preview(commands):
            native=copy.deepcopy(env.native)
            native.step(np.asarray(commands,dtype=np.int64).tolist())
            positions=np.array([(x.x,x.y) for x in native.agents],dtype=np.int64)
            effective=np.asarray(commands,dtype=np.int64).copy()
            effective[(effective==1)&np.all(positions==old,axis=1)]=0
            return positions,effective
        def inside(pos):
            result=np.zeros((n,2),dtype=bool)
            for i,(x,y) in enumerate(pos):
                zone=getattr(env.config,'zone_mode','service_halves')
                if zone=='service_halves':
                    if y>=h-3:result[i,int(x>=w//2)]=True
                elif zone=='transport_bands':
                    result[i,0]=int(y)==h//3;result[i,1]=int(y)==(2*h)//3
                else:raise ValueError('Unannounced RWARE institution region mode')
            return result
        was=inside(old);physical,pa=preview(a);commands=a.copy();admitted=physical;ia=pa
        quota,caps=env.rules()
        for _ in range(2*n+2):
            entry=inside(admitted)&~was
            reject=np.array([any(int(env.counts[i,r])+int(entry[i,r])>int(quota[i,r]) for r in range(2)) for i in range(n)])
            for r in range(2):
                excess=max(int(entry[:,r].sum())-int(caps[r]),0)
                entrants=[i for i in reversed(range(n)) if entry[i,r]]
                reject[entrants[:excess]]=True
            if not reject.any():break
            commands[reject]=0;admitted,ia=preview(commands)
        else:raise RuntimeError('Independent scalar RWARE correction did not converge')
    return {'physical_actions':pa,'admitted_actions':ia,'admitted_native_actions':commands,
            'physical_positions':physical,'admitted_positions':admitted,
            'physical_replacements':int(np.count_nonzero(a!=pa)),
            'institutional_replacements':int(np.count_nonzero(pa!=ia)),
            'total_replacements':int(np.count_nonzero(a!=ia))}
