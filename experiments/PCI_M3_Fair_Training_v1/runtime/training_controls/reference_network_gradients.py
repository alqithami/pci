"""Analytic gradients of the fixed warehouse actor's float64 smooth extension.

The stored float32 weights and numeric inputs are converted to float64 before
arithmetic. This is the same three-head/two-Tanh architecture, not an exact
derivative of rounded float32 software or of a policy-induced state distribution.
Only head weights/biases are differentiated. Masks and critic are fixed/excluded.
No policy sampling, RNG, world construction, Torch, optimizer or training occurs.
"""
from __future__ import annotations
import math
from typing import Mapping
import numpy as np

D, ACTIONS, HEADS, HIDDEN = 23, 5, 3, 64
VIEW_INDICES = (
 (0,1,2,3,4,5,6,10,11,16,17,18,19,20,21,22),
 (0,1,2,3,4,9,10,11,16,17,18,19,20),
 (0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20),
)
OVERLAPS = ((0,1,(3,4,2)), (0,2,(5,6)), (1,2,(0,9)))
SHAPES = {f'heads.{h}.{layer}.{kind}': shape
          for h in range(HEADS)
          for layer,weight,bias in ((0,(HIDDEN,2*D),(HIDDEN,)),
                                     (2,(HIDDEN,HIDDEN),(HIDDEN,)),
                                     (4,(1,HIDDEN),(1,)))
          for kind,shape in (('weight',weight),('bias',bias))}
PARAMETER_ORDER = tuple(SHAPES)
NORM_THRESHOLD = 1e-12

class GradientError(ValueError):
    pass

def _require(condition, label):
    if not condition:
        raise GradientError(label)

def masks():
    result=np.zeros((HEADS,D),dtype=np.float64)
    for h,indices in enumerate(VIEW_INDICES):result[h,list(indices)]=1.
    return result

def actor_parameters(parameters: Mapping):
    """Copy the exact eighteen actor arrays into a real-valued float64 model."""
    _require(isinstance(parameters,Mapping),'PARAMETER_MAPPING')
    _require(set(SHAPES).issubset(parameters),'MISSING_ACTOR_PARAMETER')
    allowed=set(SHAPES)|{'masks'}|{f'critic.{l}.{k}' for l in (0,2,4) for k in ('weight','bias')}
    _require(set(parameters).issubset(allowed),'UNKNOWN_PARAMETER')
    if 'masks' in parameters:
        _require(np.array_equal(np.asarray(parameters['masks']),masks()),'FIXED_SOURCE_MASKS')
    result={}
    for name,shape in SHAPES.items():
        x=np.asarray(parameters[name])
        _require(x.shape==shape and x.dtype.kind in 'fi','ACTOR_SHAPE_OR_TYPE:'+name)
        y=np.array(x,dtype=np.float64,copy=True)
        _require(np.isfinite(y).all(),'NONFINITE_ACTOR:'+name)
        result[name]=y
    return result

def softmax(logits):
    x=np.asarray(logits,dtype=np.float64)
    _require(x.shape[-1]==ACTIONS and np.isfinite(x).all(),'SOFTMAX_INPUT')
    e=np.exp(x-x.max(axis=-1,keepdims=True))
    return e/e.sum(axis=-1,keepdims=True)

def softmax_backward(probabilities,upstream):
    p=np.asarray(probabilities,dtype=np.float64);g=np.asarray(upstream,dtype=np.float64)
    _require(p.shape==g.shape and p.shape[-1]==ACTIONS,'SOFTMAX_BACKWARD_SHAPE')
    _require(np.isfinite(p).all() and np.isfinite(g).all(),'SOFTMAX_BACKWARD_FINITE')
    return p*(g-(p*g).sum(axis=-1,keepdims=True))

def _features(features):
    x=np.asarray(features)
    _require(x.ndim in (3,4) and x.shape[-2:]==(ACTIONS,D),'FEATURE_SHAPE')
    _require(x.dtype.kind in 'fi' and np.isfinite(x).all(),'FEATURE_TYPE_OR_FINITE')
    _require(1<=x.shape[-3]<=8 and (x.ndim==3 or 1<=x.shape[0]<=512),'FEATURE_BATCH_OR_AGENTS')
    return np.array(x[None] if x.ndim==3 else x,dtype=np.float64,copy=True),x.ndim==3

def _out(x,single):
    return x[0] if single else x

def forward(parameters,features):
    """Return all head logits, their mean, both probabilities and backprop cache.

    Features are [N,5,23] or [B,N,5,23]. The latter represents B separate
    states. Scalar objectives and gradients average across B when supplied.
    """
    p=actor_parameters(parameters);x,single=_features(features);cache=[];logits=[]
    for h,mask in enumerate(masks()):
        z=np.concatenate((x*mask,np.zeros_like(x)),axis=-1)
        a=np.tanh(z@p[f'heads.{h}.0.weight'].T+p[f'heads.{h}.0.bias'])
        b=np.tanh(a@p[f'heads.{h}.2.weight'].T+p[f'heads.{h}.2.bias'])
        y=(b@p[f'heads.{h}.4.weight'].T+p[f'heads.{h}.4.bias'])[...,0]
        cache.append((z,a,b));logits.append(y)
    heads=np.stack(logits,axis=-2);pooled=heads.mean(axis=-2)
    return {'head_logits':_out(heads,single),'pooled_logits':_out(pooled,single),
            'head_probabilities':_out(softmax(heads),single),
            'probabilities':_out(softmax(pooled),single),
            'model_dtype':'float64_smooth_extension_of_stored_weights',
            '_cache':{'parameters':p,'features':x,'layers':cache,'single':single,
                      'head_logits':heads,'pooled_logits':pooled}}

def _backprop(forward_result,head_logits_gradient):
    c=forward_result['_cache'];p=c['parameters'];g=np.asarray(head_logits_gradient,dtype=np.float64)
    if c['single'] and g.ndim==3:g=g[None]
    _require(g.shape==c['head_logits'].shape and np.isfinite(g).all(),'HEAD_UPSTREAM_SHAPE_OR_FINITE')
    result={}
    for h,(z,a,b) in enumerate(c['layers']):
        delta=g[...,h,:][...,None]
        flat=lambda v:v.reshape(-1,v.shape[-1])
        result[f'heads.{h}.4.weight']=flat(delta).T@flat(b)
        result[f'heads.{h}.4.bias']=flat(delta).sum(axis=0)
        delta=(delta@p[f'heads.{h}.4.weight'])*(1.-b*b)
        result[f'heads.{h}.2.weight']=flat(delta).T@flat(a)
        result[f'heads.{h}.2.bias']=flat(delta).sum(axis=0)
        delta=(delta@p[f'heads.{h}.2.weight'])*(1.-a*a)
        result[f'heads.{h}.0.weight']=flat(delta).T@flat(z)
        result[f'heads.{h}.0.bias']=flat(delta).sum(axis=0)
    return {name:result[name] for name in PARAMETER_ORDER}

def _energy_terms(f,mode):
    _require(mode in ('pci','consensus'),'ORIGINAL_ENERGY_MODE')
    c=f['_cache'];x=c['features'];q=softmax(c['head_logits']);batch,n=x.shape[:2]
    grad=np.zeros_like(q);value=0.;scale=1./(batch*n*len(OVERLAPS))
    for u,v,indices in OVERLAPS:
        difference=q[...,u,:]-q[...,v,:]
        if mode=='consensus':
            value+=float(np.sum(difference*difference))*scale/ACTIONS
            contribution=2.*difference*scale/ACTIONS
        else:
            effect=x[...,list(indices)]
            residual=np.einsum('bna,bnad->bnd',difference,effect)
            value+=float(np.sum(residual*residual))*scale/len(indices)
            contribution=2.*np.einsum('bnd,bnad->bna',residual,effect)*scale/len(indices)
        grad[...,u,:]+=contribution;grad[...,v,:]-=contribution
    return value,softmax_backward(q,grad)

def energy_value(parameters,features,mode='pci'):
    return _energy_terms(forward(parameters,features),mode)[0]

def energy_grad(parameters,features,mode='pci'):
    """Exact original energy mean over edges, features/actions, agents and batch."""
    f=forward(parameters,features);value,g=_energy_terms(f,mode)
    return {'value':value,'mode':mode,'gradient':_backprop(f,g),
            'head_logits_gradient':_out(g,f['_cache']['single']),
            'pooled_logits_gradient':None,
            'scope':'Original auxiliary energy; actual shared actor-parameter gradient, critic/masks excluded.'}

def joint_actions(n_agents):
    """C-order table rows, identical to itertools.product(range(5), repeat=N)."""
    _require(type(n_agents) is int and 1<=n_agents<=4,'EXACT_ENUMERATION_AGENTS_1_TO_4')
    return np.indices((ACTIONS,)*n_agents,dtype=np.int64).reshape(n_agents,-1).T

def risk_from_probabilities(probabilities,cost_table,*,costs_are_normalized=False):
    """Expected table cost and exact derivative w.r.t. pooled action logits.

    Default table values must be RAW integer agent Hamming counts. Division by
    N occurs here exactly once. Already-normalized tables need an explicit flag.
    Cost rows use joint_actions(N). Raw tables may be flat or N-dimensional.
    """
    p=np.asarray(probabilities,dtype=np.float64);single=p.ndim==2
    if single:p=p[None]
    _require(p.ndim==3 and p.shape[-1]==ACTIONS and 1<=p.shape[1]<=4,'RISK_PROBABILITY_SHAPE')
    _require(np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(-1),1.,rtol=0.,atol=1e-12),'RISK_PROBABILITY_SIMPLEX')
    batch,n,_=p.shape;actions=joint_actions(n);count=len(actions)
    raw=np.asarray(cost_table)
    _require(raw.dtype.kind in 'iuf' and np.isfinite(raw).all(),'COST_TABLE_FINITE_NUMERIC')
    if raw.shape in ((count,),(ACTIONS,)*n):raw=np.broadcast_to(raw.reshape(1,count),(batch,count))
    else:
        _require(raw.shape in ((batch,count),(batch,)+(ACTIONS,)*n),'COST_TABLE_SHAPE')
        raw=raw.reshape(batch,count)
    raw=raw.astype(np.float64,copy=False)
    if costs_are_normalized:
        _require((raw>=0).all() and (raw<=1).all(),'NORMALIZED_COST_BOUNDS')
        cost=raw;raw_cost=raw*n
    else:
        _require((raw>=0).all() and (raw<=n).all() and np.array_equal(raw,np.rint(raw)),'RAW_INTEGER_HAMMING_COST')
        cost=raw/n;raw_cost=raw
    product=np.ones((batch,count),dtype=np.float64)
    selected=[]
    for i in range(n):
        qi=p[:,i,actions[:,i]];selected.append(qi);product*=qi
    values=(product*cost).sum(axis=1);raw_values=(product*raw_cost).sum(axis=1)
    conditional=np.empty_like(p)
    for i in range(n):
        other=np.ones_like(product)
        for j in range(n):
            if j!=i:other*=selected[j]
        for a in range(ACTIONS):
            use=actions[:,i]==a;conditional[:,i,a]=(other[:,use]*cost[:,use]).sum(axis=1)
    constant=np.all(cost==cost[:,[0]],axis=1)
    values[constant]=cost[constant,0];raw_values[constant]=raw_cost[constant,0]
    g=p*(conditional-values[:,None,None])/batch
    # Constant costs are exactly policy independent, including boundary simplexes.
    g[constant]=0.
    return {'value':float(values.mean()),'raw_expected_cost':float(raw_values.mean()),
            'state_values':_out(values,single),'state_raw_expected_costs':_out(raw_values,single),
            'pooled_logits_gradient':_out(g,single),'conditional_normalized_cost':_out(conditional,single),
            'joint_probabilities':_out(product,single),'n_agents':n,
            'normalization':'RAW_HAMMING_DIVIDED_BY_N_EXACTLY_ONCE' if not costs_are_normalized else 'EXPLICIT_ALREADY_NORMALIZED_COST'}

def risk_value(parameters,features,cost_table,*,costs_are_normalized=False):
    return risk_from_probabilities(forward(parameters,features)['probabilities'],cost_table,
                                   costs_are_normalized=costs_are_normalized)['value']

def risk_grad(parameters,features,cost_table,*,costs_are_normalized=False):
    f=forward(parameters,features);r=risk_from_probabilities(f['probabilities'],cost_table,costs_are_normalized=costs_are_normalized)
    pooled=r['pooled_logits_gradient'];g=np.repeat(pooled[...,None,:],HEADS,axis=-2)/HEADS
    r.update(gradient=_backprop(f,g),head_logits_gradient=g,
             scope='Conditional fixed-state exact product-policy expected Hamming risk; not a full return or occupancy gradient.')
    return r

def gradient_dot(left,right):
    _require(set(left)==set(right)==set(PARAMETER_ORDER),'ACTOR_GRADIENT_KEYS')
    total=0.
    for name in PARAMETER_ORDER:
        a=np.asarray(left[name],dtype=np.float64);b=np.asarray(right[name],dtype=np.float64)
        _require(a.shape==b.shape==SHAPES[name] and np.isfinite(a).all() and np.isfinite(b).all(),'ACTOR_GRADIENT_BODY:'+name)
        total+=float(np.vdot(a,b))
    return total

def gradient_inner_products(left,right,*,norm_threshold=NORM_THRESHOLD):
    _require(math.isfinite(norm_threshold) and norm_threshold>=0,'NORM_THRESHOLD')
    dot=gradient_dot(left,right);ln=math.sqrt(max(0.,gradient_dot(left,left)));rn=math.sqrt(max(0.,gradient_dot(right,right)))
    defined=ln>norm_threshold and rn>norm_threshold
    return {'dot':dot,'left_norm':ln,'right_norm':rn,
            'cosine':max(-1.,min(1.,dot/(ln*rn))) if defined else None,
            'cosine_defined':defined,'norm_threshold':norm_threshold,
            'parameter_count':sum(math.prod(s) for s in SHAPES.values())}

def unit_direction(gradient,*,negative=True,norm_threshold=NORM_THRESHOLD):
    _require(type(negative) is bool and math.isfinite(norm_threshold) and norm_threshold>=0,'UNIT_DIRECTION_CONFIGURATION')
    norm=math.sqrt(max(0.,gradient_dot(gradient,gradient)))
    if norm<=norm_threshold:return None
    return {k:((-1. if negative else 1.)/norm)*np.asarray(gradient[k],dtype=np.float64) for k in PARAMETER_ORDER}

def perturb(parameters,direction,scale):
    _require(math.isfinite(scale),'FINITE_PERTURBATION')
    p=actor_parameters(parameters);gradient_dot(direction,direction)
    return {k:p[k]+scale*np.asarray(direction[k],dtype=np.float64) for k in PARAMETER_ORDER}

def directional_check(parameters,features,direction,*,cost_table=None,mode='pci',delta=1e-4):
    """Central finite difference of risk (when supplied) or original energy.

    This does not choose a step size adaptively. The primary study fixes delta
    to 1e-4 and the direction to a declared unit actor-parameter gradient.
    """
    _require(math.isfinite(delta) and delta>0,'POSITIVE_FIXED_DELTA')
    if direction is None:return {'status':'UNDEFINED_NULL_DIRECTION','delta':delta,'analytic':None,'finite_difference':None}
    norm=math.sqrt(max(0.,gradient_dot(direction,direction)))
    _require(norm>NORM_THRESHOLD,'NONZERO_DIRECTION')
    if cost_table is None:
        result=energy_grad(parameters,features,mode);value=lambda p:energy_value(p,features,mode)
    else:
        result=risk_grad(parameters,features,cost_table);value=lambda p:risk_value(p,features,cost_table)
    plus=value(perturb(parameters,direction,delta));minus=value(perturb(parameters,direction,-delta))
    fd=(plus-minus)/(2*delta);analytic=gradient_dot(result['gradient'],direction)
    return {'status':'DEFINED','delta':delta,'direction_norm':norm,'base':result['value'],
            'plus':plus,'minus':minus,'analytic':analytic,'finite_difference':fd,
            'absolute_error':abs(fd-analytic)}
