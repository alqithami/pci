"""Trace-normalized overlap energies and a matched cost critic.

Q preserves the constant-action vector and every singular value of the centered
semantic effect map. Thus the random-map control changes the projected action
semantics, not the state-wise spectrum/scale. This does NOT equalize realized
policy gradients; their norms are logged separately.
"""
import numpy as np
import torch
from torch import nn
from pci_bench.model import ContextPolicy,mlp
from pci_bench.environment import D,OVERLAPS
METHODS=('penalty','consensus_norm','pci_norm','random_norm','ppo_lagrangian')

def orthogonal_action_maps():
    rng=np.random.default_rng(314159)
    # Deterministic orthonormal basis of the zero-sum subspace.
    basis=np.eye(5)[:,:4]-np.eye(5)[:,[4]]
    U=np.linalg.qr(basis)[0]
    J=np.ones((5,5))/5
    return np.stack([J+U@np.linalg.qr(rng.standard_normal((4,4)))[0]@U.T for _ in OVERLAPS])

Q_ARRAY=orthogonal_action_maps()

class Policy(ContextPolicy):
    def __init__(self,method='pci_norm',hidden=64):
        if method not in METHODS:raise ValueError(method)
        super().__init__('pci',hidden)
        self.followup_method=method
        # Present and fitted in every method; identical total parameter count.
        self.cost_critic=mlp(10*D,1,hidden)
        self.register_buffer('random_action_maps',torch.as_tensor(Q_ARRAY,dtype=torch.float32))
    def forward(self,features):
        dist,value,probs=super().forward(features)
        pooled=features.mean(1,keepdim=True).expand_as(features)
        cost_value=self.cost_critic(torch.cat((features.flatten(-2),pooled.flatten(-2)),-1)).squeeze(-1)
        return dist,value,cost_value,probs

def centered_maps(features):
    result=[]
    for _,_,idx in OVERLAPS:
        F=features[...,list(idx)]
        result.append(F-F.mean(-2,keepdim=True))
    return result

def energy(probs,features,mode,Q=None):
    if mode not in METHODS:raise ValueError(mode)
    if mode in ('penalty','ppo_lagrangian'):return probs[...,0,0]*0
    total=torch.zeros_like(probs[...,0,0])
    for k,((u,v,_),F) in enumerate(zip(OVERLAPS,centered_maps(features))):
        diff=probs[...,u,:]-probs[...,v,:]
        if mode=='consensus_norm':term=diff.square().sum(-1)/4
        else:
            if mode=='random_norm':
                q=Q[k] if Q is not None else torch.as_tensor(Q_ARRAY[k],device=F.device,dtype=F.dtype)
                F=torch.einsum('ab,...bd->...ad',q,F)
            norm=F.square().sum((-2,-1))
            projected=torch.einsum('...a,...ad->...d',diff,F)
            term=projected.square().sum(-1)/norm.clamp_min(1e-12)
            term=torch.where(norm>1e-12,term,torch.zeros_like(term))
        total=total+term
    return total/len(OVERLAPS)

def dual_update(value,mean_cost,budget,rate,maximum):
    if not np.isfinite([value,mean_cost,budget,rate,maximum]).all():raise ValueError('Nonfinite dual input')
    if min(value,mean_cost,budget,rate,maximum)<0:raise ValueError('Negative dual input')
    return float(np.clip(value+rate*(mean_cost-budget),0,maximum))
